# True FFN Parallelism for UMA — Implementation Plan

## Context

PowerInfer on UMA/Metal has 3 execution modes for sparse FFN layers, but none truly
parallelizes the full FFN. The existing "parallel" mode (`UMA_PARALLEL=1`) only
parallelizes the AXPY (down projection). The expensive up/gate sparse MUL_MAT
projections always run on CPU alone.

**Goal**: Run ALL three FFN projections (up, gate, down) on CPU and GPU simultaneously,
each processing a disjoint set of neurons via gpu_idx partitioning.

**Key asymmetry found**: `llm_build_sparse_mul_mat` passes `NULL` for gpu_index on
Metal UMA (`llama.cpp:5341`), so CPU processes ALL rows. Meanwhile,
`llm_build_sparse_axpy` passes `gpu_index` (`llama.cpp:5398`). The Metal
MUL_MAT_SPARSE kernel also lacks gpu_idx filtering — only has sparse_threshold.

---

## Phase 0: Pre-requisite Bug Fixes

### Fix A1: Hardcoded Python path
**Severity**: CRITICAL  
**File**: `llama.cpp:3586`

Current:
```cpp
command_ss << "/Users/deeveshizm/miniconda3/bin/python3 -m powerinfer"
```
Fix:
```cpp
command_ss << "python3 -m powerinfer"
```
Revert to PATH-based resolution.

---

### Fix A3: Metal AXPY Q4_0 activation mismatch
**Severity**: CRITICAL  
**File**: `ggml-metal.metal` ~line 3307

Metal Q4_0 AXPY reads `src1[row]` as raw float. CPU uses Q8_0-quantized activations
(`nerual[bid].qs[qsid] * nerual[bid].d`). In parallel mode where CPU and GPU results
merge, this numerical divergence produces wrong results.

**Approach**: Align both sides to use float activations (simpler, more accurate on UMA
where bandwidth is shared).

---

### Fix A4: Data race in parallel mode temp buffer
**Severity**: HIGH  
**File**: `ggml-metal.m:2044-2046`

`memset(ggml_metal_parallel_temp_data, 0, ...)` happens during Metal encoder setup on
the Metal thread, concurrent with CPU AXPY on a different buffer. While they target
different buffers, the temp buffer zeroing timing needs explicit ordering.

**Approach**: Move `memset` to the main thread before launching the Metal thread.

---

### Fix A5: Sequential mode wasteful CPU AXPY
**Severity**: CRITICAL  
**File**: `llama.cpp:7842-7858`

In `uma_gpu_axpy` mode: runs Metal AXPY, saves result, runs CPU AXPY "for side effects",
then discards CPU output. Labeled "EXPERIMENT".

**Approach**: Remove the entire experimental block.

---

### Fix A6: UMA profiling code in production
**Severity**: MEDIUM  
**File**: `ggml-metal.m:2376-2426`

Timing/logging in `set_tensor_async`/`get_tensor_async` prints to stderr every 100MB.
Static counters not thread-safe.

**Approach**: Remove the profiling blocks.

---

### Fix A2: Metal AXPY serial loop (performance)
**Severity**: HIGH  
**File**: `ggml-metal.metal:3265-3315`

Both F16 and Q4_0 AXPY kernels iterate all ne01 rows per thread (O(ne01) serial).
GPU slower than CPU for large models (11k+ rows).

**Approach**: Defer to after Phase 1 — performance optimization, not correctness fix.
Restructure to parallelize across rows using threadgroups.

---

### Fix A7: Missing Q4_0 sparse kernel
**Severity**: CRITICAL (for SmallThinker)  
**File**: `ggml-metal.m:1949-1968`, `ggml-metal.metal`

MUL_MAT_SPARSE only supports Q4_K and F16. SmallThinker uses Q4_0 exclusively.

**Approach**: Not a blocker for root-level PowerInfer (uses Q4_K). Defer to SmallThinker
Metal integration phase.

---

## Phase 1: True FFN Parallelism

### Step 1: Pass gpu_index in MUL_MAT_SPARSE graph building for UMA
**File**: `llama.cpp:5334-5344`

Change `llm_build_sparse_mul_mat` Metal path:
```cpp
// Before:
out = ggml_mul_mat_idx(ctx, up, inp, idx, NULL);
// After:
out = ggml_mul_mat_idx(ctx, up, inp, idx, gpu_index);
```

Makes `dst->src[3]` carry gpu_idx for MUL_MAT_SPARSE ops. When `gpu_index == NULL` (no
split file), behavior unchanged. Existing modes with `has_gpu_idx=0` ignore it.

---

### Step 2: Add gpu_idx filtering to Metal MUL_MAT_SPARSE kernels
**Files**: `ggml-metal.metal`, `ggml-metal.m`

**Shader changes** — modify 3 kernels:
- `kernel_mul_mv_q4_K_f32_sparse` (two variants, lines ~2949 and ~3070)
- `kernel_mul_mv_f16_f32_sparse` (line ~3173)

Add parameters:
```metal
device const int * gpu_idx [[buffer(20)]],
constant int & has_gpu_idx [[buffer(21)]]
```

Add filtering next to existing sparse_threshold check:
```metal
if (has_gpu_idx && gpu_idx[first_row + row] == 0) {
    // Skip cold neuron — CPU handles it
    continue;
}
```

**Host dispatch** (`ggml-metal.m:1970-2001`):
- Bind `dst->src[3]` (gpu_idx) to buffer index 20
- Set `has_gpu_idx = 1` only when src[3] is non-NULL AND `ggml_metal_full_ffn_parallel`
- When `has_gpu_idx == 0`, kernel processes all rows — backward compatible

---

### Step 3: Add Metal control flag
**Files**: `ggml-metal.m` (lines 150-181), `ggml-metal.h`

```c
bool ggml_metal_full_ffn_parallel = false;
void ggml_metal_set_full_ffn_parallel(bool enable);
```

---

### Step 4: Update Metal op-filtering logic
**File**: `ggml-metal.m` (node skip logic ~lines 1002-1013 and ~2211-2218)

When `ggml_metal_full_ffn_parallel == true`:
- Process `GGML_OP_MUL_MAT_SPARSE` (hot neurons via gpu_idx)
- Skip `GGML_OP_AXPY` (separate parallel phase)
- Skip element-wise ops (UNARY, MUL) — CPU handles after join

---

### Step 5: New `UMA_FULL_PARALLEL` execution path
**File**: `llama.cpp:7663-7873`

New env var `UMA_FULL_PARALLEL=1`. Separate `else if` branch before existing
`uma_parallel`. Reuses existing infrastructure.

```
Phase 1: Metal dense ops (attention, norms, predictor)     [existing]

Phase 2a — PARALLEL MUL_MAT_SPARSE:
  Metal thread: MUL_MAT_SPARSE(gate+up), gpu_idx filters to hot rows
  CPU:          MUL_MAT_SPARSE(gate+up), skips hot rows (ggml.c:14258)
  [std::thread::join — memory fence]

Phase 2b — Element-wise ops (UNARY/ReLU, MUL):
  CPU only, sequential. Both hot+cold rows populated. Cheap ops.

Phase 2c — PARALLEL AXPY:
  Metal thread: AXPY hot neurons -> temp buffer  [reuse parallel_mode]
  CPU:          AXPY cold neurons -> dst
  [std::thread::join]

Phase 3: Merge: dst += temp                                [reuse existing]
Phase 4: CPU post-AXPY (ADD/residual)                      [existing]
```

**Guard**: Only enter when `layer.gpu_idx != NULL && 0 < gpu_offload_ratio < 1.0`.
Falls back to existing modes otherwise.

---

### Step 6: Debug/timing instrumentation
Gate behind `UMA_FULL_PARALLEL_DEBUG=1`:
- Per-phase wall-clock time
- Hot vs cold neuron counts per layer
- Total FFN time comparison vs existing modes

---

### Step 7: Document findings
Update docs md files with:
- Execution flow diagram + code snippets
- Timing results from debug mode
- Design reasoning (row-partitioned split, element-wise on CPU, etc.)

---

## No Regression Guarantee

- Phase 0 fixes are targeted, don't change execution flow for working modes
- Phase 1 is gated behind `UMA_FULL_PARALLEL=1` as a completely separate branch
- Existing modes (`UMA_CPU_ONLY`, `UMA_PARALLEL`, `UMA_GPU_AXPY`, default) untouched

## Verification Steps

1. `make -j LLAMA_METAL=1`
2. Test existing modes still work: `UMA_CPU_ONLY=1`, `UMA_PARALLEL=1`, `UMA_GPU_AXPY=1`
3. Test `UMA_FULL_PARALLEL=1` output matches `UMA_CPU_ONLY=1` baseline
4. Run `UMA_FULL_PARALLEL=1 UMA_FULL_PARALLEL_DEBUG=1` for timing data
5. Compare token/s across all modes
6. Test with and without gpu_idx split file for fallback behavior
