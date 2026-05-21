# Deterministic clock-cycle (torchgpipe)

## Core schedule generation
- [external/torchgpipe/pipeline.py](torchgpipe/pipeline.py#L49-L65) defines `clock_cycles(m, n)` and iterates `for k in range(m+n-1)` to emit all `(i, j)` pairs for each clock tick.
  - Code uses 0-based indices: $i + j = k$ because `i = k - j` in `yield [(k-j, j) ...]`.
  - Converting to 1-based indexing gives the paper form: $i' + j' - 1 = k'$ with $i' = i+1$, $j' = j+1$, $k' = k+1$.

## Clock tick execution order
- [external/torchgpipe/pipeline.py](torchgpipe/pipeline.py#L113-L115) runs `for schedule in clock_cycles(m, n)` and then calls `self.fence(...)` followed by `self.compute(...)` for each tick.

## Per-tick copy/compute
- [external/torchgpipe/pipeline.py](torchgpipe/pipeline.py#L117-L143) `fence()` loops over each `(i, j)` in the tick and issues copy and dependency setup.
- [external/torchgpipe/pipeline.py](torchgpipe/pipeline.py#L144-L222) `compute()` loops over each `(i, j)` in the tick and issues compute tasks (with optional checkpointing).
