# Checkpoint Line-Range Filters

The two numeric ranges in the classifier are not arbitrary. They map to the two different `backward` implementations in the installed torchgpipe checkpoint code.

- `258-273`: `Checkpoint.backward(...)` in torchgpipe. This is the step that kicks off the final backward for a micro-batch by calling `torch.autograd.backward(...)` on the recomputed outputs. Allocations here are labeled **H** (Final Backward Tensors).
- `295-308`: `Recompute.backward(...)` in torchgpipe. This is the recompute path that runs the forward under `enable_recomputing()` to rebuild activations before the final backward. Allocations here are labeled **G** (Recomputed Activations).

Why the classifier uses line ranges

The memory snapshot stack frames include only `filename`, `line`, and `name`, and the `name` field is just `backward` for both functions. The line numbers are the reliable way to distinguish which `backward` implementation produced the allocation.

When to update these ranges

If you upgrade torchgpipe, the line numbers may shift. Recompute the ranges by locating the two `backward` methods in the installed torchgpipe checkpoint module and updating the bounds in the classifier.