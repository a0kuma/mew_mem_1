# TorchGPipe GPU-to-GPU Copy Locations

This note lists where tensor data is transferred between GPUs for forward and backward passes in the local `torchgpipe` code.

## Forward pass (activations to next GPU)

- The actual device transfer happens in `Copy.forward` via `x.to(get_device(next_stream))`: [torchgpipe/copy.py](torchgpipe/copy.py#L41).
- The forward transfer is inserted into the pipeline by `Pipeline.fence()` when moving from partition `j-1` to `j`: [torchgpipe/pipeline.py](torchgpipe/pipeline.py#L144).
- The helper that wires in the autograd-aware copy op (so it creates a backward edge) is here: [torchgpipe/pipeline.py](torchgpipe/pipeline.py#L41-L42).

## Backward pass (gradients to previous GPU)

- Gradients are moved back to the previous device in `Copy.backward` via `x.to(get_device(prev_stream))`: [torchgpipe/copy.py](torchgpipe/copy.py#L64).
- This backward copy is triggered automatically by autograd because the forward path used `Copy.apply`: [torchgpipe/pipeline.py](torchgpipe/pipeline.py#L41-L42).
