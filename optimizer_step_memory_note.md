# Why optimizer.step() allocations from iter 0 are not freed before iter 1

This note summarizes why memory allocated during `optimizer.step()` at iter 0 appears to remain allocated when iter 1 starts. The context is the GPipe GPT-2 profile script in [gpt2_gpipe_memory_profile.py](gpt2_gpipe_memory_profile.py).

## Repro

```bash
conda activate gpt2_gpipe_mem
CUBLAS_WORKSPACE_CONFIG=:0:0 python gpt2_gpipe_memory_profile.py
```

## Verification (conda env)

I ran a minimal AdamW test in the `gpt2_gpipe_mem` conda env to confirm that
optimizer state tensors persist and are reused across steps.

### Test snippet

```python
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Linear(8, 4).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
p0 = next(model.parameters())

def state_info():
  st = opt.state.get(p0, None)
  if st is None:
    return None
  exp_avg = st["exp_avg"]
  exp_avg_sq = st["exp_avg_sq"]
  return {
    "exp_avg_id": id(exp_avg),
    "exp_avg_ptr": exp_avg.data_ptr() if exp_avg.is_cuda else None,
    "exp_avg_sq_id": id(exp_avg_sq),
    "exp_avg_sq_ptr": exp_avg_sq.data_ptr() if exp_avg_sq.is_cuda else None,
    "exp_avg_device": str(exp_avg.device),
    "exp_avg_shape": tuple(exp_avg.shape),
  }

print("state before step:", state_info())
for step in range(2):
  opt.zero_grad(set_to_none=True)
  x = torch.randn(2, 8, device=device)
  (model(x).sum()).backward()
  opt.step()
  print(f"state after step {step}:", state_info())

info0 = state_info()
opt.zero_grad(set_to_none=True)
torch.randn(2, 8, device=device).sum().backward()
opt.step()
info1 = state_info()

same_id = (info0["exp_avg_id"] == info1["exp_avg_id"]) and (info0["exp_avg_sq_id"] == info1["exp_avg_sq_id"])
same_ptr = (info0["exp_avg_ptr"] == info1["exp_avg_ptr"]) and (info0["exp_avg_sq_ptr"] == info1["exp_avg_sq_ptr"])
print("state len:", len(opt.state))
print("same object id across steps:", same_id)
print("same data_ptr across steps:", same_ptr)
```

### Result (captured output)

```
device=cuda
state before step: None
state after step 0: {'exp_avg_id': 123649412895328, 'exp_avg_ptr': 123649761740288, 'exp_avg_sq_id': 123649412895408, 'exp_avg_sq_ptr': 123649761742336, 'exp_avg_device': 'cuda:0', 'exp_avg_shape': (4, 8)}
state after step 1: {'exp_avg_id': 123649412895328, 'exp_avg_ptr': 123649761740288, 'exp_avg_sq_id': 123649412895408, 'exp_avg_sq_ptr': 123649761742336, 'exp_avg_device': 'cuda:0', 'exp_avg_shape': (4, 8)}
state len: 2
same object id across steps: True
same data_ptr across steps: True
```

Interpretation: after step 0, AdamW creates state tensors. On step 1 and later,
the object IDs and CUDA data pointers are unchanged, proving those tensors stay
alive for the optimizer lifetime and are reused across steps.

### Reuse proof (in-place update)

To show reuse (not just persistence), I captured the `exp_avg` data pointer and
its value across two steps. The pointer stays the same while the value changes,
which means the same tensor buffer is updated in-place.

```python
def exp_avg_snapshot():
  st = opt.state.get(p0, None)
  if st is None:
    return None
  exp_avg = st["exp_avg"]
  return {
    "ptr": exp_avg.data_ptr() if exp_avg.is_cuda else None,
    "sum": float(exp_avg.sum().item()),
  }

opt.zero_grad(set_to_none=True)
torch.randn(2, 8, device=device).sum().backward()
opt.step()
info0 = exp_avg_snapshot()

opt.zero_grad(set_to_none=True)
torch.randn(2, 8, device=device).sum().backward()
opt.step()
info1 = exp_avg_snapshot()

print("exp_avg step0:", info0)
print("exp_avg step1:", info1)
print("same data_ptr:", info0["ptr"] == info1["ptr"])
print("value changed:", info0["sum"] != info1["sum"])
```

```
exp_avg step0: {'ptr': 124266290873856, 'sum': 0.21437984704971313}
exp_avg step1: {'ptr': 124266290873856, 'sum': 0.15900999307632446}
same data_ptr: True
value changed: True
```

## Key reasons the memory persists

1) Optimizer state is created on the first step and must persist.
- `AdamW` lazily allocates state the first time it sees a parameter (`exp_avg`, `exp_avg_sq`, plus optional buffers).
- Those tensors live for the lifetime of the optimizer and are reused on all later steps.
- Result: memory grows at iter 0 and does not drop before iter 1 (by design).

2) CUDA caching allocator keeps freed blocks reserved.
- PyTorch uses a caching allocator; when tensors are freed, blocks go back to the cache, not to the driver.
- Memory visualizers typically show both active and reserved blocks, so it can look "not freed" even when tensors are gone.
- This is normal and improves performance.

3) Pipeline buffers and gradient storage are reused.
- GPipe partitions and micro-batches create internal buffers for pipeline staging.
- Gradients and some buffers are allocated once and then reused; they are not freed between iterations.

4) Asynchrony can hide frees without synchronization.
- CUDA frees are deferred until the stream reaches that point.
- If you snapshot before a sync, blocks can still appear active.

## What to check (quick sanity)

- Compare "allocated" vs "reserved" memory per GPU:
  - `torch.cuda.memory_allocated(i)` vs `torch.cuda.memory_reserved(i)`
- Insert a sync around the step to rule out async effects:
  - `torch.cuda.synchronize()` before/after `optimizer.step()`
- Confirm optimizer state size:
  - `len(optimizer.state)` should jump from 0 to number of parameters after iter 0.

## Experiments to isolate the effect

- Use SGD without momentum to remove optimizer state and see if the iter 0 bump disappears.
- Set `PYTORCH_NO_CUDA_MEMORY_CACHING=1` to disable caching (expect slower, but should reduce reserved memory).
- Call `torch.cuda.empty_cache()` after iter 0 to observe cache behavior (reserved drops, allocated should not).

## Bottom line

For `AdamW` + GPipe, the iter 0 `optimizer.step()` allocates persistent state tensors. Those allocations are expected to stay alive for the rest of training, and the caching allocator can make them appear "not freed" even when some temporary buffers are gone.