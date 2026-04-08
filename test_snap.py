import torch
for i in range(4):
    torch.cuda.memory._record_memory_history(enabled='all', max_entries=100)
    with torch.cuda.device(i):
        x = torch.randn(10, device=f"cuda:{i}")
print("Keys in snapshot:", [d["device"] for d in torch.cuda.memory._snapshot()["device_traces"]])
