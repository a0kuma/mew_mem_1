import torch
import torch.nn as nn

if not torch.cuda.is_available():
	raise SystemExit("CUDA is required for this demo.")

torch.cuda.memory._record_memory_history()

# 只使用 GPU 0
device = torch.device("cuda:0")

print(f"初始 GPU 記憶體: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

# 1. 建立一個稍微大一點的假模型 (假設有 1000 萬個參數)
# 10,000,000 * 4 bytes (float32) 約等於 38.1 MB
model = nn.Linear(10000, 1000).to(device)
print(f"載入模型後 GPU 記憶體: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

# 2. 宣告 AdamW Optimizer (這就是你問的那一行！)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
print(f"宣告 AdamW 後 GPU 記憶體: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
# 👆 你會發現，記憶體幾乎沒有增加！

# 3. 模擬一次 Forward 與 Backward
dummy_input = torch.randn(16, 10000).to(device)
output = model(dummy_input)
loss = output.sum()
loss.backward()
print(f"Backward 後 (產生梯度) GPU 記憶體: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
# 👆 記憶體會增加，因為每個參數 p 都多了一個 p.grad (梯度)

# 4. 關鍵時刻：執行第一次 Optimizer Step
optimizer.step()
print(f"執行第一次 optimizer.step() 後 GPU 記憶體: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
# 👆 這裡記憶體會暴增！因為 Adam 分配了 exp_avg 和 exp_avg_sq (2倍的模型大小)

snapshot_file = "gpu_memory_snapshot-adam.pickle"
torch.cuda.synchronize()
torch.cuda.memory._dump_snapshot(snapshot_file)

# 5. 清除梯度 (set_to_none) 後再存一份
optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize()
snapshot_file_set_to_none = "do_the_set_to_none.pickle"
torch.cuda.memory._dump_snapshot(snapshot_file_set_to_none)

torch.cuda.memory._record_memory_history(enabled=None)