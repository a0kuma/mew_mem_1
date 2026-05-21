import torch
import torch.nn as nn
from torchviz import make_dot
import json

# 1. 建立一個最簡單的 2 層線性網路 (線性代數的矩陣相乘 Y = XW + B)
model = nn.Sequential(
    nn.Linear(3, 4),  # 第一層：把 3 維資料變成 4 維
    nn.ReLU(),        # 激活函數：把負數變成 0
    nn.Linear(4, 2)   # 第二層：把 4 維資料變成 2 維（預報分類）
)

# 2. 給一組假資料（代表 1 筆 training data）
inputs = torch.randn(1, 3)
labels = torch.tensor([[1.0, 0.0]])  # 答案標籤

print("--- 1. 執行前向傳播 (Forward Pass) ---")
outputs = model(inputs)

# 3. 算損失值 (Loss)
criterion = nn.MSELoss()
loss = criterion(outputs, labels)

print("--- 2. 前向傳播結束，當場畫出計算圖 ---")
# 這裡我們把 loss 丟給 torchviz，它會順著 loss.grad_fn 一路往回挖祖先
print(dict(model.named_parameters()))
graph = make_dot(loss, params=dict(model.named_parameters()))

# 儲存圖片
graph.render("pure_pytorch_graph", format="png")
print("🎉 成功生成 'pure_pytorch_graph.png'！快打開看看！")

print("--- 3. 點燃引信，執行反向傳播 (Backward Pass) ---")
# 這一行會呼叫 C++ 引擎，看著這張地圖倒著走回來，算好每個 Weights 的梯度
loss.backward()
print("完工！")