import torch
import torch.nn as nn
from torchviz import make_dot

# ==========================================
# 1. 定義論文中的 Fork 和 Join 隱形紅線元件
# ==========================================
class Fork(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        # 變出一個沒有重量、不佔空間的虛擬幽靈 (phony)
        phony = torch.empty(0, device=input_tensor.device, requires_grad=False)
        return input_tensor.detach(), phony.detach()

    @staticmethod
    def backward(ctx, grad_input, grad_phony):
        # 反向傳播時，強迫等待兩路人馬會合，並把梯度傳回
        return grad_input

class Join(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, phony):
        return input_tensor.detach()

    @staticmethod
    def backward(ctx, grad_input):
        # 把梯度傳回主線，幽靈線傳回 None
        return grad_input, None

def depend(fork_from_tensor, join_to_tensor):
    # 用虛擬幽靈 (phony) 把兩個原本無關的張量綁在一起
    fork_from_tensor, phony = Fork.apply(fork_from_tensor)
    join_to_tensor = Join.apply(join_to_tensor, phony)
    return fork_from_tensor, join_to_tensor

# ==========================================
# 2. 模擬一幅超簡單的 2 關卡、2 微批次流水線
# ==========================================
# 假設我們有 2 個微批次 (MB0, MB1)
mb0_in = torch.randn(2, 4, requires_grad=True)
mb1_in = torch.randn(2, 4, requires_grad=True)

# 模擬神經網路的兩大關卡 (Layer 1, Layer 2)
layer1 = nn.Linear(4, 4)
layer2 = nn.Linear(4, 4)

print("--- 開始跑前向傳播 (Forward) ---")

# Step 1: 跑第一批次 (MB0)
mb0_layer1 = layer1(mb0_in)
mb0_out = layer2(mb0_layer1)  # MB0 順利抵達終點

# Step 2: 跑第二批次 (MB1)
mb1_layer1 = layer1(mb1_in)

# 🌟 關鍵核心：在 MB1 算最後一層前，強行在計算圖上插上 depend 柵欄！
# 規定：MB0 必須在反向傳播時，排在 MB1 的後面 (即 MB1 先算反向，才換 MB0)
mb1_layer1, mb0_out = depend(mb1_layer1, mb0_out)
#mb0_out ,mb1_layer1= depend( mb0_out,mb1_layer1)

mb1_out = layer2(mb1_layer1)  # MB1 接著抵達終點

# 把兩個微批次的輸出加起來，當成最終的總 Loss
total_loss = mb0_out.sum() + mb1_out.sum()

print("--- 前向傳播結束，正在傾印計算圖... ---")

# ==========================================
# 3. 傾印計算圖（關鍵現場！）
# ==========================================
# 我們把 total_loss 丟進去，把這張剛畫好的地圖印成 PNG
graph = make_dot(total_loss, params={
    'mb0_in': mb0_in, 
    'mb1_in': mb1_in,
    'L1_weight': layer1.weight,
    'L2_weight': layer2.weight,
    'L1_bias': layer1.bias,
    'L2_bias': layer2.bias
})

# 儲存圖片
graph.render("mini_gpipe_graph", format="png")
print("🎉 成功生成 'mini_gpipe_graph.png' 圖片！")

# 4. 真正發動 C++ 引擎去算它
print("--- 點燃引信：發動反向傳播 ---")
total_loss.backward()
print("完工！")