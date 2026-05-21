import torch
import time
from torchviz import make_dot

# ==========================================
# 0. 自訂的 Dummy 節點 (拉長公路用)
# ==========================================
class OPdummyA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        print("🏃 [Dummy A - Forward] 通過")
        return input_tensor * 1.0  # 稍微做點運算

    @staticmethod
    def backward(ctx, grad_input):
        print(f"[{time.time():.4f}] ↩️ [Dummy A - Backward] 執行完了！")
        return grad_input


class OPdummyB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor1, input_tensor2):
        print("🏃 [Dummy B - Forward] 兩條路匯合相加")
        return input_tensor1 + input_tensor2

    @staticmethod
    def backward(ctx, grad_input):
        print("b backward")
        # ⚠️ 關鍵修正 1：前向吃 2 個，反向就要吐出 2 個梯度！
        return grad_input, grad_input

    
# ==========================================
# 1. 定義 Fork（被卡住的苦主）
# ==========================================
class OpFork(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        phony = torch.empty(0, requires_grad=False)
        print("fork...")
        return input_tensor.detach(), phony.detach()

    @staticmethod
    def backward(ctx, grad_input, grad_phony):
        print("fork backward")
        return grad_input

# ==========================================
# 2. 定義 Join（故意拖時間的大牌）
# ==========================================
class OpJoin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, phony):
        print("join...")
        return input_tensor.detach()

    @staticmethod
    def backward(ctx, grad_input):
        print("join backward")
        return grad_input, None
    

# ==========================================
# 3. 實地建立兩條公路與十字路口
# ==========================================
l1_x = torch.randn(1, requires_grad=True)
l2_x = torch.randn(1, requires_grad=True)

print("--- 🏃 啟動前向傳播 (Forward) ---")

# ⚠️ 關鍵修正 3：要把 OPdummyA 算完的東西接住！
lane1_x = OPdummyA.apply(l1_x)
lane2_x = OPdummyA.apply(l2_x)

# Lane 1 通過 Fork
lane1_y, phony = OpFork.apply(lane1_x)

# Lane 2 通過 Join，綁定 phony
lane2_y = OpJoin.apply(lane2_x, phony)

# 把兩條路匯合 (當作總 Loss)
# ⚠️ 關鍵修正 2：要加上 .apply
loss = OPdummyB.apply(lane1_y, lane2_y)

# 畫圖
graph = make_dot(loss, params={"lane1_x": lane1_x, "lane2_x": lane2_x})
graph.render("wait_experiment_graph", format="png")
print("🎉 成功生成 'wait_experiment_graph.png'！")

print("\n--- ↩️ 點燃引信：啟動反向傳播 (loss.backward) ---")
loss.backward()
print("--- 🎉 實驗結束 ---")