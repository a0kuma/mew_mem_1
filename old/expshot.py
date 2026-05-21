import torch
import time


# ==========================================
# 1. 準備資料：在 GPU 記憶體中建立兩個 2x2 矩陣
# ==========================================
A = torch.randn(2, 2, device='cuda')
B = torch.randn(2, 2, device='cuda')

# ==========================================
# 2. 開闢專屬車道（對應 torchgpipe 的 new_stream）
# ==========================================
# 我們向 GPU 顯示卡申請一條全新的獨立指令佇列
my_custom_stream = torch.cuda.Stream()

print(f"[{time.time():.5f}] 🚦 主程式：準備切換車道...")

# ==========================================
# 3. 切換車道並發射指令（對應 torchgpipe 的 with use_stream）
# ==========================================
with torch.cuda.stream(my_custom_stream):
    print(f"[{time.time():.5f}] 🚗 [專屬車道] 進入 Context Manager！")
    
    # 🌟 這裡就是神經網路的數學運算！
    # 當 Python 執行到這一行時，它「不會」在這裡等矩陣算完。
    # 它只是把「A @ B」這個動作打包，當作指令「發射」進 my_custom_stream 裡。
    C = A @ B  
    
    print(f"[{time.time():.5f}] 🚗 [專屬車道] 數學指令發射完畢！(但 GPU 此時可能還在苦命計算中)")

# ==========================================
# 4. 同步與等待（對應 torchgpipe 的 wait_stream / synchronize）
# ==========================================
print(f"[{time.time():.5f}] 🚦 主程式：已經離開 with 區塊，繼續往下執行。")

# ⚠️ 致命關鍵：因為指令是非同步發射的，如果矩陣超級大（例如 10000x10000），
# 此時 GPU 絕對還沒算完。如果我們不等待就直接拿 C 去用，程式可能會出錯或拿到空值。
# 所以我們必須強迫 CPU 總經理停下來，等這條車道上的工作全數完工：
my_custom_stream.synchronize()

print(f"[{time.time():.5f}] ✅ 主程式：確認專屬車道回報完工！")
print("\n最終計算結果 C =")
print(C)