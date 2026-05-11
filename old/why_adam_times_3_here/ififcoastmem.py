import torch

# 確保環境中有可用的 GPU
if not torch.cuda.is_available():
    print("這個範例需要 GPU 環境才能正確捕捉 CUDA 記憶體喔！")
    exit()

# 1. 開啟記憶體歷史記錄 (這是 _dump_snapshot 必須的前提)
torch.cuda.memory._record_memory_history()

try:
    print("--- 開始執行運算 ---")
    
    # 2. 分配 3x3 的 tensor 到 GPU (對應你說的 a 和 b)
    # 這裡預設會是 float32，每個 tensor 約佔 36 bytes 的「純資料」空間
    a = torch.randn(3, 3, device='cuda')
    b = torch.randn(3, 3, device='cuda')

    # 3. 陣列相乘 (element-wise multiplication)
    # 會產生一個暫存的 3x3 結果 tensor
    c = a * b

    # 4. 把所有元素加起來 (Reduce 操作)
    # total_sum 是一個只包含 1 個數值的標量 tensor (純量)
    total_sum = c.sum()

    # 5. 執行 if 判斷 (關鍵點！)
    # 這裡的 "> 0" 比較，會迫使 GPU 把 total_sum 的數值同步傳回給 CPU
    if total_sum > 0:
        print(f"結果大於 0: {total_sum.item():.4f}")
    elif total_sum < 0:
        print(f"結果小於 0: {total_sum.item():.4f}")
    else:
        print("結果等於 0")

    print("--- 運算結束 ---")

    # 6. 匯出記憶體快照 (snapshot)
    snapshot_file = "gpu_memory_snapshot.pickle"
    torch.cuda.memory._dump_snapshot(snapshot_file)
    print(f"\n✅ 記憶體快照已成功儲存至: {snapshot_file}")
    print("💡 提示：你可以把這個檔案拖曳到 https://pytorch.org/memory_viz 視覺化查看！")

finally:
    # 7. 關閉記憶體歷史記錄，釋放追蹤所佔用的資源
    torch.cuda.memory._record_memory_history(enabled=None)

# 補充：印出當前簡單的記憶體使用量給你做快速參考
print("\n--- 簡單記憶體統計 ---")
print(f"目前 GPU 實際分配 (Allocated): {torch.cuda.memory_allocated()} bytes")
print(f"目前 GPU 被快取保留 (Reserved): {torch.cuda.memory_reserved()} bytes")