import torch

def verify_snapshot_behavior():
    print(f"PyTorch Version: {torch.__version__}")
    
    # 確保系統至少有兩個 GPU 可供測試
    if torch.cuda.device_count() < 2:
        print("需要至少 2 個 GPU 來進行測試。")
        return

    # 開啟記憶體紀錄 (Global 設定)
    torch.cuda.memory._record_memory_history(max_entries=100000)

    # 刻意在 GPU 0 及 GPU 1 各分配一塊記憶體
    tensor_gpu0 = torch.randn(1000, 1000, device="cuda:0")
    tensor_gpu1 = torch.randn(1000, 1000, device="cuda:1")
    
    print("\n[測試中...] 在 GPU 0 及 GPU 1 已經分配並紀錄記憶體。")
    print("\n現在我們進入 `with torch.cuda.device(0):` 的區塊...")

    with torch.cuda.device(0):
        # 此時當前的預設裝置為 cuda:0
        current_device = torch.cuda.current_device()
        print(f"目前 Pytorch 指定的 CUDA 裝置為: cuda:{current_device}")
        
        # 進行 snapshot 抓取
        snapshot = torch.cuda.memory._snapshot()

        # 檢驗 snapshot 中的 device_traces
        device_traces = snapshot.get("device_traces", [])
        
        print("\n結果顯示 `_snapshot()` 所擷取到的 GPU 軌跡如下：")
        
        # device_traces 是一個 list of lists，對應 device_index
        for idx, trace_list in enumerate(device_traces):
            if len(trace_list) > 0:
                print(f"  👉 發現 CUDA 裝置 {idx} 內有 {len(trace_list)} 筆記憶體紀錄！")

if __name__ == "__main__":
    verify_snapshot_behavior()
