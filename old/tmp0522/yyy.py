def dynamic_counter():
    for i in range(3):  # 內部的 for 迴圈
        print(f"  [產生器內部] 準備生產，當前 i = {i}")
        yield i
        print(f"  [產生器內部] 醒來了！準備進入下一次迴圈")
print("start")
# 外部呼叫它
for num in dynamic_counter():  # 外部的 for 迴圈
    print(f"[主程式外部] 收到號碼牌：{num}")
    print("-" * 30)