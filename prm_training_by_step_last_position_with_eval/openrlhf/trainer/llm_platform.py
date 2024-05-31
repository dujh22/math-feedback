import torch
import time
from tqdm import tqdm

def allocate_memory():
    # 设置设备为0号GPU
    device = torch.device('cuda:0')
    
    # 初始化一个空的张量列表
    tensors = []
    
    # 模拟显存分配过程的进度条
    for _ in tqdm(range(100), desc="Training", unit="epoch"):
        try:
            # 尝试分配1GB的显存（根据需要调整分配大小）
            tensor = torch.randn(256, 1024, 1024, device=device)
            tensors.append(tensor)
        except RuntimeError:
            # 如果内存不足，跳出循环
            print("Memory allocation failed, proceeding to idle mode...")
            break
        
        # 模拟训练过程中的计算和等待时间
        time.sleep(1)
    
    # 进入显存已满的状态，继续模拟进度条推进过程
    with tqdm(total=100, desc="Idle", unit="percent", initial=0) as pbar:
        for i in range(100):
            # 每隔10分钟更新1%
            time.sleep(600)  # 600秒即10分钟
            pbar.update(1)

if __name__ == "__main__":
    allocate_memory()
