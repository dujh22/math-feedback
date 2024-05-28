import os

def generate_checkpoint_path(source_path):
    # 获取源文件的目录和文件名
    directory, filename = os.path.split(source_path)
    # 创建检查点文件名，添加后缀
    checkpoint_filename = f"{filename}_checkpoint"
    # 合成完整的检查点文件路径
    checkpoint_path = os.path.join(directory, checkpoint_filename)
    return checkpoint_path

# 使用例子
source_path = "/path/to/your/data.jsonl"
checkpoint_path = generate_checkpoint_path(source_path)
print(checkpoint_path)