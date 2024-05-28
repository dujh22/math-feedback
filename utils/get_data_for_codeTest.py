# 读取指定文件夹下所有的 jsonl、json 或 txt 文件，然后从每个文件中提取前10个数据点并将这些数据点保存到新的文件夹中。
# 注意事项: 这个脚本会创建一个新的子文件夹，名字是原文件夹名字加上 _for_codeTest。

import os
import json

def get_data_for_codeTest(folder_path, new_folder_suffix='_for_codeTest', num_points=100):
    # 确定新文件夹的路径
    base_folder_path = os.path.dirname(folder_path)
    new_folder_path = os.path.join(base_folder_path, os.path.basename(folder_path) + new_folder_suffix)
    
    print("为编码进行小样本测试，提取前", num_points, "个数据点……")

    # 创建新文件夹
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jsonl', '.json', '.txt')):
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(new_folder_path, filename)
            
            print("正在处理文件:", old_file_path)
            
            # 根据不同文件类型处理
            if filename.endswith('.jsonl') or filename.endswith('.txt'):
                # 处理文本或jsonl文件
                with open(old_file_path, 'r', encoding='utf-8') as file:
                    try:
                        lines = [json.loads(next(file)) for _ in range(num_points)]
                    except StopIteration:
                        # 处理文件中数据点不足的情况
                        continue
                with open(new_file_path, 'w', encoding='utf-8') as new_file:
                    for line in lines:
                        new_file.write(json.dumps(line, ensure_ascii=False) + '\n')
            elif filename.endswith('.json'):
                # 处理json文件
                with open(old_file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                with open(new_file_path, 'w', encoding='utf-8') as new_file:
                    json.dump(data[:num_points], new_file, indent=4, ensure_ascii=False)

def main():
    folder_path = "F://code//github//ChatGLM-MathV2//raw_data//peiyi9979_Math_Shepherd"
    # folder_path = input("请输入文件夹路径: ")
    get_data_for_codeTest(folder_path)

if __name__ == '__main__':
    main()
