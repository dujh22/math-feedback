import json
import csv
import os

def Check1_JsonlVisualization(input_file_path):
    # 生成输出文件路径
    base_dir, filename = os.path.split(input_file_path)
    output_filename = filename.replace('.jsonl', '.csv')
    output_file_path = os.path.join(base_dir.replace('Step', 'CheckStep'), output_filename)

    # 确保输出文件夹存在
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # 打开输入和输出文件
    with open(input_file_path, 'r', encoding='utf-8') as jsonl_file, open(output_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = None
        # 逐行读取JSONL文件
        for line in jsonl_file:
            # 解析JSON行
            data = json.loads(line)
            # 获取问题
            question = data['question']
            # 遍历所有解决步骤
            for step_name, step_data in data['solution'].items():
                # 准备写入CSV的行数据，开始时只有问题和步骤内容
                row_data = {
                    'question': question,
                    'step': step_name,
                    'content': step_data['content'],
                    'label': step_data.get('label', '')
                }
                # 添加其他键值对
                for key, value in step_data.items():
                    if key not in ['content', 'label']:
                        row_data[key] = value
                
                # 特殊处理
                if 'LLMJudgmentStepReasoningCorrectly' not in row_data.keys():
                    row_data['LLMJudgmentStepReasoningCorrectly'] = ""
                
                # 如果是第一次循环，初始化csv_writer并写入表头
                if csv_writer is None:
                    headers = list(row_data.keys())  # 获取所有键作为表头
                    csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
                    csv_writer.writeheader()
                
                # 写入数据行
                csv_writer.writerow(row_data)

    # 返回完成提示
    print("CSV文件已成功创建。")

def main():
    # 输入文件路径
    # input_file_path = 'F://code//github//ChatGLM-MathV2//data//peiyi9979_Math_Shepherd_for_codeTest_Step4_JudgmentStepReasoningCorrectly//math-shepherd.jsonl'
    input_file_path = 'F://code//github//ChatGLM-MathV2//data//peiyi9979_Math_Shepherd_for_codeTest_Check2Step4_JudgmentStepReasoningCorrectly//math-shepherd.jsonl_1-10.jsonl'
    Check1_JsonlVisualization(input_file_path)

if __name__ == '__main__':
    main()
