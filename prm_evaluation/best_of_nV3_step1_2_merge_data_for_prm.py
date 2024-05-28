import os
import json

def merge_jsonl_files(input_folder, input_file, output_file):
    raw_data = []
    with open(input_file, 'r', encoding='utf-8') as infile0:
        for line in infile0:
            data = json.loads(line.strip())
            raw_data.append(data)

    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 遍历输入文件夹中的所有jsonl文件
        for filename in os.listdir(input_folder):
            if filename.endswith('.jsonl'):
                with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as infile:
                    for line in infile:
                        data = json.loads(line.strip())
                        question = data.get('question')
                        responses = data.get('responses', [])
                        data["prm_value"] = []
                        for response in responses:
                            response = response.replace("ки", "<|reserved_special_token_250|>")
                            prompt = "Question: " + question + "\nAnswer: " + response
                            # 在raw_data中找到这个prompt对应的reward = None
                            for raw_entry in raw_data:
                                if raw_entry.get('prompt') == [prompt]:
                                    reward = raw_entry.get('reward')
                                    data["prm_value"].append(reward)
                                    break
                        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

# 使用示例
# input_folder = 'F://code//github//math-feedback//math-feedback//prm_evaluation//data//test1'  # 替换为文件夹路径
# output_file = 'F://code//github//math-feedback//math-feedback//prm_evaluation//data//test1_1//test.jsonl'
input_folder = '/workspace/dujh22/math_feedback/prm_evaluation/data/test1'  # 替换为文件夹路径
input_file = '/workspace/dujh22/math_feedback/prm_evaluation/data/test1_1/test_rm.jsonl'
output_file = '/workspace/dujh22/math_feedback/prm_evaluation/data/test1_1/test_rm2.jsonl'

# 确保输出目录存在
if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))

merge_jsonl_files(input_folder, input_file, output_file)
