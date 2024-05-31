import os
import json

def merge_jsonl_files(input_folder, output_file):
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
                        for response in responses:
                            new_entry = {
                                'question': question,
                                'response': response
                            }
                            outfile.write(json.dumps(new_entry, ensure_ascii=False) + '\n')

def main():
    project_path = '/workspace/dujh22/math_feedback/prm_evaluation/data/'
    dataset_name = "gsm8k"

    input_folder = project_path + dataset_name + '1/'
    output_file = project_path + dataset_name + '1_1/' + dataset_name + '.jsonl'
    
    # 确保输出目录存在
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    merge_jsonl_files(input_folder, output_file)

if __name__ == "__main__":
    main()
