import json
import re
import os
from tqdm import tqdm

def data_preprocess_for_raw_math_shepherd(data_file_path:str, output_file_path:str):
    with open(data_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 确保输出路径存在
    output_dir = os.path.dirname(output_file_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file_path, "w", encoding="utf-8") as f:
        for line in tqdm(lines, desc="Processing"):
            # 读取数据
            data = json.loads(line)
            question = data["question"]
            answer = ""
            label = []
            for step, item in data["solution"].items():
                temp_content = item["content"]
                # 删除所有可能存在的<<20*40=8000>>
                temp_content = re.sub(r"<<[^<>]*>>", "", temp_content)
                answer += temp_content
                answer += "<|reserved_special_token_250|>"

                if item["label"] != 0 and item["label"] != 1:
                    item["label"] = 1
                label.append(item["label"])

            # 生成jsonl格式数据
            new_data = {
                "prompt": "Question: " + question + "\nAnswer: " + answer,
                "label": label
            }
            f.write(json.dumps(new_data) + "\n")

def main():
    input_file_path = "F://code//github//math-feedback//math-feedback//prm_training_by_step_last_position//raw_data//math-shepherd.jsonl"
    output_file_path = "F://code//github//math-feedback//math-feedback//prm_training_by_step_last_position//raw_data//math-shepherd2.jsonl"
    data_preprocess_for_raw_math_shepherd(input_file_path, output_file_path)

if __name__ == "__main__":
    main()