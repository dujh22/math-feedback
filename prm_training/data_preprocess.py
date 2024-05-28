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
                answer += '\n'

                if item["label"] != 0 and item["label"] != 1:
                    item["label"] = 1
                label.append(item["label"])

            split_label = '\n'
            # 生成jsonl格式数据
            new_data = {
                "question": question,
                "answer": answer,
                "label": label,
                "split_label": split_label
            }
            f.write(json.dumps(new_data) + "\n")

def data_preprocess(data_file_path:str, output_file_path:str):
    with open(data_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    with open(output_file_path, "w", encoding="utf-8") as f:
        for line in tqdm(lines, desc="Processing"):
            data = json.loads(line)
            question = data["question"]
            answer = data["answer"]
            label = data["label"]
            split_label = data["split_label"]

            # 将原字符串按 split_label 拆分成部分
            parts_chosen  = answer.split(split_label)
            parts_rejected  = answer.split(split_label)
            # 重新连接字符串
            answer_with_label_chosen = parts_chosen[0]
            answer_with_label_rejected = parts_rejected[0]

            # 找到anwer_with_label中所有的split_label，然后在其前面加上label
            for i in range(len(label)):
                if label[i] != 0 and label[i] != 1:
                    label[i] = 1
                insert_chosen_label = '+'
                insert_reject_label = '-'
                if label[i] == 0:
                    insert_chosen_label = '-'
                    insert_reject_label = '+'
                
                answer_with_label_chosen +=  " " + insert_chosen_label + split_label
                if i + 1 < len(label):
                    answer_with_label_chosen += parts_chosen[i + 1]
                
                answer_with_label_rejected += " " + insert_reject_label + split_label
                if i + 1 < len(label):
                    answer_with_label_rejected += parts_rejected[i + 1]

            chosen_str = "Human: " + question + "\n\n" + "Assistant: " + answer_with_label_chosen
            rejected_str = "Human: " + question + "\n\n" + "Assistant: " + answer_with_label_rejected

            new_data = {
                "chosen": chosen_str,
                "rejected": rejected_str
            }

            f.write(json.dumps(new_data) + "\n")

def main():
    input_file_path = "F://code//github//math-feedback//math-feedback//prm_training//raw_data//math-shepherd.jsonl"
    output_file_path = "F://code//github//math-feedback//math-feedback//prm_training//raw_data//math-shepherd2.jsonl"
    data_preprocess_for_raw_math_shepherd(input_file_path, output_file_path)
    # input_file_path = "F://code//github//math-feedback//math-feedback//prm_training//raw_data//math-shepherd.jsonl"
    # output_file_path = "F://code//github//math-feedback//math-feedback//prm_training//raw_data//math-shepherd2.jsonl"
    # data_preprocess(input_file_path, output_file_path)

if __name__ == "__main__":
    main()