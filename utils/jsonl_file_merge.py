import json
import os

def load_jsonl(file_path):
    """
    加载 JSONL 文件并返回一个包含所有 JSON 对象的列表。
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    file_path1 = "F://code//github//ChatGLM-MathV2//data//test_data100//test_data100_tgi_math_critic_path_math_critic2.jsonl"
    file_path2 = "F://code//github//ChatGLM-MathV2//data//test_data100//front_Check2Step4//test_data100.jsonl"
    output_file_path = "F://code//github//ChatGLM-MathV2//data//test_data100//pipeline_merge//test_data100.jsonl"

    new_data1 = load_jsonl(file_path1)
    new_data2 = load_jsonl(file_path2)
    
    # 合并两个数据集，以questions键作为键
    # 构造一个索引字典，方便后续合并，同时构造过程就会自动完成数据去重，因为只会保留同一个问题最后一次出现的id
    index_dict = {}
    for id, item in enumerate(new_data1):
        index_dict[item['question']] = {
            "new_data1": id,
            "new_data2": None
        }
    for id, item in enumerate(new_data2):
        if item['question'] in index_dict:
            index_dict[item['question']]["new_data2"] = id
        else:
            index_dict[item['question']] = {
                "new_data1": None,
                "new_data2": id
            }
    
    # 保存合并数据
    # 确保输出目录存在
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))
    with open(output_file_path, 'w', encoding='utf-8') as outfile:

        for question, index in index_dict.items():
            item = {}
            if index["new_data1"] is not None and index["new_data2"] is not None:
                item = new_data1[index["new_data1"]]
                item['solution'] = new_data2[index["new_data2"]]['solution']
                item['JudgmentAllCorrectly'] = new_data2[index["new_data2"]]['JudgmentAllCorrectly']
                # 将更新后的字典转换为JSON格式并写入文件
                json.dump(item, outfile, ensure_ascii=False)
                outfile.write('\n')  # 确保每个条目在新的一行


if __name__ == "__main__":
    main()