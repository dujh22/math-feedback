import json
import csv

def load_jsonl(file_path):
    """
    加载 JSONL 文件并返回一个包含所有 JSON 对象的列表。
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def compare_question_counts(file_path1, file_path2):
    """
    比较两个 JSONL 文件中不同问题的数量以及交集和差集。
    """
    data1 = load_jsonl(file_path1)
    data2 = load_jsonl(file_path2)

    unique_questions_data1 = len(set(item['question'] for item in data1))
    unique_questions_data2 = len(set(item['question'] for item in data2))

    questions_both = set(item['question'] for item in data1) & set(item['question'] for item in data2)
    questions_only_data1 = set(item['question'] for item in data1) - set(item['question'] for item in data2)
    questions_only_data2 = set(item['question'] for item in data2) - set(item['question'] for item in data1)

    return {
        "unique_questions_data1": unique_questions_data1,
        "unique_questions_data2": unique_questions_data2,
        "questions_both": len(questions_both),
        "questions_only_data1": len(questions_only_data1),
        "questions_only_data2": len(questions_only_data2)
    }

def compare_steps_and_export_csv(file_path1, file_path2, output_file):
    """
    比较两个 JSONL 文件中相同问题下生成的步骤数是否不同，并将结果保存到 CSV 文件中。
    """
    data1 = load_jsonl(file_path1)
    data2 = load_jsonl(file_path2)

    # 按问题对两个数据集进行排序
    data1.sort(key=lambda x: x['question'])
    data2.sort(key=lambda x: x['question'])

    different_steps = []
    index1 = 0
    index2 = 0

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['question', 'steps1', 'steps2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        while index1 < len(data1) and index2 < len(data2):
            item1 = data1[index1]
            item2 = data2[index2]
            question1 = item1['question']
            question2 = item2['question']

            if question1 == question2:
                steps1 = len(item1['generated_paths'])
                steps2 = len(item2['solution'])

                if steps1 != steps2:
                    writer.writerow({'question': question1, 'steps1': steps1, 'steps2': steps2})

                index1 += 1
                index2 += 1
            elif question1 < question2:
                writer.writerow({'question': question1, 'steps1': len(item1['generated_paths']), 'steps2': 0})
                index1 += 1
            else:
                writer.writerow({'question': question2, 'steps1': 0, 'steps2': len(item2['solution'])})
                index2 += 1

        # 处理剩余的条目
        while index1 < len(data1):
            item1 = data1[index1]
            writer.writerow({'question': item1['question'], 'steps1': len(item1['generated_paths']), 'steps2': 0})
            index1 += 1

        while index2 < len(data2):
            item2 = data2[index2]
            writer.writerow({'question': item2['question'], 'steps1': 0, 'steps2': len(item2['solution'])})
            index2 += 1

    print("CSV 文件已生成。")

    return different_steps

if __name__ == "__main__":
    file_path1 = "F://code//github//ChatGLM-MathV2//data//test_data100//test_data100_tgi_math_critic_path_math_critic2.jsonl"
    file_path2 = "F://code//github//ChatGLM-MathV2//data//test_data100//front_Check2Step4//test_data100.jsonl"
    
    counts = compare_question_counts(file_path1, file_path2)
    print("data1中不同问题的数量:", counts["unique_questions_data1"])
    print("data2中不同问题的数量:", counts["unique_questions_data2"])
    print("两者都有的问题数量:", counts["questions_both"])
    print("data1有而data2没有的问题数量:", counts["questions_only_data1"])
    print("data2有而data1没有的问题数量:", counts["questions_only_data2"])
    
    
    
    differences = compare_steps_and_export_csv(file_path1, file_path2, "output_temp.csv")
    if differences:
        print("以下问题的步骤数不同：")
        for question, steps in differences.items():
            print(f"问题：{question[:10]}")
            print(f"文件1的步骤数：{steps['steps1']}")
            print(f"文件2的步骤数：{steps['steps2']}")
            #print(f"文件1的细节:{steps['step1_detail']}")
            #print(f"文件2的细节:{steps['step2_detail']}")
            print()
    else:
        print("两个文件中相同问题的步骤数相同。")
