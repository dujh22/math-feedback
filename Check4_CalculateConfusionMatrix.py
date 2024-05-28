import json
import sys
from tqdm import tqdm
import csv

# 读取JSONL文件
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def print_padded_line(key, value, explanation, width=60):
    """格式化输出一行，并附带解释，确保总宽度为固定值"""
    line = f"{key}: {value}"
    print(line + ' ' * (width - len(line)) + explanation)

# 统计准确率
def calculate_accuracy(data, output_file):
    # 用于存储样例相关数值
    case_num = {"T": 0, "F": 0}
    front_case_ConfusionMatrix = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
    back_case_ConfusionMatrix = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
    # 用于存储步骤相关数值
    step_num = {"T": 0, "F": 0}
    front_step_ConfusionMatrix = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
    back_step_ConfusionMatrix = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

    for entry in tqdm(data, desc='Processing'):
        # 记录标准标签
        standard_label = []

        # 统计前向
        total_label = True # 记录当前数据样例是否为步骤全对样例
        step_label = True
        for step_key, step_info in entry['solution'].items():
            if step_info.get('label') is not None:
                standard_label.append(step_info['label'])

                if step_info['label'] == 1:
                    step_num['T'] += 1
                    if any(temp for temp in step_info['JudgmentStepCalculatedCorrectly']) == 0 or any(temp for temp in step_info['JudgmentStepEquationCorrectly']) == 0 or step_info['JudgmentStepReasoningCorrectly'] == 0:
                        front_step_ConfusionMatrix['FN'] += 1
                    else:
                        front_step_ConfusionMatrix['TP'] += 1
                else:
                    step_num['F'] += 1
                    total_label = False
                    if any(temp for temp in step_info['JudgmentStepCalculatedCorrectly']) == 0 or any(temp for temp in step_info['JudgmentStepEquationCorrectly']) == 0 or step_info['JudgmentStepReasoningCorrectly'] == 0:
                        front_step_ConfusionMatrix['TN'] += 1
                    else:
                        front_step_ConfusionMatrix['FP'] += 1
                
        
        if total_label == True:
            case_num['T'] += 1
            if entry['JudgmentAllCorrectly'] == 1:
                front_case_ConfusionMatrix["TP"] += 1
            else:
                front_case_ConfusionMatrix["FN"] += 1
        else:
            case_num["F"] += 1
            if entry['JudgmentAllCorrectly'] == 1:
                front_case_ConfusionMatrix["FP"] += 1
            else:
                front_case_ConfusionMatrix["TN"] += 1

        # 统计后向
        for it, step_info in enumerate(entry['generated_paths']):
            if len(standard_label) > 0:
                if standard_label[it] == 1:
                    if step_info['hard_label'] == 0:
                        back_step_ConfusionMatrix['FN'] += 1
                    else:
                        back_step_ConfusionMatrix['TP'] += 1
                else:
                    if step_info['hard_label'] == 0:
                        back_step_ConfusionMatrix['TN'] += 1
                    else:
                        back_step_ConfusionMatrix['FP'] += 1
        
        if total_label == True:
            if int(entry['critic_result'][0]["rating"]) > 8:
                back_case_ConfusionMatrix["TP"] += 1
            else:
                back_case_ConfusionMatrix["FN"] += 1
        else:
            if int(entry['critic_result'][0]["rating"]) > 8:
                back_case_ConfusionMatrix["FP"] += 1
            else:
                back_case_ConfusionMatrix["TN"] += 1
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['键', '值', '说明'])

        print("步骤统计数据")
        writer.writerow(['步骤统计数据'])
        
        step_total = step_num['F'] + step_num['T']
        print_padded_line("总步骤数", step_total, "步骤的总数")
        writer.writerow(["总步骤数", step_total, "步骤的总数"])
        print_padded_line("正确步骤数", step_num['T'], "正确步骤的总数")
        writer.writerow(["正确步骤数", step_num['T'], "正确步骤的总数"])
        print_padded_line("错误步骤数", step_num['F'], "错误步骤的总数")
        writer.writerow(["错误步骤数", step_num['F'], "错误步骤的总数"])

        print("样例统计数据")
        writer.writerow(['样例统计数据'])
        
        case_total = case_num['F'] + case_num['T']
        print_padded_line("总样例数", case_total, "样例的总数")
        writer.writerow(["总样例数", case_total, "样例的总数"])
        print_padded_line("正确样例数", case_num['T'], "正确样例的总数")
        writer.writerow(["正确样例数", case_num['T'], "正确样例的总数"])
        print_padded_line("错误样例数", case_num['F'], "错误样例的总数")
        writer.writerow(["错误样例数", case_num['F'], "错误样例的总数"])


        print("前向步骤统计数据")
        writer.writerow(['前向步骤统计数据'])

        descriptions = {
            'TP': '真正例（True Positive）',
            'FP': '假正例（False Positive）',
            'FN': '假负例（False Negative）',
            'TN': '真负例（True Negative）'
        }
        for key, value in front_step_ConfusionMatrix.items():
            line_description = descriptions[key]
            print_padded_line(key, value, line_description)
            writer.writerow([key, value, line_description])

        print("前向步骤分析数据")
        writer.writerow(['前向步骤分析数据'])

        if step_total > 0:
            step_accuracy = 100 * (front_step_ConfusionMatrix['TP'] + front_step_ConfusionMatrix['TN']) / step_total
            print_padded_line("步骤准确率", f"{step_accuracy:.2f}%", "总步骤的准确率")
            writer.writerow(["步骤准确率", f"{step_accuracy:.2f}%", "总步骤的准确率"])
        else:
            print_padded_line("步骤准确率", "N/A", "总步骤的准确率")
            writer.writerow(["步骤准确率", "N/A", "总步骤的准确率"])
        if front_step_ConfusionMatrix['TP'] + front_step_ConfusionMatrix['FP'] > 0:
            step_precision = 100 * front_step_ConfusionMatrix['TP'] / (front_step_ConfusionMatrix['TP'] + front_step_ConfusionMatrix['FP'])
            print_padded_line("步骤精确率", f"{step_precision:.2f}%", "预测为正的步骤中实际为正的比率")
            writer.writerow(["步骤精确率", f"{step_precision:.2f}%", "预测为正的步骤中实际为正的比率"])
        else:
            print_padded_line("步骤精确率", "N/A", "预测为正的步骤中实际为正的比率")
            writer.writerow(["步骤精确率", "N/A", "预测为正的步骤中实际为正的比率"])
        if front_step_ConfusionMatrix['TP'] + front_step_ConfusionMatrix['FN'] > 0:
            step_recall = 100 * front_step_ConfusionMatrix['TP'] / (front_step_ConfusionMatrix['TP'] + front_step_ConfusionMatrix['FN'])
            print_padded_line("步骤召回率", f"{step_recall:.2f}%", "实际为正的步骤中预测为正的比率")
            writer.writerow(["步骤召回率", f"{step_recall:.2f}%", "实际为正的步骤中预测为正的比率"])
        else:
            print_padded_line("步骤召回率", "N/A", "实际为正的步骤中预测为正的比率")
            writer.writerow(["步骤召回率", "N/A", "实际为正的步骤中预测为正的比率"])

        print("前向样例统计数据")
        writer.writerow(['前向样例统计数据'])

        descriptions = {
            'TP': '真正例（True Positive）',
            'FP': '假正例（False Positive）',
            'FN': '假负例（False Negative）',
            'TN': '真负例（True Negative）'
        }
        for key, value in front_case_ConfusionMatrix.items():
            line_description = descriptions[key]
            print_padded_line(key, value, line_description)
            writer.writerow([key, value, line_description])

        print("前向样例分析数据")
        writer.writerow(['前向样例分析数据'])

        if case_total > 0:
            case_accuracy = 100 * (front_case_ConfusionMatrix['TP'] + front_case_ConfusionMatrix['TN']) / case_total
            print_padded_line("样例准确率", f"{case_accuracy:.2f}%", "总样例的准确率")
            writer.writerow(["样例准确率", f"{case_accuracy:.2f}%", "总样例的准确率"])
        else:
            print_padded_line("样例准确率", "N/A", "总样例的准确率")
            writer.writerow(["样例准确率", "N/A", "总样例的准确率"])
        if front_case_ConfusionMatrix['TP'] + front_case_ConfusionMatrix['FP'] > 0:
            case_precision = 100 * front_case_ConfusionMatrix['TP'] / (front_case_ConfusionMatrix['TP'] + front_case_ConfusionMatrix['FP'])
            print_padded_line("样例精确率", f"{case_precision:.2f}%", "预测为正的样例中实际为正的比率")
            writer.writerow(["样例精确率", f"{case_precision:.2f}%", "预测为正的样例中实际为正的比率"])
        else:
            print_padded_line("样例精确率", "N/A", "预测为正的样例中实际为正的比率")
            writer.writerow(["样例精确率", "N/A", "预测为正的样例中实际为正的比率"])
        if front_case_ConfusionMatrix['TP'] + front_case_ConfusionMatrix['FN'] > 0:
            case_recall = 100 * front_case_ConfusionMatrix['TP'] / (front_case_ConfusionMatrix['TP'] + front_case_ConfusionMatrix['FN'])
            print_padded_line("样例召回率", f"{case_recall:.2f}%", "实际为正的样例中预测为正的比率")
            writer.writerow(["样例召回率", f"{case_recall:.2f}%", "实际为正的样例中预测为正的比率"])
        else:
            print_padded_line("样例召回率", "N/A", "实际为正的样例中预测为正的比率")
            writer.writerow(["样例召回率", "N/A", "实际为正的样例中预测为正的比率"])




        print("后向步骤统计数据")
        writer.writerow(['后向步骤统计数据'])

        descriptions = {
            'TP': '真正例（True Positive）',
            'FP': '假正例（False Positive）',
            'FN': '假负例（False Negative）',
            'TN': '真负例（True Negative）'
        }
        for key, value in back_step_ConfusionMatrix.items():
            line_description = descriptions[key]
            print_padded_line(key, value, line_description)
            writer.writerow([key, value, line_description])

        print("后向步骤分析数据")
        writer.writerow(['后向步骤分析数据'])

        if step_total > 0:
            step_accuracy = 100 * (back_step_ConfusionMatrix['TP'] + back_step_ConfusionMatrix['TN']) / step_total
            print_padded_line("步骤准确率", f"{step_accuracy:.2f}%", "总步骤的准确率")
            writer.writerow(["步骤准确率", f"{step_accuracy:.2f}%", "总步骤的准确率"])
        else:
            print_padded_line("步骤准确率", "N/A", "总步骤的准确率")
            writer.writerow(["步骤准确率", "N/A", "总步骤的准确率"])
        if back_step_ConfusionMatrix['TP'] + back_step_ConfusionMatrix['FP'] > 0:
            step_precision = 100 * back_step_ConfusionMatrix['TP'] / (back_step_ConfusionMatrix['TP'] + back_step_ConfusionMatrix['FP'])
            print_padded_line("步骤精确率", f"{step_precision:.2f}%", "预测为正的步骤中实际为正的比率")
            writer.writerow(["步骤精确率", f"{step_precision:.2f}%", "预测为正的步骤中实际为正的比率"])
        else:
            print_padded_line("步骤精确率", "N/A", "预测为正的步骤中实际为正的比率")
            writer.writerow(["步骤精确率", "N/A", "预测为正的步骤中实际为正的比率"])
        if back_step_ConfusionMatrix['TP'] + back_step_ConfusionMatrix['FN'] > 0:
            step_recall = 100 * back_step_ConfusionMatrix['TP'] / (back_step_ConfusionMatrix['TP'] + back_step_ConfusionMatrix['FN'])
            print_padded_line("步骤召回率", f"{step_recall:.2f}%", "实际为正的步骤中预测为正的比率")
            writer.writerow(["步骤召回率", f"{step_recall:.2f}%", "实际为正的步骤中预测为正的比率"])
        else:
            print_padded_line("步骤召回率", "N/A", "实际为正的步骤中预测为正的比率")
            writer.writerow(["步骤召回率", "N/A", "实际为正的步骤中预测为正的比率"])

        print("后向样例统计数据")
        writer.writerow(['后向样例统计数据'])

        descriptions = {
            'TP': '真正例（True Positive）',
            'FP': '假正例（False Positive）',
            'FN': '假负例（False Negative）',
            'TN': '真负例（True Negative）'
        }
        for key, value in back_case_ConfusionMatrix.items():
            line_description = descriptions[key]
            print_padded_line(key, value, line_description)
            writer.writerow([key, value, line_description])

        print("后向样例分析数据")
        writer.writerow(['后向样例分析数据'])

        if case_total > 0:
            case_accuracy = 100 * (back_case_ConfusionMatrix['TP'] + back_case_ConfusionMatrix['TN']) / case_total
            print_padded_line("样例准确率", f"{case_accuracy:.2f}%", "总样例的准确率")
            writer.writerow(["样例准确率", f"{case_accuracy:.2f}%", "总样例的准确率"])
        else:
            print_padded_line("样例准确率", "N/A", "总样例的准确率")
            writer.writerow(["样例准确率", "N/A", "总样例的准确率"])
        if back_case_ConfusionMatrix['TP'] + back_case_ConfusionMatrix['FP'] > 0:
            case_precision = 100 * back_case_ConfusionMatrix['TP'] / (back_case_ConfusionMatrix['TP'] + back_case_ConfusionMatrix['FP'])
            print_padded_line("样例精确率", f"{case_precision:.2f}%", "预测为正的样例中实际为正的比率")
            writer.writerow(["样例精确率", f"{case_precision:.2f}%", "预测为正的样例中实际为正的比率"])
        else:
            print_padded_line("样例精确率", "N/A", "预测为正的样例中实际为正的比率")
            writer.writerow(["样例精确率", "N/A", "预测为正的样例中实际为正的比率"])
        if back_case_ConfusionMatrix['TP'] + back_case_ConfusionMatrix['FN'] > 0:
            case_recall = 100 * back_case_ConfusionMatrix['TP'] / (back_case_ConfusionMatrix['TP'] + back_case_ConfusionMatrix['FN'])
            print_padded_line("样例召回率", f"{case_recall:.2f}%", "实际为正的样例中预测为正的比率")
            writer.writerow(["样例召回率", f"{case_recall:.2f}%", "实际为正的样例中预测为正的比率"])
        else:
            print_padded_line("样例召回率", "N/A", "实际为正的样例中预测为正的比率")
            writer.writerow(["样例召回率", "N/A", "实际为正的样例中预测为正的比率"])


# 主函数
def main():
    if len(sys.argv) > 2:
        file_path = sys.argv[1]
        output_file_path = sys.argv[2]
    else:
        file_path = 'F://code//github//ChatGLM-MathV2//data//math_shepherd_test_data10//front_Check2Step4//math_shepherd_test_data10.jsonl'
        output_file_path = 'F://code//github//ChatGLM-MathV2//data//math_shepherd_test_data10//front_Check2Step4//math_shepherd_test_data10_ConfusionMatrix.csv'
    data = read_jsonl(file_path)
    calculate_accuracy(data, output_file_path)

if __name__ == "__main__":
    main()
