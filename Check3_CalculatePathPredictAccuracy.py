import json
from statistics import mean
import csv
import sys

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
    case_result_correct = 0
    case_total_correct = 0
    case_total = 0
    step_total_correct = 0
    step_total = 0
    step_soft_total_correct = 0
    step_hard_total_correct = 0

    for item in data:
        case_total += 1

        standard_label = []
        if item.get("solution") is not None:
            for step, info in item["solution"].items():
                if info.get('label') is not None:
                    standard_label.append(info['label'])

        # 获得整体结果
        critic_result = int(item['critic_result'][0]['rating'])
        binary_classification = 0 if critic_result <= 5 else 1
        
        if item.get("label") is None:
            if len(standard_label) > 0:
                if any(item == 0 for item in standard_label):
                    standard_binary_classification = 0
            else:
                    standard_binary_classification = 1
        else:
            standard_binary_classification = item.get("label")
        if binary_classification == standard_binary_classification:
            case_result_correct += 1

        case_correct = True
        # 获得单步结果
        steps = 0
        for info in item['generated_paths']:
            step_total += 1

            step_ratings = info['ratings']
            avg_critic_score = mean(step_ratings)
            binary_classification = 0 if avg_critic_score <= 5 else 1
            
        
            if len(standard_label) > 0:
                standard_binary_classification = standard_label[steps]
                steps += 1
            else:
                standard_binary_classification = info.get('label', 1)

            if binary_classification == standard_binary_classification:
                step_total_correct += 1
            else:
                case_correct = False
            
            # 获得软判断结果
            soft_label = info['soft_label']
            if soft_label > 0.5 and standard_binary_classification > 0.5:
                step_soft_total_correct += 1
            elif soft_label <= 0.5 and standard_binary_classification <= 0.5:
                step_soft_total_correct += 1

            # 获得硬判断结果
            hard_label = info['hard_label']
            if hard_label == standard_binary_classification:
                step_hard_total_correct += 1


        if case_correct:
            case_total_correct += 1


    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['键', '值', '说明'])
        
        print("步骤统计数据")
        writer.writerow(['步骤统计数据'])
        explanation = [
            '总步骤数',
            '规则判断正确的步数',
            '软判断正确的步数',
            '硬判断正确的步数'
        ]
        print_padded_line("step_total", step_total, explanation[0])
        writer.writerow(["step_total", step_total, explanation[0]])
        print_padded_line("step_total_correct", step_total_correct, explanation[1])
        writer.writerow(["step_total_correct", step_total_correct, explanation[1]])
        print_padded_line("step_soft_total_correct", step_soft_total_correct, explanation[2])
        writer.writerow(["step_soft_total_correct", step_soft_total_correct, explanation[2]])
        print_padded_line("step_hard_total_correct", step_hard_total_correct, explanation[3])
        writer.writerow(["step_hard_total_correct", step_hard_total_correct, explanation[3]])                   

        print("案例统计数据")
        writer.writerow(['案例统计数据'])
        explanation = [
            '总案例数',
            '规则判断步骤正确的案例数',
            '规则判断结果正确的案例数'
        ]
        print_padded_line("case_total", case_total, explanation[0])
        writer.writerow(["case_total", case_total, explanation[0]])
        print_padded_line("case_total_correct", case_total_correct, explanation[1])
        writer.writerow(["case_total_correct", case_total_correct, explanation[1]])
        print_padded_line("case_result_correct", case_result_correct, explanation[2])
        writer.writerow(["case_result_correct", case_result_correct, explanation[2]])

        print("步骤统计比率及其解释")
        writer.writerow(['步骤统计比率及其解释'])
        explanation = [
            '步骤正确率',
            '软判断正确率',
            '硬判断正确率'
        ]
        step_accuracy = step_total_correct / step_total * 100
        print_padded_line("step_accuracy", f"{step_accuracy:.2f}%", explanation[0])
        writer.writerow(["step_accuracy", f"{step_accuracy:.2f}%", explanation[0]])
        step_soft_accuracy = step_soft_total_correct / step_total * 100
        print_padded_line("step_soft_accuracy", f"{step_soft_accuracy:.2f}%", explanation[1])
        writer.writerow(["step_soft_accuracy", f"{step_soft_accuracy:.2f}%", explanation[1]])
        step_hard_accuracy = step_hard_total_correct / step_total * 100
        print_padded_line("step_hard_accuracy", f"{step_hard_accuracy:.2f}%", explanation[2])
        writer.writerow(["step_hard_accuracy", f"{step_hard_accuracy:.2f}%", explanation[2]])
                        
        
        print("案例统计比率及其解释")
        writer.writerow(['案例统计比率及其解释'])
        explanation = [
            '案例过程正确率',
            '案例结果正确率'
        ]
        case_process_accuracy = case_total_correct / case_total * 100
        print_padded_line("case_process_accuracy", f"{case_process_accuracy:.2f}%", explanation[0])
        writer.writerow(["case_process_accuracy", f"{case_process_accuracy:.2f}%", explanation[0]])
        case_result_accuracy = case_result_correct / case_total * 100
        print_padded_line("case_result_accuracy", f"{case_result_accuracy:.2f}%", explanation[1])
        writer.writerow(["case_result_accuracy", f"{case_result_accuracy:.2f}%", explanation[1]])

# 主函数
def main():
    if len(sys.argv) > 2:
        file_path = sys.argv[1]
        output_file_path = sys.argv[2]
    else:
        file_path = 'F://code//github//ChatGLM-MathV2//data//test_data100//test_data100_tgi_math_critic_path_math_critic2.jsonl'
        output_file_path = 'F://code//github//ChatGLM-MathV2//data//test_data100//test_data100_tgi_math_critic_path_math_critic2_statistics.csv'
    data = read_jsonl(file_path)
    calculate_accuracy(data, output_file_path)

if __name__ == "__main__":
    main()
