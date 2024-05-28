import os
import json
from get_data_for_codeTest import get_data_for_codeTest
from Step1_SplitByRow import Step1_SplitByRow
from Step1_SplitByRow_forMathShepherd import Step1_SplitByRow_forMathShepherd
from Step2_IsCalculationOrReasoning import Step2_IsCalculationOrReasoning
from Step3_JudgmentStepCalculatedCorrectly import Step3_JudgmentStepCalculatedCorrectly
from Step4_JudgmentStepReasoningCorrectly import Step4_JudgmentStepReasoningCorrectly

# 获取项目所在位置
base_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = base_folder.replace("\\", "//")

def pipeline_api(question, solution, dataset='test'):
    # 构造数据
    line = {
        "question": question,
        "solution": solution,
        "dataset": dataset
    }
    # 保存到指定路径
    dataset_name = "api" # 数据集名称
    source_folder = base_folder + '//raw_data//' + dataset_name # 原始数据集所在位置
    mid_name = base_folder + '//data//' + dataset_name + '//' + dataset_name # 中间文件所在位置
    target_folder1 = mid_name + "_Step1_SplitByRow"
    target_folder2 = mid_name + "_Step2_IsCalculationOrReasoning"
    target_folder3 = mid_name + "_Step3_JudgmentStepCalculatedCorrectly"
    target_folder4 = mid_name + "_Step4_JudgmentStepReasoningCorrectly"
    Step1_SplitByRow(source_folder, target_folder1, line)
    Step2_IsCalculationOrReasoning(target_folder1, target_folder2)
    Step3_JudgmentStepCalculatedCorrectly(target_folder2, target_folder3)
    Step4_JudgmentStepReasoningCorrectly(target_folder3, target_folder4)
    # 读取结果并返回
    result = {}
    # 识别target_folder4下的jsonl文件
    for root, dirs, files in os.walk(target_folder4):
        for file in files:
            if file.endswith(".jsonl"):
                source_file_path = os.path.join(root, file)
                with open(source_file_path, 'r', encoding='utf-8') as src_file:
                    for line in src_file:
                        data = json.loads(line)
                        result = data
    
    # 只保留需要的部分
    need_result = {}
    need_result["question"] = result["question"]
    need_result['solution'] = {}
    for step, info in result["solution"].items():
        need_result['solution'][step] = {
            "content": info["content"],
            "JudgmentStepCalculatedCorrectly": info["JudgmentStepCalculatedCorrectly"],
            "JudgmentStepEquationCorrectly": info["JudgmentStepEquationCorrectly"],
            "JudgmentStepReasoningCorrectly": info["JudgmentStepReasoningCorrectly"],
            "StepCalculatedCorrectlyResult": info["StepCalculatedCorrectlyResult"],
            "StepEquationCorrectlyFormat": info["StepEquationCorrectlyFormat"],
            "StepReasoningCorrectlyResult": info["StepReasoningCorrectlyResult"]
        }
    
    return need_result, result

def pipeline_file():
    is_test = True # 小样本开关：如果为True，则只处理少量数据
    test_num = 10

    dataset_name = "peiyi9979_Math_Shepherd" # 数据集名称
    source_folder = base_folder + '//raw_data//' + dataset_name # 原始数据集所在位置
    mid_name = base_folder + '//data//' + dataset_name # 中间文件所在位置

    if is_test:
        get_data_for_codeTest(source_folder, new_folder_suffix='_for_codeTest', num_points=test_num)
        source_folder = source_folder + "_for_codeTest"

        target_folder1 = mid_name + "_for_codeTest" + "_Step1_SplitByRow_forMathShepherd"
        target_folder2 = mid_name + "_for_codeTest" + "_Step2_IsCalculationOrReasoning"
        target_folder3 = mid_name + "_for_codeTest" + "_Step3_JudgmentStepCalculatedCorrectly"
        target_folder4 = mid_name + "_for_codeTest" + "_Step4_JudgmentStepReasoningCorrectly"
    else:
        target_folder1 = mid_name + "_Step1_SplitByRow_forMathShepherd"
        target_folder2 = mid_name + "_Step2_IsCalculationOrReasoning"
        target_folder3 = mid_name + "_Step3_JudgmentStepCalculatedCorrectly"
        target_folder4 = mid_name + "_Step4_JudgmentStepReasoningCorrectly"

    if dataset_name == "peiyi9979_Math_Shepherd":
        Step1_SplitByRow_forMathShepherd(source_folder, target_folder1) 
    else:
        Step1_SplitByRow(source_folder, target_folder1)
    
    Step2_IsCalculationOrReasoning(target_folder1, target_folder2)
    Step3_JudgmentStepCalculatedCorrectly(target_folder2, target_folder3)
    Step4_JudgmentStepReasoningCorrectly(target_folder3, target_folder4)

def main():
    type = 'api'
    if type == 'api':
        question = "Janet pays $40/hour for 3 hours per week of clarinet lessons and $28/hour for 5 hours a week of piano lessons. How much more does she spend on piano lessons than clarinet lessons in a year?"
        solution = "Step 1: Janet spends 3 hours + 5 hours = 8 hours per week on music lessons. ки\nStep 2: She spends 40 * 3 = 120 on clarinet lessons per week. ки\nStep 3: She spends 28 * 5 = 140 on piano lessons per week. ки\nStep 4: Janet spends 120 + 140 = 260 on music lessons per week. ки\nStep 5: She spends 260 * 52 = 13520 on music lessons in a year. The answer is: 13520 ки"
        result, _ = pipeline_api(question, solution)
        print(result)
    else:
        pipeline_file()

if __name__ == '__main__':
    main()