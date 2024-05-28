import json
import os
import re
from tqdm import tqdm
from highlight_equations import highlight_equations

solution = "First, we need to calculate the weekly cost for each type of lesson.\n\nFor clarinet lessons:\nHourly rate = $40\nHours per week = 3\nWeekly cost for clarinet lessons = Hourly rate × Hours per week\nWeekly cost for clarinet lessons = $40 × 3 = $120\n\nFor piano lessons:\nHourly rate = $28\nHours per week = 5\nWeekly cost for piano lessons = Hourly rate × Hours per week\nWeekly cost for piano lessons = $28 × 5 = $140\n\nNow, we need to calculate the annual cost for each type of lesson.\n\nThere are 52 weeks in a year, so:\n\nAnnual cost for clarinet lessons = Weekly cost for clarinet lessons × 52\nAnnual cost for clarinet lessons = $120 × 52 = $6240\n\nAnnual cost for piano lessons = Weekly cost for piano lessons × 52\nAnnual cost for piano lessons = $140 × 52 = $7280\n\nNow, we need to find the difference in cost between piano and clarinet lessons for the year:\n\nDifference in annual cost = Annual cost for piano lessons - Annual cost for clarinet lessons\nDifference in annual cost = $7280 - $6240 = $1040\n\nJanet spends $1040 more on piano lessons than clarinet lessons in a year."

def split_response(response): # 使用正则表达式按换行符分割响应文本
    # 首先判断\n\n存在的数量，如果超过超过一个则按照这个划分
    if response.count('\n\n') >= 2:
        steps = re.split(r"\n\n", response)
        steps = [x.strip() for x in steps if len(x.strip()) > 0] # 去除空白字符
        return steps
    # 然后判断\n存在的数量，如果超过一个则按照这个划分
    if response.count('\n') >= 2:
        steps = re.split(r"\n", response)
        steps = [x.strip() for x in steps if len(x.strip()) > 0] # 去除空白字符
        return steps
    # 否则按照句号划分
    else:
        # 使用正则表达式按句号切割非小数点
        if '。' in response:
            steps = re.split(r'。', response)
        else:
            steps = re.split(r'(?<=[^.0-9])\.(?=[^0-9])', response)
        steps = [x.strip() for x in steps if len(x.strip()) > 0] # 去除空白字符
        return steps

def process_json_line(data):
    # 加载原始JSON
    new_json = {"solution": {}, "dataset": "math23k", "question": "4"}
    # 方案拆分
    split_responses = split_response(data)

    # 处理每个解决方案部分
    for i, solution in enumerate(split_responses):
        new_json["solution"][f"Step {i+1}"] = {
            "content": solution.strip(),
            "label": 1  # 默认标签为1
        }

    # 处理每个解决方案部分的数学公式高亮
    for step, info in new_json["solution"].items():
        temp_content = info["content"]
        # info["content"] = highlight_equations(temp_content)
            
    # 返回新的JSON格式
    return json.dumps(new_json, ensure_ascii=False, indent=4)

print(process_json_line(solution))