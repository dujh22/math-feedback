import json
import uuid
import re

def all_alpha(str):
    if str == 'x': # 特例
        return False
    if all(temp.isalpha() for temp in str):
        return True
    if any(temp.isalpha() for temp in str):
        if any(temp.isdigit() for temp in str) == False:
            return True
    
    return False

def highlight_equations(text):
    # 预处理，在数字和字母之间加一个空格
    raw_text = text

    # 去除$
    text = text.replace('$', '')
    
    text2 = text
    text3 = ""
    
    i = 0
    while i < len(text2): 
        if text2[i].isdigit():
            if i + 1 < len(text2):
                if text2[i + 1].isdigit(): # 连续两个数字
                    text3 = text3 + text2[i]
                    i += 1
                elif text2[i + 1] == '.' and i + 2 < len(text2) and text2[i + 2].isdigit(): # 小数点后面是数字
                    text3 = text3 + text2[i] + text2[i + 1] + text2[i + 2]
                    i += 3
                elif text2[i + 1] == ')' or text2[i + 1] == '）': # 有括号
                    text3 = text3 + text2[i] + ')'
                    i += 2
                elif text2[i + 1] == '(' or text2[i + 1] == '（': # 有括号
                    if i + 2 < len(text2):
                        if text2[i + 2].isdigit():
                            text3 = text3 + text2[i] + " *( " + text2[i + 2]
                            i += 3
                        else:
                            text3 = text3 + text2[i]   + " ("
                            i += 2
                    else:
                       text3 = text3 + text2[i]
                       i += 2
                else:
                    text3 = text3 + text2[i] + ' '
                    i += 1
            else:
                text3 += text2[i]
                i += 1
        elif text2[i] in ['+', '-', '*', '/']:
            text3 = text3 + ' ' + text2[i] + ' '
            i += 1
        else: # 非数字
            if i + 1 < len(text2):
                if text2[i + 1].isdigit():
                    if text2[i] == '.':
                        if i - 1 >= 0:
                            if text3[i - 1].isdigit():
                                text3 = text3 + text2[i] + text2[i + 1]
                                i += 2
                            else:
                                text3 = text3 + text2[i] + ' ' + text2[i + 1]
                                i += 2
                        elif i == 0:
                            text3 = 0 + text2[i] + text2[i + 1]
                            i += 2
                    else: 
                        text3 = text3 + text2[i] + ' ' + text2[i + 1]
                        i += 2
                else:
                    text3 = text3 + text2[i]
                    i += 1
            else:
                text3 = text3 + text2[i]
                i += 1
    
    # 将连续的空格替换为1个
    temp_text = ""
    for i in range(0, len(text3)):
        if text3[i] == ' ':
            if i + 1 < len(text3):
                if text3[i + 1] == ' ':
                    continue
                else:
                    temp_text = temp_text + text3[i]
            else:
                continue
        else:
            temp_text = temp_text + text3[i]
    text3 = temp_text

    # 去寻找x, 如果其前后有数字，那么应该替换为*
    for i in range(0, len(text3) - 1):
        if text3[i] == 'x':
            if i - 1 > 0:
                if text3[i - 1].isdigit():
                    text3 = text3[:i] + '*' + text3[i + 1:]
                elif text3[i - 1] == ')':
                    text3 = text3[:i] + '*' + text3[i + 1:]
                elif text3[i - 1] == '）':
                    text3 = text3[:i] + '*' + text3[i + 1:]
                elif text3[i - 1] == ' ':
                    if i - 2 > 0:
                        if text3[i - 2].isdigit():
                            text3 = text3[:i] + '*' + text3[i + 1:]

    text = text3          
            
    # 下面是主逻辑       

    parts = text.split(' ')
    if len(parts) == 1:
        return text  # 没有等号，返回原文本

    # 找到所有的等式
    part2 = parts.copy()

    part3 = []
    equations = []
    # 首先去除所有数字后面存在的字母串
    i = 0
    while i < len(part2):
        if part2[i].isdigit() and i + 1 < len(part2) and all_alpha(part2[i + 1]):
            # 如果字母串后面还是数字，那么这个字母串其实不需要去
            if i + 2 < len(part2) and part2[i + 2].isdigit():
                part3.append(part2[i])
                i += 1
            else:   
                part3.append(part2[i])
                i += 2         
        else:
            part3.append(part2[i])
            i += 1
    # 然后开始在part3中找等式
    for i in range(len(part3)):
        # 找到=所在的位置
        if '=' not in part3[i]:
            continue
        else:
            start_pos = i - 1
            end_pos = i + 1
            
            while True:
                if start_pos - 1 >= 0:
                    if all_alpha(part3[start_pos - 1]) == False:
                        start_pos = start_pos - 1
                    else:
                        break
                else:
                    break

            equation = "".join(part3[start_pos:end_pos + 1])

            equations.append(equation)

    equation_id = 0
    result = []
    for i in raw_text:
        if equation_id < len(equations) and i == '=':
            result.append(f'= <<{equations[equation_id]}>>')
            equation_id += 1
        else:
            result.append(i) 

    return "".join(result), "".join(result).replace('>> ', '>>')

def split_response(response): # 使用正则表达式按换行符分割响应文本
    # 首先判断\n\n存在的数量，如果超过超过一个则按照这个划分
    if response.count('\n\n') >= 2:
        steps = re.split(r"\n\n", response)
        steps = [x.strip() for x in steps if len(x.strip()) > 0] # 去除空白字符
        return steps, '\n\n'
    # 然后判断\n存在的数量，如果超过一个则按照这个划分
    if response.count('\n') >= 2:
        steps = re.split(r"\n", response)
        steps = [x.strip() for x in steps if len(x.strip()) > 0] # 去除空白字符
        return steps, '\n'
    # 否则按照句号划分
    else:
        steps = re.split(r'。', response)
        steps = [x.strip() for x in steps if len(x.strip()) > 0] # 去除空白字符
        return steps, '。'

def split_into_max_parts(text, splitID, max_parts=10):
    # 使用正则表达式拆分文本
    parts = re.split(splitID, text)
    # 计算每个新部分应该包含的原始部分数目
    num_parts_per_section = len(parts) // max_parts + (len(parts) % max_parts > 0)
    
    # 重新组合部分以不超过最大部分数
    new_parts = []
    for i in range(0, len(parts), num_parts_per_section):
        # 将一定数量的部分合并为一个新部分
        new_parts.append('\n\n'.join(parts[i:i+num_parts_per_section]))
    
    # 如果合并后的部分数仍然多于最大允许的部分数，再次合并最后两个部分直到满足条件
    while len(new_parts) > max_parts:
        new_parts[-2:] = ['\n\n'.join(new_parts[-2:])]
    
    return new_parts



def split_responses(data):
    # 分解数据并创建新ID和共同的DID
    common_did = str(uuid.uuid4())  # 生成共同的DID
    new_entries1to10 = []
    new_entries11to20 = []

    for entry in data:
        responses = entry['response']
        base_entry = {k: v for k, v in entry.items() if k != 'response' and k != 'id'}
        for response in responses:
            new_id = str(uuid.uuid4())  # 为每个新记录生成新的ID
            new_entry = base_entry.copy()
            new_entry['id'] = new_id
            new_entry['did'] = common_did
            new_entry['response'] = response[1].strip('\n')
            # 按行拆分
            new_entry['steps'] = {}
            steps, splitID = split_response(response[1].strip('\n'))
            for i, step in enumerate(steps):
                new_entry['steps'][f"Step {i+1}"] = step
            
            # 保存
            if len(new_entry['steps']) <= 10:
                new_entries1to10.append(new_entry)
            elif len(new_entry['steps']) <= 20:
                new_entry['steps'] = {}
                merge_steps = split_into_max_parts(response[1].strip('\n'), splitID, 10)
                for i, step in enumerate(merge_steps):
                    new_entry['steps'][f"Step {i+1}"] = step
                new_entries11to20.append(new_entry)
            # split_response(response[1].strip('\n'))
            # # 公式高亮
            # new_entry['highlight'] = {}
            # for i, step in new_entry['steps'].items():
            #     new_entry['highlight'][i] = highlight_equations(step)
            #     # new_entry['highlight'].append(highlight_equations(item))
    return new_entries1to10, new_entries11to20

def read_jsonl_file(input_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

def write_jsonl_file(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + '\n')

# 示例使用
input_file_path = 'F://code//github//ChatGLM-MathV2//raw_data//internal_data//tiku_prm_1k.jsonl'
output_file_path1 = 'F://code//github//ChatGLM-MathV2//raw_data//internal_data//tiku_prm_1k_BranchHighlighting_1to10.jsonl'
output_file_path2 = 'F://code//github//ChatGLM-MathV2//raw_data//internal_data//tiku_prm_1k_BranchHighlighting_11to20.jsonl'
data = read_jsonl_file(input_file_path)
new_data1to10, new_dat11to20 = split_responses(data)
write_jsonl_file(new_data1to10, output_file_path1)
write_jsonl_file(new_dat11to20, output_file_path2)


# 针对new_dat11to20进行统计
# 1.统计各种type的数据占比
# 2.统计各种grade的数据占比
# 3.统计各种difficulty的数据占比
type_num = {}
grade_num = {}
difficulty_num = {}
for item in new_dat11to20:
    type_num[item['type']] = type_num.get(item['type'], 0) + 1
    grade_num[item['grade']] = grade_num.get(item['grade'], 0) + 1
    difficulty_num[item['difficulty']] = difficulty_num.get(item['difficulty'], 0) + 1
# 进一步统计各种组合的占比

# 输出
print("type_num:", type_num)
print("grade_num:", grade_num)
print("difficulty_num:", difficulty_num)


# 假设 new_dat11to20 是这样的列表：
# new_dat11to20 = [
#     {'type': '解答题', 'grade': '六年级', 'difficulty': '一般'},
#     {'type': '填空题', 'grade': '九年级', 'difficulty': '一般'},
#     ...
# ]

# 初始化组合字典
combo_num = {}

for item in new_dat11to20:
    # 构造组合的key，例如 "六年级_解答题"
    combo_key = f"{item['grade']}_{item['type']}"
    # 计数
    combo_num[combo_key] = combo_num.get(combo_key, 0) + 1

# 输出各种组合的数量
print("combo_num:", combo_num)

# 计算并输出各组合的占比
total = sum(combo_num.values())
combo_percentages = {k: v / total * 100 for k, v in combo_num.items()}
print("Combo Percentages:", combo_percentages)
