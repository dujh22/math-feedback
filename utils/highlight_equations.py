import json
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

def is_numeric(s):
    """
    Check if the string is purely numeric or in the form of a decimal.
    
    Args:
    s (str): The string to check.
    
    Returns:
    bool: True if the string is numeric, False otherwise.
    """
    # Check if the string is purely numeric or a decimal number
    if s.replace('.', '', 1).isdigit():
        # Ensure there's no more than one decimal point
        return s.count('.') <= 1
    return False


def extract_formulas_improved(text):
    """
    提取"="字符周围的数学表达式，在中文字符、换行符等处停止。
    
    参数
    text (str)： 要从中提取数学公式的文本。
    
    返回值
    list： 提取的数学公式列表。
    """
    import re
    # 定义一个列表来存储提取的公式
    formulas = []
    
    # Find all positions of '=' in the text
    for match in re.finditer(r"=", text):
        start_index, end_index = match.start(), match.end()
        
        # 从"="开始后扫描，找到公式的起点
        start_formula = start_index
        while start_formula > 0 and not re.match(r"[\u4e00-\u9fff。\n]", text[start_formula - 1]):
            start_formula -= 1
        
        # 从"="开始前扫描，找到公式的末尾
        end_formula = end_index
        while end_formula < len(text) and not re.match(r"[\u4e00-\u9fff。\n]", text[end_formula]):
            end_formula += 1
        
        # 提取公式并将其添加到列表中
        formulas.append(text[start_formula:end_formula].strip())
    
    return formulas

def split_sentences(response):
    if response.count('\n\n') >= 2:
        fuhao = '\n\n'
        steps = re.split(r"\n\n", response)
        steps = [x.strip() for x in steps if len(x.strip()) > 0] # 去除空白字符
        return steps, fuhao
    # 然后判断\n存在的数量，如果超过一个则按照这个划分
    if response.count('\n') >= 2:
        fuhao = '\n'
        steps = re.split(r"\n", response)
        steps = [x.strip() for x in steps if len(x.strip()) > 0] # 去除空白字符
        return steps, fuhao
    # 否则按照句号划分
    else:
        # 使用正则表达式按句号切割非小数点
        fuhao = '.'
        steps = re.split(r'(?<=[^.0-9])\.(?=[^0-9])', response)
        steps = [x.strip() for x in steps if len(x.strip()) > 0] # 去除空白字符
    return response, fuhao

def contains_consecutive_equations(text):
    # 查找所有等号的位置
    indices = [i for i, char in enumerate(text) if char == '=']
    
    # 如果少于两个等号，连等式不可能存在
    if len(indices) < 2:
        return False
    
    # 遍历每对连续的等号，检查它们之间的字符
    for i in range(len(indices) - 1):
        # 获取两个连续等号之间的内容
        between_equals = text[indices[i] + 1:indices[i + 1]]
        
        # 检查这部分内容是否只包含数字、字母、括号或计算符号
        if not all(char.isdigit() or char.isalpha() or char in ' +-*/()' for char in between_equals.strip()):
            return False

    return True

def split_equations(text):
    # 使用正则表达式匹配等号，并确保等号前后都有数字或括号
    parts = re.split(r'(?<=\d) = (?=\d|\()', text)
    # 创建最终的等式列表
    equations = []
    # 从拆分的部分中重新组装等式，确保每个等式包含一个等号
    for i in range(len(parts)-1):
        equations.append(parts[i] + " = " + parts[i+1].split(' = ', 1)[0])
    return equations

def highlight_equations(text):
    # 如果存在<<等式>>要首先删除，比如<<75+18=93>>
    text = re.sub(r"<<[^>]*>>", "", text)
    
    # 首先处理可能存在的连等的情况
    # 按照句号分割文本
    text_list, fuhao= split_sentences(text)
    # 逐个处理句子
    for i in range(len(text_list)):
        # 如果句子中包含连续的等式，就拆分它们
        if contains_consecutive_equations(text_list[i]):
            text_list[i] = " so that ".join(split_equations(text_list[i]))
    # 重新组合文本
    text = ''.join(text_list)

    # print(text)
    # print("-----------------------")

    # 计算包含中文字符的比率
    def chinese_ratio(check_str):
        count = 0
        for ch in check_str:
            if u'\u4e00' <= ch <= u'\u9fff':
                count += 1
        return count / len(check_str)

    if chinese_ratio(text) < 0.2: # 英文字符串说明是 
    
        # 预处理，在数字和字母之间加一个空格
        raw_text = text

        # 去除$
        text = text.replace('$', '')
        
        text2 = text
        text3 = ""
        
        i = 0
        while i < len(text2): 
            if text2[i] == ' ':
                text3 = text3 + text2[i]
                i += 1
            elif text2[i] == '.':
                if i - 1 >= 0 and text3[-1].isdigit() and i + 1 < len(text2) and text2[i + 1].isdigit():
                    text3 = text3 + text2[i] + text2[i + 1]
                    i += 2
                elif i - 1 >= 0 and text3[-1].isdigit():
                    text3 = text3 + ' ' + text2[i] + ' '
                    i += 2
                else:
                    text3 = text3 + ' ' + text2[i]
                    i += 1
            elif text2[i].isalpha():
                text3 = text3 + text2[i]
                i += 1
            elif text2[i] in ['+', '-', '*', '/']:
                if i - 1 >= 0 and text3[-1] != ' ':
                    text3 = text3 + ' '
                text3 = text3 +  text2[i]
                if i + 1 < len(text2) and text2[i + 1] != ' ':
                    text3 = text3 + ' '
                i += 1
            elif text2[i].isdigit():
                if i + 1 < len(text2):
                    if text2[i + 1].isdigit(): # 连续两个数字
                        text3 = text3 +text2[i] + text2[i + 1]
                        i += 2
                    elif text2[i + 1] == '.':
                        if i + 2 < len(text2) and text2[i + 2].isdigit(): # 小数点后面是数字
                            text3 = text3 + text2[i] + text2[i + 1] + text2[i + 2]
                            i += 3
                        else:
                            text3 = text3 + text2[i] + ' '
                            i += 1
                    elif text2[i + 1] == ')' or text2[i + 1] == '）': # 有括号
                        text3 = text3 + text2[i] + ' )'
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
            elif text2[i] == '(' or text2[i] == '（':
                text3 = text3 + " ("
                i += 1
            elif text2[i] == ')' or text2[i] == '）':
                text3 = text3 + ") "
                i += 1
            elif text2[i] == ',' or text2[i] == '，':
                text3 = text3 + " , "
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
            if text3[i] == 'x' or text3[i] == '×':
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
        
        # 处理所有的x
        for i in range(len(part2)):
            if part2[i] == 'x' or part2[i] == '×':
                if i - 1 >= 0 and is_numeric(part2[i - 1]):
                    part2[i] = '*'
                elif i + 1 < len(part2) and is_numeric(part2[i + 1]):
                    part2[i] = '*'
                
        
        # 去除所有数字后面存在的字母串
        i = 0
        while i < len(part2):
            if is_numeric(part2[i]):
                # 处理 5 / year 类似情况
                if i + 2 < len(part2) and part2[i + 1] == '/' and all_alpha(part2[i + 2]):
                    part3.append(part2[i])
                    i += 3
                # 处理 5 s / year 类似情况
                elif i + 3 < len(part2) and all_alpha(part2[i + 1]) and part2[i + 2] == '/' and all_alpha(part2[i + 3]):
                    part3.append(part2[i])
                    i += 4
                elif i + 1 < len(part2) and all_alpha(part2[i + 1]):
                    # 如果字母串后面还是数字，那么这个字母串其实不需要去
                    if i + 2 < len(part2) and is_numeric(part2[i + 2]):
                        part3.append(part2[i])
                        i += 1
                    else:   
                        part3.append(part2[i])
                        i += 2   
                else:
                    part3.append(part2[i])
                    i += 1
            else:
                part3.append(part2[i])
                i += 1
        # print("~~~~~~~~~")
        # print(part3)

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
                
                while True:
                    if end_pos + 1 < len(part3):
                        if all_alpha(part3[end_pos + 1]) == False:
                            end_pos = end_pos + 1
                        else:
                            break
                    else:
                        break

                equation = "".join(part3[start_pos:end_pos + 1])

                if equation[-1] == '.':
                    equation = equation[:-1]

                equations.append(equation)

        # print("********")
        # print(equations)     


        equation_id = 0
        result = []
        for i in raw_text:
            if i == '=':
                result.append(f'= <<{equations[equation_id]}>>')
                equation_id += 1
            else:
                result.append(i) 

        return "".join(result).replace('>> ', '>>')

    else: # 中文字符串
        # 提取所有的数学公式
        formulas = extract_formulas_improved(text)
        for formula in formulas:
            raw_formulas = formula
            # 在text中定位formula位置并将<<formula>>插入
            formula = formula.replace('$', '')

            # 处理所有的x
            for i in range(len(formula)):
                if formula[i] == 'x' or formula[i] == '×':
                    if i - 1 >= 0 and is_numeric(formula[i - 1]):
                        formula[i] = '*'
                    elif i + 1 < len(formula) and is_numeric(formula[i + 1]):
                        formula[i] = '*'

            text = text.replace(raw_formulas, f"<<{formula}>>")
        return text


if __name__ == '__main__':
    example_text = []
    answer_text = []
    with open('F://code//github//ChatGLM-MathV2//data//test_data100//test_data100.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            example_text.append(data['solution'])

    with open('F://code//github//ChatGLM-MathV2//raw_data//peiyi9979_Math_Shepherd_for_codeTest//math-shepherd1-100.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            input_text = data['input']
            # 提取问题和解决方案
            split_point = input_text.find("Step 1:")
            question = input_text[:split_point].strip()
            solution = input_text[split_point:].strip()
            # 移除所有的“Step n: ”和“ки”
            solution = solution.replace("ки", "")  # 删除所有的ки
            for i in range(1, 20):  # 假设步骤不超过20
                solution = solution.replace(f"Step {i}: ", "")
            answer_text.append(solution)


    # 测试代码

    # ans = 0
    # for i, item in enumerate(example_text):
    #     highlighted_text1 = highlight_equations(item)
    #     if highlighted_text1 != answer_text[i]:
    #         print("第", i + 1, "个样本:------------------------------------------------------")
    #         print("原文本：", item)
    #         print("高亮后：", highlighted_text1)
    #         print("标答案：", answer_text[i])
    #         print("第", i + 1, "个样本:------------------------------------------------------\n")
    #         ans += 1
    # print("error:", ans)



    i = 3
    example_text[i] = "The total ratio of slices of pizza that Buzz and the waiter are sharing is 5 + 8 = 13 parts.\n\nBuzz's share is 5 parts out of 13, so he ate 5/13 of the pizza.\n\nThe waiter's share is 8 parts out of 13, so he ate 8/13 of the pizza.\n\nTo find out how many slices the waiter ate, we calculate 8/13 of the total number of slices:\n\n(8/13) * 78 = (8 * 78) / 13 = 624 / 13 = 48 slices"
    highlighted_text = highlight_equations(example_text[i])
    print("原文本：", example_text[i])
    print("高亮后：", highlighted_text)
    print("标准答案：", answer_text[i])
    print()