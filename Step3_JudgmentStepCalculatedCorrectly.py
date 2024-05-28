# 如果打开下面一行，命令行会自动输出代码执行的时间
import time
# 这个装饰器 time_it 可以被应用到任何你希望测量执行时间的函数上。它通过计算函数开始和结束时的时间来计算执行时间，并将时间转换为小时、分钟和秒的格式。
def time_it(func):
    """
    装饰器，用于测量函数执行时间。
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 获取开始时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 获取结束时间
        time_taken = end_time - start_time  # 计算耗时
        hours, rem = divmod(time_taken, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"{func.__name__} executed in: {int(hours):02d}h:{int(minutes):02d}m:{seconds:06.3f}s")
        return result
    return wrapper

# import hunter # 用于调试
# hunter.trace(module=__name__)

# 计算单步

import os
import json
import re
import sys
from utils.get_data_for_codeTest import get_data_for_codeTest
from Step1_SplitByRow_forMathShepherd import Step1_SplitByRow_forMathShepherd
from Step2_IsCalculationOrReasoning import Step2_IsCalculationOrReasoning
from Check1_JsonlVisualization import Check1_JsonlVisualization

from utils.run_python_func import run_python
from tqdm import tqdm
import sympy as sp
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import time
import logging

from llm.llm_response import llm_response, TGI_URL, CRITIC_URL

# 配置日志记录器
logging.basicConfig(filename='Step3_JudgmentStepCalculatedCorrectly.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 修正函数
# 1.在 content 字符串中查找被 << 和 >> 包围的表达式。
# 2.替换 = 后面直到 >> 的内容为 StepCalculatedCorrectlyResult 的值。
# 3.如果 >> 后的内容（如果存在）与 = 和 >> 之间的内容相同，则也将其替换为 StepCalculatedCorrectlyResult。
# content = "The equation a = 2 results in a = 2, and then we see b = 3, which leads to b = 3."
# equations = ["a = 2", "b = 3"]
# judge = [0, 1]  # 只替换a的结果
# result = ["5", "4"]  # 将a的结果替换为5，b的结果不变（虽然这里b不需要替换）
def replace_calculated_result(content, equations, equations_need, equations_correct_format, judge, result):
    # 这里会有两种情况
    # 情况1：result[i]是正确的数值，那么直接进行错误数值的替换是没有问题的
    # 情况2：result[i]是一个字符串，那么就不是直接替换，而是作为一个补充信息只能。因为不可能把所有的错误数值替换为一个相同的文字串
    for i, equation in enumerate(equations):
        if judge[i] == 0:  # 需要修改的等式            
            # 等式本身是正确的了
            # 分解等式，获取左侧变量和原始结果
            variable, original_result = equation.split('=')
            variable = variable.strip()
            original_result = original_result.strip()
            
            if variable != "":
                # 构造用于搜索和替换的正则表达式
                search_pattern = re.escape(variable) + r'\s*=\s*' + re.escape(original_result)
                replace_pattern = f'{variable} = {result[i]}'
                
                # 替换等式
                content = re.sub(search_pattern, replace_pattern, content)
                
                # 替换全文中的原结果
                content = re.sub(r'\b' + re.escape(original_result) + r'\b', result[i], content)

                # 等式本身是不正确的，还需要修改等式本身
                if equations_need[i] == 0:
                    if equations_correct_format[i] != "":
                        content = content.replace(variable, str(equations_correct_format[i]))
                
    return content

# 删除公式中可能存在的单位
def simplify_expression(expr):
    import re
    result = ""
    i = 0
    while i < len(expr):
        if expr[i].isdigit() or expr[i] in "+-*/().":
            # 如果是数字或者符号，则添加到结果中
            result += expr[i]
            i += 1
        elif expr[i].isalpha():
            # 如果是字母，则向后扫描直到遇到非字母
            start = i
            while i < len(expr) and expr[i].isalpha():
                i += 1
            if i == len(expr):
                # 如果到达字符串末尾，则停止
                break
            elif expr[i] in "+-*/().":
                # 如果字母后是符号，检查后续内容
                if i + 1 < len(expr) and expr[i + 1].isalpha():
                    # 如果符号后是字母，忽略之前的所有字母
                    i += 1
            elif expr[i].isdigit():
                # 如果字母后是数字，忽略之前的所有字母
                i += 1
                continue
        else:
            # 其他情况，直接增加索引
            i += 1

    # 清理可能出现的连续符号错误，例如多个连续的乘除号
    result = re.sub(r'\*+/', '*', result)
    result = re.sub(r'/\*+', '/', result)

    return result

# 公式预处理
def formula_preprocessing(input_str):
    # 删除等号与等号右侧的内容
    input_str = re.sub(r'=.+', '', input_str)

    # 删除货币符号
    input_str = input_str.replace('$', '')

    # 去除数字前的前导零
    input_str = re.sub(r'\b0+(\d+)', r'\1', input_str)
    
    # 处理时间表示 (假设时间表示为小时和分钟，如 4:30 转换为 4/30)
    input_str = re.sub(r'(\d+):(\d+)', r'\1/\2', input_str)

    # 处理百分比
    input_str = re.sub(r'(\d+)%', r'(\1/100)', input_str)

    # 处理分数表示中的空格（假设意图是将 3 3/4 写成 33/4）
    input_str = re.sub(r'(\d+)\s+(\d+)/(\d+)', r'\1\2/\3', input_str)

    # 使用正则表达式添加必要的乘法符号
    input_str = re.sub(r'(\d)(\()', r'\1*\2', input_str)  # 处理数字后跟左括号的情况16+3(16)+7
    input_str = re.sub(r'(\))(\d)', r'\1*\2', input_str)  # 处理右括号后跟数字的情况(1/2)20
    input_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', input_str)  # 处理数字后跟变量的情况2x
    input_str = re.sub(r'([a-zA-Z])(\()', r'\1*\2', input_str)  # 处理变量后跟左括号的情况x(5+2)

    # 处理货币符号
    currency_match = re.match(r'\$(\d+(\.\d+)?)(\*\d+(\.\d+)?)?', input_str)
    if currency_match:
        # 提取金额和可能的乘数
        amount = float(currency_match.group(1))
        multiplier = currency_match.group(3)
        if multiplier:
            # 计算乘积
            result = amount * float(multiplier.strip('*'))
            input_str = f"${result:.2f}"
        input_str = f"${amount:.2f}"
        
    # 删除那些前后没有运算符且前面有数字的字符串，例如 '45 months' -> '45'
    input_str = re.sub(r'(\d+)\s+([a-zA-Z\' ]+)(?![\*\/\-\+\(\)0-9])', r'\1', input_str)
    
    # 处理33.333...3333333333为33.333
    # 寻找省略号的位置
    ellipsis_index = input_str.find('...')
    if ellipsis_index != -1:
        # 找到省略号后面紧跟的数字部分并截断
        end_index = ellipsis_index + 3
        while end_index < len(input_str) and input_str[end_index].isdigit():
            end_index += 1
        
        # 生成新的表达式，省略后面的数字
        input_str = input_str[:ellipsis_index] + input_str[end_index:]
    
    # 重新处理可能由于删除字符串导致的多余空格
    input_str = ' '.join(input_str.split())

    # 删除包含逗号的数字
    input_str = ' '.join([part for part in input_str.split() if ',' not in part])

    # 重新处理可能由于删除字符串导致的多余空格
    input_str = re.sub(r'\s+', ' ', input_str).strip()

    # 删除可能存在的单位
    input_str = simplify_expression(input_str)

    return input_str

def extract_glm_code(text):
    # 使用正则表达式提取代码块
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""  # 如果没有找到匹配项，返回空字符串

def get_llm_calculate_result(input_str, question, content, history, backbone = "tgi", url = TGI_URL):
    # 构建给语言模型的提示
    # prompt = f"""我正在尝试解决一个数学问题，具体问题是：{question}。\n\n 我目前采用的解题步骤如下：{history}。\n\n 现在我正在计算的步骤是：{content}。\n\n 我需要计算一个数学表达式，这个表达式是：{input_str}。请帮我生成可执行的Python代码用于计算这个表达式，注意代码的最终打印输出应该是这个表达式的值或者化简后的表达式，不需要输出任何提示信息或者单位"""
    prompt = f"""I am attempting to solve a math problem with the following specific question:{question} \n\n The steps I am currently using to solve the problem are: {history}. \n\n The step I am calculating now is: {content}. \n\n I need to compute a math expression which is: {input_str}. Please help me generate executable Python code for calculating this expression, noting that the final printout of the code should be the value of the expression or the simplified expression, without outputting any hints or units."""
    response = ""
    for i in range(10):  # 尝试最多10次以获取有效的代码响应
        # 使用ChatGLM模型生成代码
        response = llm_response(prompt=prompt, backbone=backbone, url=url)
        if type(response) == str:
            # print("response:", response)
            code = extract_glm_code(response) # 假设响应即为代码
            stdout, stderr = run_python(code)
            if not stderr:  # 如果没有错误，返回结果
                return stdout, code
            else:
                continue
        else:
            continue
    return "", response  # 经过10次尝试失败后返回错误信息

def judge_equation_need(expr, question, input_str, history, backbone="tgi", url=TGI_URL):
    # 构建给语言模型的提示
    # prompt = f"""我正在尝试解决一个数学问题，具体问题是：{question}。\n\n 我目前采用的解题步骤如下：{history}。\n\n 现在我正在计算的步骤是：{input_str}。\n\n 我需要计算一个数学表达式，这个表达式是：{expr}。请问这个表达式是否正确？\n\n 如果正确请只回答yes。 \n\n 如果不正确，请帮我修正这个表达式，注意表达式应该是可计算的表达式，不能包含任何单位。请返回唯一的正确的表达式，用<<>>包围起来。也就是说，<<>>只出现一次"""
    prompt = f"""I am attempting to solve a math problem with the following specific question:{question} \n\n The steps I am currently using to solve the problem are: {history}. \n\n The step I am currently calculating is: {input_str}. \n\n I need to calculate a math expression which is: {expr}. Is this expression correct? \n\n If it is correct please answer yes only. \n\n If it is not correct, please help me to correct this expression, noting that the expression should be a computable expression and should not contain any units. Please return the only correct expression, surrounded by <<>>. That is, <<>> appears only once."""
    response = ""
    for i in range(10):  # 尝试最多10次以获取有效的代码响应
        # 使用ChatGLM模型生成结果
        response = llm_response(prompt=prompt, backbone = backbone, url=url)
        # 获取response的第一句话或者如果没有符号就是完整的response
        if type(response) == str and len(response) > 0:
            response_first_sentence = response.split(".")[0]
            if response_first_sentence[:3].lower() == "yes" or "yes" in response_first_sentence.lower():  # 如果是一个正确的计算公式
                return 1, expr, response
            else:
                # 尝试修正表达式
                match = re.search(r'<<(.+?)>>', response)
                if match:
                    temp_expr = match.group(1)
                    if temp_expr.strip() != "":
                        return 0, temp_expr, response
                    else:
                        continue
                else:
                    continue
    return 0, "", response  # 经过10次尝试失败后返回错误信息

# 删除小数后面多余的0
def evaluate_and_simplify(expr):
    # 计算表达式的数值结果
    numerical_result = expr.evalf()
    
    # 检查结果是否非常接近某个整数
    if numerical_result.is_Integer:
        return int(numerical_result)  # 结果是整数
    else:
        # 尝试四舍五入看看是否能得到整数
        rounded_result = round(numerical_result)
        if abs(numerical_result - rounded_result) < 1e-10:
            return int(rounded_result)  # 四舍五入后的整数
        else:
            return float(numerical_result)  # 保留为浮点数

def get_sympy_calculate_result(input_str, question, content, history, backbone = "tgi", url = TGI_URL):
    use_sympy_or_llm = 'sympy'
    temp_key = input_str
    intermediate_process = input_str
    input_str = formula_preprocessing(input_str) # 公式预处理，采用规则库
    if input_str != intermediate_process:
        intermediate_process = intermediate_process + "=>" + input_str
    code = ""

    # 定义可能的符号变量，确保它们被识别为符号而不是字符串
    symbols = re.findall(r'[a-zA-Z]+', input_str)
    symbols = set(symbols)  # 去除重复项
    local_dict = {s: sp.symbols(s) for s in symbols}

    try:
        # 将字符串转换为sympy表达式 expr = sp.sympify(input_str)
        # 使用 locals 参数显式地将这些变量定义为符号
        expr = sp.sympify(input_str, locals=local_dict)
        if str(expr) != input_str:
            intermediate_process = intermediate_process + "=>" + str(expr)
        # 计算表达式
        result = expr
        if str(result) != str(expr):
            intermediate_process = intermediate_process + "=>" + str(result)
        # 化简表达式
        simplified_expr = sp.simplify(result)
        if str(simplified_expr) != str(result):
            intermediate_process = intermediate_process + "=>" + str(simplified_expr)
        # 检查结果是否为布尔值
        if isinstance(simplified_expr, bool):  # 直接使用 Python 的内建类型 bool
            return simplified_expr, use_sympy_or_llm, intermediate_process, code
        try:
            # 如果是数学表达式，返回计算结果
            actual_result = simplified_expr.evalf()
            # 删除后面可能存在的多余的0
            actual_result = evaluate_and_simplify(actual_result)
            if str(actual_result) != str(simplified_expr):
                intermediate_process = intermediate_process + "=>" + str(actual_result)
            return actual_result, use_sympy_or_llm, intermediate_process, code
        except Exception as e:
            logging.error(f"the simplified expression is << {temp_key} = {simplified_expr} >> --- incalculable << {input_str} >>: {str(e)}")
            # actual_result = simplified_expr
            use_sympy_or_llm = 'sympy and llm'
            actual_result_temp, code = get_llm_calculate_result(simplified_expr, question, content, history, backbone, url)
            actual_result = formula_preprocessing(actual_result_temp) # 结果预处理，采用规则库
            if actual_result != actual_result_temp:
                intermediate_process = intermediate_process + "\n\n Code simplifued:" + actual_result_temp + "=>" + actual_result
            return actual_result, use_sympy_or_llm, intermediate_process, code
    except Exception as e:
        logging.error(f" the simplified expression is << {temp_key} = {input_str} >> --- Unsimplified << {input_str} >>: {str(e)}")
        simplified_expr = input_str
        # actual_result = simplified_expr
        use_sympy_or_llm = 'sympy and llm'
        actual_result_temp, code = get_llm_calculate_result(simplified_expr, question, content, history, backbone, url)
        actual_result = formula_preprocessing(actual_result_temp) # 结果预处理，采用规则库
        if actual_result != actual_result_temp:
            intermediate_process = intermediate_process + "\n\n Code simplifued:" + actual_result_temp + "=>" + actual_result
     
    return actual_result, use_sympy_or_llm, intermediate_process, code

def check_calculation(info, question, history, backbone="tgi", url=TGI_URL):
    input_str = info['content'] # 获取当前步骤的内容

    info['leftSideOfEqualSign'] = [] # 用于存储等号左侧的表达式的推理过程
    info['rightSideOfEqualSign'] = [] # 用于存储等号右侧的表达式的推理过程
    info['leftSideOfEqual_use_sympy_or_llm'] = [] # 用于存储等号左侧的表达式的计算方式
    info['rightSideOfEqual_use_sympy_or_llm'] = [] # 用于存储等号右侧的表达式的计算方式
    info['leftSideOfEqual_code'] = [] # 用于存储等号左侧的表达式的代码
    info['rightSideOfEqual_code'] = [] # 用于存储等号右侧的表达式的代码
    info['JudgmentStepCalculatedCorrectly'] = [] # 用于存储表达式是否正确计算的判断结果
    info['StepCalculatedCorrectlyResult'] = [] # 用于存储表达式的计算结果
    info['JudgmentStepEquationCorrectly'] = [] # 用于存储表达式是否正确的判断结果
    info['StepEquationCorrectlyFormat'] = [] # 用于存储表达式的正确格式
    info['StepEquationCorrectlyFormatLLMResponse'] = [] # 用于存储表达式的正确格式的LLM响应

    if info.get('equation') is None:
        info['equation'] = [] # 用于存储表达式

        # 使用正则表达式查找计算表达式和结果
        pattern = r"<<(.+?)=(.+?)>>" # <>是必须的, =是必须的
        # pattern = r"<<?(.+?)=(.+?)>>?" # <>是可选的
        matches = re.findall(pattern, input_str)

        # 如果不存在，则需要自行寻找=号，以及其前后的数学表达式
        if len(matches) == 0:
            # 使用正则表达式查找计算表达式和结果
            pattern = r"([\d\s\/*\-+.%]+)=([\d\s\/*\-+.%]+)"
            matches = re.findall(pattern, input_str)

        # 遍历所有匹配项进行检查
        for expr, expected_result in matches: # expr变量会接收表达式的内容（如"20*40"），而expected_result变量会接收表达式的结果（如"800"）。
            # logging.info(f"expr: {expr}, expected_result: {expected_result}")
            # 为什么能根据=号分割，因为=号是必须的
            # 去除头尾的 <<
            expr = expr.lstrip("<")
            # 如果还存在=，保留=前面的，如果不存在=，那就是其本身
            expr = expr.split("=")[0]
            # 去除头尾的 >>
            expected_result = expected_result.rstrip(">")
            # 如果还存在-，则保留=后面的
            expected_result = expected_result.split("=")[-1]

            # 添加表达式
            info['equation'].append(f"{expr}={expected_result}")

            # 判断该表达式所进行的计算是否合适
            judge_eqation_correct, correct_expr, llm_responses = judge_equation_need(expr, question, input_str, history, backbone, url) 

            info['JudgmentStepEquationCorrectly'].append(judge_eqation_correct)
            info['StepEquationCorrectlyFormat'].append(correct_expr)
            info['StepEquationCorrectlyFormatLLMResponse'].append(llm_responses)
            
            # 判断该表达式是否正确计算
            if judge_eqation_correct == 1: # 如果是正确的计算表达式
                # 使用 sympy 计算表达式的结果
                actual_result, use_sympy_or_llm1, intermediate_process1, code1 = get_sympy_calculate_result(expr, question, input_str, history, backbone, url)
                expected_result, use_sympy_or_llm2, intermediate_process2, code2 = get_sympy_calculate_result(expected_result, question, input_str, history, backbone, url)
            
                info['leftSideOfEqualSign'].append(intermediate_process1)
                info['rightSideOfEqualSign'].append(intermediate_process2)
                info['leftSideOfEqual_use_sympy_or_llm'].append(use_sympy_or_llm1)
                info['rightSideOfEqual_use_sympy_or_llm'].append(use_sympy_or_llm2)
                info['leftSideOfEqual_code'].append(code1)
                info['rightSideOfEqual_code'].append(code2)

                # 比较实际结果和期望结果
                if actual_result != expected_result: # sympify(expected_result).evalf()是将expected_result转换为sympy对象并计算其值，evalf()方法返回计算结果。
                    info['JudgmentStepCalculatedCorrectly'].append(0)
                    info['StepCalculatedCorrectlyResult'].append(f"{actual_result}")    
                else:
                    info['JudgmentStepCalculatedCorrectly'].append(1)
                    info['StepCalculatedCorrectlyResult'].append(f"{actual_result}")
            else: # 如果不是正确的计算表达式
                info['JudgmentStepCalculatedCorrectly'].append(0)
                # 如果返回了正确的公式那么可以进行计算
                if correct_expr != "":
                    # 使用 sympy 计算表达式的结果
                    actual_result, use_sympy_or_llm1, intermediate_process1, code1 = get_sympy_calculate_result(correct_expr, question, input_str, history, backbone, url)
                else:
                    # actual_result, use_sympy_or_llm1, intermediate_process1, code1 = get_sympy_calculate_result(expr, question, input_str, history)
                    actual_result, use_sympy_or_llm1, intermediate_process1, code1 = "", "", "", ""

                info['leftSideOfEqualSign'].append(intermediate_process1) # 如果不是正确的计算表达式，那么左边和右边的表达式都是空的,这两个位置是记录公式推导过程的
                info['rightSideOfEqualSign'].append("")
                info['leftSideOfEqual_use_sympy_or_llm'].append(use_sympy_or_llm1)
                info['rightSideOfEqual_use_sympy_or_llm'].append("")
                info['leftSideOfEqual_code'].append(code1)
                info['rightSideOfEqual_code'].append("")

                info['StepCalculatedCorrectlyResult'].append(f"{actual_result}")
    else:
        # 遍历所有匹配项进行检查
        for raw_expr in info['equation']: # expr变量会接收表达式的内容（如"20*40"），而expected_result变量会接收表达式的结果（如"800"）。
            # 找到等号的位置，等号前是expr，等号后是expected_result
            expr = raw_expr
            expected_result = raw_expr
            # 去除头尾的 <<
            expr = expr.lstrip("<")
            # 如果还存在=，保留=前面的，如果不存在=，那就是其本身
            expr = expr.split("=")[0]
            # 去除头尾的 >>
            expected_result = expected_result.rstrip(">")
            # 如果还存在-，则保留=后面的
            expected_result = expected_result.split("=")[-1]

            # 判断该表达式所进行的计算是否合适
            judge_eqation_correct, correct_expr, llm_responses = judge_equation_need(expr, question, input_str, history, backbone, url) 

            info['JudgmentStepEquationCorrectly'].append(judge_eqation_correct)
            info['StepEquationCorrectlyFormat'].append(correct_expr)
            info['StepEquationCorrectlyFormatLLMResponse'].append(llm_responses)
            
            # 判断该表达式是否正确计算
            if judge_eqation_correct == 1: # 如果是正确的计算表达式
                # 使用 sympy 计算表达式的结果
                actual_result, use_sympy_or_llm1, intermediate_process1, code1 = get_sympy_calculate_result(expr, question, input_str, history, backbone, url)
                expected_result, use_sympy_or_llm2, intermediate_process2, code2 = get_sympy_calculate_result(expected_result, question, input_str, history, backbone, url)
            
                info['leftSideOfEqualSign'].append(intermediate_process1)
                info['rightSideOfEqualSign'].append(intermediate_process2)
                info['leftSideOfEqual_use_sympy_or_llm'].append(use_sympy_or_llm1)
                info['rightSideOfEqual_use_sympy_or_llm'].append(use_sympy_or_llm2)
                info['leftSideOfEqual_code'].append(code1)
                info['rightSideOfEqual_code'].append(code2)

                # 比较实际结果和期望结果
                if actual_result != expected_result: # sympify(expected_result).evalf()是将expected_result转换为sympy对象并计算其值，evalf()方法返回计算结果。
                    info['JudgmentStepCalculatedCorrectly'].append(0)
                    info['StepCalculatedCorrectlyResult'].append(f"{actual_result}")    
                else:
                    info['JudgmentStepCalculatedCorrectly'].append(1)
                    info['StepCalculatedCorrectlyResult'].append(f"{actual_result}")
            else: # 如果不是正确的计算表达式
                info['JudgmentStepCalculatedCorrectly'].append(0)
                # 如果返回了正确的公式那么可以进行计算
                if correct_expr != "":
                    # 使用 sympy 计算表达式的结果
                    actual_result, use_sympy_or_llm1, intermediate_process1, code1 = get_sympy_calculate_result(correct_expr, question, input_str, history, backbone, url)
                else:
                    # actual_result, use_sympy_or_llm1, intermediate_process1, code1 = get_sympy_calculate_result(expr, question, input_str, history)
                    actual_result, use_sympy_or_llm1, intermediate_process1, code1 = "", "", "", ""

                info['leftSideOfEqualSign'].append(intermediate_process1) # 如果不是正确的计算表达式，那么左边和右边的表达式都是空的,这两个位置是记录公式推导过程的
                info['rightSideOfEqualSign'].append("")
                info['leftSideOfEqual_use_sympy_or_llm'].append(use_sympy_or_llm1)
                info['rightSideOfEqual_use_sympy_or_llm'].append("")
                info['leftSideOfEqual_code'].append(code1)
                info['rightSideOfEqual_code'].append("")

                info['StepCalculatedCorrectlyResult'].append(f"{actual_result}")

# 串行处理
def process_jsonl_file(source_path, dest_path):
    with open(source_path, 'r', encoding='utf-8') as src_file, \
         open(dest_path, 'w', encoding='utf-8') as dest_file:
        for line in tqdm(src_file, desc='Processing'):
            data = json.loads(line)
            # 遍历每一步的解决方案
            question = data['question']
            history = ""
            if 'solution' in data:
                for step, info in data['solution'].items(): # step变量会接收步骤的名称（如"Step 1"），而info变量会接收与这个步骤名称对应的字典值。
                    # 判断并添加新键
                    check_calculation(info, question, history)
                    temp_content = info['content']
                    if len(info["JudgmentStepCalculatedCorrectly"]) > 0:
                        # 找到info["content"]中错误的部分进行替换
                        # 主要是在字符串中找到<<80*2=1600>>1600比如，然后替换1600>>1600
                        temp_content = replace_calculated_result(info["content"], info["equation"], info['JudgmentStepEquationCorrectly'], info['StepEquationCorrectlyFormat'], info["JudgmentStepCalculatedCorrectly"], info["StepCalculatedCorrectlyResult"])
                    history += f"{step}: {temp_content}\n"
                    info["history"] = history
            # 将修改后的数据写回新的JSONL文件
            json.dump(data, dest_file, ensure_ascii=False)
            dest_file.write('\n')

# 并发处理
def process_line(line, backbone="tgi", url=TGI_URL):
    data = json.loads(line)
    question = data['question']
    history = ""
    history_json = {}
    if 'solution' in data:
        for step, info in data['solution'].items():
            check_calculation(info, question, history, backbone, url)
            temp_content = info['content']
            # 为了方便后续进行本步骤的判断，需要修正历史步骤中的错误部分
            if len(info["JudgmentStepCalculatedCorrectly"]) > 0:
                # 找到info["content"]中错误的部分进行替换
                # 主要是在字符串中找到<<80*2=1600>>1600比如，然后替换1600>>1600
                temp_content = replace_calculated_result(info["content"], info["equation"], info['JudgmentStepEquationCorrectly'], info['StepEquationCorrectlyFormat'], info["JudgmentStepCalculatedCorrectly"], info["StepCalculatedCorrectlyResult"])
            history += f"{step}: {temp_content}\n"
            history_json[step] = temp_content
            # 创建一个history_json的副本，并将其赋值给info['history_json']
            info['history_json'] = history_json.copy()
    # 将处理后的数据转换为字符串以便写入文件
    return json.dumps(data, ensure_ascii=False) + '\n'

def process_jsonl_file_concurrent(source_path, dest_path):
    # 读取文件的所有行
    with open(source_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    results = []

    # 使用 ThreadPoolExecutor 来并发处理每一行
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 提交所有行到线程池
        future_to_line = {executor.submit(process_line, line): line for line in tqdm(lines, desc='Processing')}

        # 使用tqdm创建进度条
        with tqdm(total=len(future_to_line), desc='Processing lines') as progress:
            # 收集处理结果
            for future in concurrent.futures.as_completed(future_to_line):
                results.append(future.result())
                progress.update(1)  # 更新进度条

    # 写入结果到目标文件
    with open(dest_path, 'w', encoding='utf-8') as file:
        file.writelines(results)


def process_jsonl_file_concurrent2(source_path, dest_path, max_workers = 10, backbone="tgi", url=TGI_URL):
    processed_questions = set()
    
    # 读取已处理的问题
    if os.path.exists(dest_path):
        with open(dest_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    processed_questions.add(data['question'])
                except json.JSONDecodeError:
                    continue

    # 读取源文件并过滤已处理的行
    lines_to_process = []
    with open(source_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line)
                if data['question'] not in processed_questions:
                    lines_to_process.append(line)
            except json.JSONDecodeError:
                continue
    
    # 使用 ThreadPoolExecutor 来并发处理每一行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_line, line, backbone, url): line for line in lines_to_process}
        
        with tqdm(total=len(futures), desc='Processing lines') as progress:
            with open(dest_path, 'a', encoding='utf-8') as output_file:
                for future in as_completed(futures):
                    result = future.result()
                    output_file.write(result)
                    output_file.flush()  # 确保每次写入后立即保存
                    progress.update(1)  # 更新进度条

@time_it
def Step3_JudgmentStepCalculatedCorrectly(source_folder, target_folder, max_workers = 10, backbone="tgi", url=TGI_URL):
    
    print("第三步判断单步计算是否正确,包括公式的正确性和结果的正确性……")

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.jsonl'):
            source_path = os.path.join(source_folder, filename)
            print("正在处理文件:", source_path)
            dest_path = os.path.join(target_folder, filename)
            # 串行处理
            # process_jsonl_file(source_path, dest_path)
            # 并行处理
            # process_jsonl_file_concurrent(source_path, dest_path)
            # 并行处理，保留检查点
            process_jsonl_file_concurrent2(source_path, dest_path, max_workers, backbone, url)

            # 可视化结果输出，用于debug
            Check1_JsonlVisualization(dest_path) 
# 使用方法：
def main2():
    code_test_state = True
    base_folder = "F://code//github//ChatGLM-MathV2"
    dataset_name = "peiyi9979_Math_Shepherd"
    source_folder = base_folder + '//raw_data//' + dataset_name

    mid_name = base_folder + '//data//' + dataset_name

    if code_test_state:
        get_data_for_codeTest(source_folder, new_folder_suffix='_for_codeTest', num_points=10)
        source_folder = source_folder + "_for_codeTest"

        target_folder1 = mid_name + "_for_codeTest" + "_Step1_SplitByRow_forMathShepherd"
        target_folder2 = mid_name + "_for_codeTest" + "_Step2_IsCalculationOrReasoning"
        target_folder3 = mid_name + "_for_codeTest" + "_Step3_JudgmentStepCalculatedCorrectly"
    else:
        target_folder1 = mid_name + "_Step1_SplitByRow_forMathShepherd"
        target_folder2 = mid_name + "_Step2_IsCalculationOrReasoning"
        target_folder3 = mid_name + "_Step3_JudgmentStepCalculatedCorrectly"

    Step1_SplitByRow_forMathShepherd(source_folder, target_folder1)
    Step2_IsCalculationOrReasoning(target_folder1, target_folder2)
    Step3_JudgmentStepCalculatedCorrectly(target_folder2, target_folder3)

def main():
    if len(sys.argv) > 5:
        source_folder = sys.argv[1]
        target_folder = sys.argv[2]
        max_workers = int(sys.argv[3])
        backbone = sys.argv[4]
        url = sys.argv[5]
    else:
        backbone = "tgi"
        url = TGI_URL
        source_folder = 'F://code//github//ChatGLM-MathV2//data//test_data100//front_step2'
        target_folder = 'F://code//github//ChatGLM-MathV2//data//test_data100//front_step3'    
        max_workers = 10
    Step3_JudgmentStepCalculatedCorrectly(source_folder, target_folder, max_workers=max_workers, backbone=backbone, url=url)

if __name__ == '__main__':
    main()