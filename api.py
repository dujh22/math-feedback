import re
import json
import sympy as sp
import subprocess

# 如果打开下面两行，命令行会自动输出代码执行的全部日志
import hunter # 用于调试
hunter.trace(module=__name__) # 用于调试

# 该部分用于设置调用的LLM相关信息
import llm.config as config
import openai
# 设定API密钥和基本URL
openai.api_key = config.GPT_API_KEY
openai.api_base = config.GPT_BASE_URL
from chatglm import ChatGLM
ChatGLM = ChatGLM()
USE_GLM_OR_GPT = 'glm'
# 这里设置使用的llm进行生成，注意在本项目中只有这里一个地方进行相关设置
def llm_response(prompt, use_glm_or_gpt = USE_GLM_OR_GPT):
    response = "ERROR for LLM"
    for i in range(10):
        if use_glm_or_gpt == 'glm':
            try:
                response = ChatGLM.generate(prompt)
                return response
            except:
                continue
        else:
            try:
                # 构造messages
                messages = [{"role": "user", "content": prompt}]
                # 调用GPT接口
                # model = "gpt-3.5-turbo"
                model = "gpt-4-1106-preview"
                chat_completion = openai.ChatCompletion.create(model=model, messages = messages)
                response = chat_completion.choices[0].message.content
                return response
            except:
                continue
    return response

def run_python(code, timeout=None):
    result = subprocess.run(['python', '-c', code], capture_output=True, text=True, timeout=timeout)
    stdout = result.stdout
    stderr = result.stderr
    return stdout, stderr

def all_alpha(str):
    if str == 'x': # 特例
        return False
    if all(temp.isalpha() for temp in str):
        return True
    else:
        return False

def highlight_equations(text):
    parts2 = text.split(' ')
    if len(parts2) == 1:
        return text  # 没有等号，返回原文本
    
    # 预处理，如果数字后面紧跟着一个字母串，可以删去这个字母串
    parts = []
    i = 0
    while i < len(parts2):
        parts.append(parts2[i])
        if parts2[i].isdigit():
            if i + 1 < len(parts2) and all_alpha(parts2[i + 1]):
                i += 2
            else:
                i += 1
        else:
            i += 1

    result = []
    last_pos = 0
    
    for i in range(len(parts)):
        # 找到=所在的位置
        if '=' not in parts[i]:
            continue
        else:
            start_pos = i
            end_pos = i
            if start_pos > 0:
                while all_alpha(parts[start_pos - 1]) == False:
                    start_pos -= 1
                    if start_pos == 0:
                        break
            if end_pos < len(parts) - 1:
                while all_alpha(parts[end_pos + 1]) == False:
                    end_pos += 1
                    if end_pos == len(parts) - 1:
                        break
            # print(parts[start_pos:end_pos + 1])
            equation = ' '.join(parts[start_pos:end_pos + 1])
            equation = equation.replace('=', f'=<<{equation}>>')
            for item in range(last_pos, start_pos):
                result.append(parts[item])
            result.append(equation)
            last_pos = end_pos + 1
    
    # 添加最后一个没有等号的部分
    for item in range(last_pos, len(parts)):
        result.append(parts[item])

    return ' '.join(result)

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

def SplitByRow(data):
    # 初始化新的JSON格式
    new_json = {
        "question": data["question"],
        "solution": {},
    }

    # 切分成独立的步骤
    solutions = split_response(data['solution'])

    # 处理每个解决方案部分
    for i, solution in enumerate(solutions):
        if solution.strip():  # 确保切割后的文本不为空
            new_json["solution"][f"Step {i+1}"] = {
                "content": solution.strip(),
                "label": 1  # 默认标签为1
            }

    # 处理每个解决方案部分的数学公式高亮
    for step, info in new_json["solution"].items():
        temp_content = info["content"]
        info["content"] = highlight_equations(temp_content)
    
    # 返回新的JSON格式
    return new_json

def is_calculation(content):
    # 检查是否存在常见的计算符号
    if re.search(r'[\+\-\*/=%^]', content):
        return 1
    # 检查括号内是否存在计算符号，需要使用更复杂的正则表达式
    if re.search(r'\([^)]*[\+\-\*/=%^][^)]*\)', content):
        return 1
    return 0

def IsCalculationOrReasoning(data):
    # 遍历每一步的解决方案
    if 'solution' in data:
        for step, info in data['solution'].items(): # step变量会接收步骤的名称（如"Step 1"），而info变量会接收与这个步骤名称对应的字典值。
            # 判断并添加新键
            info['is_calculation_or_reasoning'] = is_calculation(info['content'])
    # 将修改后的数据写回新的JSONL文件
    return data

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
            
            # 构造用于搜索和替换的正则表达式
            search_pattern = re.escape(variable) + r'\s*=\s*' + re.escape(original_result)
            replace_pattern = f'{variable} = {result[i]}'
            
            # 替换等式
            content = re.sub(search_pattern, replace_pattern, content)
            
            # 替换全文中的原结果
            content = re.sub(r'\b' + re.escape(original_result) + r'\b', result[i], content)

            # 等式本身是不正确的，还需要修改等式本身
            if equations_need[i] == 0:
                if equations_correct_format[i] != "ERROR: LLM cannot correct the formula":
                    content = content.replace(variable, str(equations_correct_format[i]))
                else:
                    temp_variable = variable + "(The formula may be wrong)"
                    content = content.replace(variable, temp_variable)
                
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

def get_llm_calculate_result(input_str, question, content, history):
    # 构建给语言模型的提示
    # prompt = f"""我正在尝试解决一个数学问题，具体问题是：{question}。\n\n 我目前采用的解题步骤如下：{history}。\n\n 现在我正在计算的步骤是：{content}。\n\n 我需要计算一个数学表达式，这个表达式是：{input_str}。请帮我生成可执行的Python代码用于计算这个表达式，注意代码的最终打印输出应该是这个表达式的值或者化简后的表达式，不需要输出任何提示信息或者单位"""
    prompt = f"""I am attempting to solve a math problem with the following specific question:{question} \n\n The steps I am currently using to solve the problem are: {history}. \n\n The step I am calculating now is: {content}. \n\n I need to compute a math expression which is: {input_str}. Please help me generate executable Python code for calculating this expression, noting that the final printout of the code should be the value of the expression or the simplified expression, without outputting any hints or units."""
    response = ""
    for i in range(10):  # 尝试最多10次以获取有效的代码响应
        # 使用ChatGLM模型生成代码
        response = llm_response(prompt)
        # print("response:", response)
        code = extract_glm_code(response) # 假设响应即为代码
        stdout, stderr = run_python(code)
        if not stderr:  # 如果没有错误，返回结果
            return stdout, code
    return "ERROR: Python scripts not running", response  # 经过10次尝试失败后返回错误信息

def judge_equation_need(expr, question, input_str, history):
    # 构建给语言模型的提示
    # prompt = f"""我正在尝试解决一个数学问题，具体问题是：{question}。\n\n 我目前采用的解题步骤如下：{history}。\n\n 现在我正在计算的步骤是：{input_str}。\n\n 我需要计算一个数学表达式，这个表达式是：{expr}。请问这个表达式是否正确？\n\n 如果正确请只回答yes。 \n\n 如果不正确，请帮我修正这个表达式，注意表达式应该是可计算的表达式，不能包含任何单位。请返回唯一的正确的表达式，用<<>>包围起来。也就是说，<<>>只出现一次"""
    prompt = f"""I am attempting to solve a math problem with the following specific question:{question} \n\n The steps I am currently using to solve the problem are: {history}. \n\n The step I am currently calculating is: {input_str}. \n\n I need to calculate a math expression which is: {expr}. Is this expression correct? \n\n If it is correct please answer yes only. \n\n If it is not correct, please help me to correct this expression, noting that the expression should be a computable expression and should not contain any units. Please return the only correct expression, surrounded by <<>>. That is, <<>> appears only once."""
    response = ""
    for i in range(10):  # 尝试最多10次以获取有效的代码响应
        # 使用ChatGLM模型生成结果
        response = llm_response(prompt)
        # 获取response的第一句话或者如果没有符号就是完整的response
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
    return 0, "ERROR: LLM cannot correct the formula", response  # 经过10次尝试失败后返回错误信息

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
        
def get_sympy_calculate_result(input_str, question, content, history):
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
            # actual_result = simplified_expr
            use_sympy_or_llm = 'sympy and llm'
            actual_result_temp, code = get_llm_calculate_result(simplified_expr, question, content, history)
            actual_result = formula_preprocessing(actual_result_temp) # 结果预处理，采用规则库
            if actual_result != actual_result_temp:
                intermediate_process = intermediate_process + "\n\n Code simplifued:" + actual_result_temp + "=>" + actual_result
            return actual_result, use_sympy_or_llm, intermediate_process, code
    except Exception as e:
        simplified_expr = input_str
        # actual_result = simplified_expr
        use_sympy_or_llm = 'sympy and llm'
        actual_result_temp, code = get_llm_calculate_result(simplified_expr, question, content, history)
        actual_result = formula_preprocessing(actual_result_temp) # 结果预处理，采用规则库
        if actual_result != actual_result_temp:
            intermediate_process = intermediate_process + "\n\n Code simplifued:" + actual_result_temp + "=>" + actual_result
     
    return actual_result, use_sympy_or_llm, intermediate_process, code

def check_calculation(info, question, history):
    input_str = info['content']
    info['equation'] = []
    info['leftSideOfEqualSign'] = []
    info['rightSideOfEqualSign'] = []
    info['leftSideOfEqual_use_sympy_or_llm'] = []
    info['rightSideOfEqual_use_sympy_or_llm'] = []
    info['leftSideOfEqual_code'] = []
    info['rightSideOfEqual_code'] = []
    info['JudgmentStepCalculatedCorrectly'] = []
    info['StepCalculatedCorrectlyResult'] = []
    info['JudgmentStepEquationCorrectly'] = []
    info['StepEquationCorrectlyFormat'] = []
    info['StepEquationCorrectlyFormatLLMResponse'] = []

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
        judge_eqation_correct, correct_expr, llm_response = judge_equation_need(expr, question, input_str, history) 

        info['JudgmentStepEquationCorrectly'].append(judge_eqation_correct)
        info['StepEquationCorrectlyFormat'].append(correct_expr)
        info['StepEquationCorrectlyFormatLLMResponse'].append(llm_response)
        
        # 判断该表达式是否正确计算
        if judge_eqation_correct == 1: # 如果是正确的计算表达式
            # 使用 sympy 计算表达式的结果
            actual_result, use_sympy_or_llm1, intermediate_process1, code1 = get_sympy_calculate_result(expr, question, input_str, history)
            expected_result, use_sympy_or_llm2, intermediate_process2, code2 = get_sympy_calculate_result(expected_result, question, input_str, history)
        
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
            # 如果返回了正确的公式没那么可以进行计算
            if correct_expr != "ERROR: LLM cannot correct the formula":
                # 使用 sympy 计算表达式的结果
                actual_result, use_sympy_or_llm1, intermediate_process1, code1 = get_sympy_calculate_result(correct_expr, question, input_str, history)
            else:
                actual_result, use_sympy_or_llm1, intermediate_process1, code1 = "ERROR: LLM cannot correct the formula", "sympy and llm", "ERROR: LLM cannot correct the formula", "ERROR: LLM cannot correct the formula"

            info['leftSideOfEqualSign'].append(intermediate_process1) # 如果不是正确的计算表达式，那么左边和右边的表达式都是空的,这两个位置是记录公式推导过程的
            info['rightSideOfEqualSign'].append("")
            info['leftSideOfEqual_use_sympy_or_llm'].append(use_sympy_or_llm1)
            info['rightSideOfEqual_use_sympy_or_llm'].append("")
            info['leftSideOfEqual_code'].append(code1)
            info['rightSideOfEqual_code'].append("")

            info['StepCalculatedCorrectlyResult'].append(f"{actual_result}")

def JudgmentStepCalculatedCorrectly(data):
    question = data['question']
    history = ""
    history_json = {}
    if 'solution' in data:
        for step, info in data['solution'].items():
            check_calculation(info, question, history)
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
    return data

# 判断单步推理是否正确
def check_reasoning(per_step, content, question, history):
    # promt = f"""我正在尝试解决一个数学问题，具体问题是：“{question}”。\n\n 我之前的解题步骤如下：“{history}” \n\n 现在我正在推理这一步是：“{per_step}”，具体推理内容是：“{content}”。\n\n 请评估我这一步的推理是否正确。\n\n 如果我目前的推理步骤正确并且与问题相关，只需要回答“yes”。\n\n 如果这一步推理错误或者不相关，需要你进行修正，并直接提供正确或更相关的推理步骤，用<<>>包围起来。"""
    prompt = f"""I am trying to solve a math problem, the specific question is: "{question}". \n\n My previous step in solving the problem was as follows:"{history}" \n\n Now I'm reasoning that this step is:"{per_step}" and the specific reasoning is:"{content}". \n\n Please evaluate if my reasoning in this step is correct. \n\n If my current reasoning step is correct and relevant to the question, just answer "yes". \n\n If this step of reasoning is incorrect or irrelevant, you are required to correct it and provide the correct or more relevant step of reasoning directly, surrounded by <<>>."""
    for i in range(10):
        # response = gpt_generate(prompt)
        response = llm_response(prompt)  # 调用生成方法
        # 获取response的第一句话或者如果没有符号就是完整的response
        response_first_sentence = response.split(".")[0]
        # 提取 response 的前三个字符，并将它们转换成小写来进行判断。
        if response_first_sentence[:3].lower() == "yes" or "yes" in response_first_sentence.lower():  
            return 1, response
        else:
            # 尝试修正表达式
            match = re.search(r'<<(.+?)>>', response)
            if match:
                return 0, match.group(1)
    return 0, "Error: LLM cannot generate correct reasoning."

def JudgmentStepReasoningCorrectly(data):
    if 'solution' in data:
        # 获取历史信息
        history = ""
        question = data['question']
        for step, info in data['solution'].items(): # step变量会接收步骤的名称（如"Step 1"），而info变量会接收与这个步骤名称对应的字典值。
            history_json = info['history_json']
            if info['is_calculation_or_reasoning'] == 1: # 如果是计算步
                info['JudgmentStepReasoningCorrectly'], info['StepReasoningCorrectlyResult'] = 1, "This is a calculation step."
            else:
                # 判断并添加新键
                info['JudgmentStepReasoningCorrectly'], info['StepReasoningCorrectlyResult'] = check_reasoning(step, info['content'], question, history)
                # 添加到历史信息中
                if info['JudgmentStepReasoningCorrectly'] == 1:
                    history_json[step] = info['content']
                else:
                    history_json[step] = info['StepReasoningCorrectlyResult']
            history += f"{step}: {history_json[step]}\n"

    # 将处理后的数据转换为字符串以便写入文件
    return data

def out_to_file(data):
    results = [json.dumps(data, ensure_ascii=False) + '\n']
    with open("output.jsonl", 'w', encoding='utf-8') as file:
        file.writelines(results)

def postprocess(data):
    # 只保留需要的部分
    need_result = {}
    need_result["question"] = data["question"]
    need_result['solution'] = {}
    temp_history = {}
    for step, info in data["solution"].items():
        need_result['solution'][step] = {
            "content": info["content"],
            'is_calculation_or_reasoning': info['is_calculation_or_reasoning'],
            "JudgmentStepCalculatedCorrectly": info["JudgmentStepCalculatedCorrectly"],
            "JudgmentStepEquationCorrectly": info["JudgmentStepEquationCorrectly"],
            "JudgmentStepReasoningCorrectly": info["JudgmentStepReasoningCorrectly"],
            "StepCalculatedCorrectlyResult": info["StepCalculatedCorrectlyResult"],
            "StepEquationCorrectlyFormat": info["StepEquationCorrectlyFormat"],
            "StepReasoningCorrectlyResult": info["StepReasoningCorrectlyResult"]
        }
        temp_history = info["history_json"]
    need_result['modifiedResult'] = temp_history
    return need_result

def api(question, solution):
    data = {"question": question, "solution": solution}
    data1 = SplitByRow(data) # 数据分行与公式高亮
    data2 = IsCalculationOrReasoning(data1) # 判断计算步还是推理步
    data3 = JudgmentStepCalculatedCorrectly(data2) # 针对计算步的自动标注
    data4 = JudgmentStepReasoningCorrectly(data3) # 针对推理步的自动标注
    out_data = data4

    # 如果全量输出，则关闭下面一行，否则只输出必要信息
    out_data = postprocess(data4)

    # 如果有到导出文件的必要，可打开下面一行
    out_to_file(out_data)

    # 返回处理后的数据
    return json.dumps(out_data, ensure_ascii=False)

def main():
    question = "Janet pays $40/hour for 3 hours per week of clarinet lessons and $28/hour for 5 hours a week of piano lessons. How much more does she spend on piano lessons than clarinet lessons in a year?"
    solution = "Step 1: Janet spends 3 hours + 5 hours = 8 hours per week on music lessons. \nStep 2: She spends 40 * 3 = 120 on clarinet lessons per week. \nStep 3: She spends 28 * 5 = 140 on piano lessons per week. \nStep 4: Janet spends 120 + 140 = 260 on music lessons per week. \nStep 5: She spends 260 * 52 = 13520 on music lessons in a year. The answer is: 13520 "
    result = api(question, solution)
    print(result)

if __name__ == '__main__':
    main()