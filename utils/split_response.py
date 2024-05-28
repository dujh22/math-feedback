import re
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

def main():
    # 测试样例列表
    test_cases = [
        "First step.\n\nSecond step.\n\nThird step.",
        "Step one includes these elements:\n1. First element\n2. Second element\nEnd of step one.",
        "Single step with no breaks.",
        "This is a sentence with a number 3.14 in it.",
        "Example with multiple\nbreaks but not double breaks.\nAnother line.",
        "Another\n\nExample\n\nWith double\n\nBreaks.",
        "Test without proper punctuation but has breaks\nAnother test line\nAnd another line",
        "Mixing two\n\nbreak types.\nSingle break here.",
        "No breaks and no final period",
        "Very complicated example:\n1. First point.\n2. Second point.\n\nThis concludes our example."
    ]

    # 运行测试并打印输出结果
    for idx, test in enumerate(test_cases):
        print(f"Test Case {idx+1}:")
        result = split_response(test)
        for step in result:
            print(f"  - {step}")
        print()  # 为了更好的可读性添加一个空行

if __name__ == "__main__":
    main()