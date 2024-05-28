import re

def highlight_equations(text):
    text2 = text
    text3 = ""
    
    i = 0
    while i < len(text2): 
        if text2[i] == '\n':
            text3 = text3 + text2[i]
            i += 1
        elif text2[i].isdigit():
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
                            text3 = text3 + text2[i] + ' ' + '*' + '(' +  + ' ' + text2[i + 2]
                            i += 3
                        else:
                            text3 = text3 + text2[i]   + ' ' + '('
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
                            if text3[i - 1].is_digit():
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
    for i in range(1, len(text3)):
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
    for i in range(1, len(text3) - 1):
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

    return text3

# 测试
text = "There are 20 x 40 = 8000 black seeds.\nThere are 20 x 40 = 8000 white seeds.\nSo, the total number of seeds is 8000 + 8000 = 16000. The answer is: 16000"
print(highlight_equations(text))