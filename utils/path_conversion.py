# 将路径中的\转换为//

import os
import sys

def convert_backslashes(path):
    return path.replace('\\', '//')

if __name__ == '__main__':
    path = input('Enter a path: ')
    print(convert_backslashes(path))