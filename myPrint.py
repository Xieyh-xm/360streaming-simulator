"""
myPrint.py 文件
打印语句，可通过flag进行debug print的全局关闭或打开
"""

import os


def print_debug(*args, **kwargs):
    # flag = True
    flag = False
    if flag:
        print(*args, **kwargs)
    else:
        pass
