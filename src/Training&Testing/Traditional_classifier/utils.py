# -*- encoding = utf-8 -*-
"""
@description: 
@date: 2023/5/11 17:32
@File : utils
@Software : PyCharm
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def solve(path, typ):
    lbs = []
    def read_file(pt):
        if not os.path.exists(pt):
            return
        with open(pt, 'r') as fp:
            lines = fp.read().split('\n')
            for line in lines:
                if line == '':
                    break
                ranges = [int(x) for x in line.split(',')[-2:]]
                flag = 0
                with open(path + '\\extract_range.csv', 'r') as er:
                    ls = er.read().split('\n')
                    for li in ls:
                        if li == '':
                            continue
                        rgs = [int(x) for x in li.split(',')]
                        if (rgs[0] >= ranges[0] and rgs[1] <= ranges[1]) or (rgs[0] <= ranges[0] and rgs[1] >= ranges[1]):
                            flag = 1
                            break

                if flag == 1:
                    lbs.append(1)
                else:
                    lbs.append(0)
    read_file(path + '\\method_range.csv')
    read_file(path + '\\field_range.csv')
    ebs = []
    with open(path + '\\method_embedding.csv', 'r') as me:
        lines = me.read().split('\n')
        for line in lines:
            if line == '':
                continue
            li = [float(x) for x in line.split(',')]
            ebs.append(li)
    # codeBERT codeGPT codeT5 coTexT graphCodeBERT PLBART
    if os.path.exists(path + '\\codeEmbedding\\' + typ + '.csv'):
        with open(path + '\\codeEmbedding\\' + typ + '.csv', 'r') as fe:
            lines = fe.read().split('\n')
            for line in lines:
                if line == '':
                    continue
                li = [float(x) for x in line.split(',')]
                ebs.append(li)
    return ebs, lbs
