'''
Created on Apr 13, 2020

@author: monky
'''


def get_symbollist(datapath=''):
    """加载拼音符号列表，用于标记符号,返回一个列表list类型变量
        eg. ['a1', 'a2', 'a3',...]
    """
    if(datapath != ''):
        if(datapath[-1]!='/' or datapath[-1]!='\\'):
            datapath = datapath + '/'
    
    txt_obj=open(datapath + 'dict.txt','r',encoding='UTF-8') # 打开文件并读入
    txt_text=txt_obj.read()
    txt_lines=txt_text.split('\n') # 文本分割
    list_symbol=[] # 初始化符号列表
    for i in txt_lines:
        if(i!=''):
            txt_l=i.split('\t')
            list_symbol.append(txt_l[0])
    txt_obj.close()
    list_symbol.append('_')
    #SymbolNum = len(list_symbol)
    return list_symbol


def get_pinyin_dict(dictfile):
    """加载拼音dict，返回dict结构
        eg. {'lia3': ['俩'], 'reng2': ['仍'], 'hei1': ['黑', '嘿', '嗨'], ... }
    """
    txt_obj = open(dictfile, 'r', encoding='UTF-8') # 打开文件并读入
    txt_text = txt_obj.read()
    txt_obj.close()
    txt_lines = txt_text.split('\n') # 文本分割
    dict_pinyin = {} # 初始化符号字典
    for i in txt_lines:
        list_symbol=[] # 初始化符号列表
        if(i!=''):
            txt_l=i.split('\t')
            pinyin = txt_l[0]
            for word in txt_l[1]:
                list_symbol.append(word)
        dict_pinyin[pinyin] = list_symbol
    return dict_pinyin


def get_language_model(modelfile):
    """加载语音模型，返回dict结构
        eg. {'游戏': ‘6459’, '市场': ‘6153’,  ... }
    """
    txt_obj = open(modelfile, 'r', encoding='UTF-8') # 打开文件并读入
    txt_text = txt_obj.read()
    txt_obj.close()
    txt_lines = txt_text.split('\n') # 文本分割
    
    dict_model = {} # 初始化符号字典
    for i in txt_lines:
        if(i!=''):
            txt_l=i.split('\t')
            if(len(txt_l) == 1):
                continue
            #print(txt_l)
            dict_model[txt_l[0]] = txt_l[1]
    return dict_model


import difflib

def get_editdistance(str1, str2):
    """计算编辑距离
        :param str1 类型为list
        :param str2 类型为list
    """
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        #print('{:7} a[{}: {}] --> b[{}: {}] {} --> {}'.format(tag, i1, i2, j1, j2, str1[i1: i2], str2[j1: j2]))
        if tag == 'replace':
            leven_cost += max(i2-i1, j2-j1)
        elif tag == 'insert':
            leven_cost += (j2-j1)
        elif tag == 'delete':
            leven_cost += (i2-i1)
    return leven_cost







