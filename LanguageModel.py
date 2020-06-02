# -*- coding: utf-8 -*-
'''
Created on May 28, 2020

@author: monky
'''
from utils import get_pinyin_dict, get_language_model


class LanguageModel():
    
    def __init__(self, modelpath):
        self.modelpath = modelpath
        
    
    def load_model(self):
        """ 加载词典，1元和2元语音模型
        """
        self.dict_pinyin = get_pinyin_dict('dict.txt')
        self.model1 = get_language_model(self.modelpath + 'language_model1.txt')
        self.model2 = get_language_model(self.modelpath + 'language_model2.txt')
        model = (self.dict_pinyin, self.model1, self.model2 )
        return model
    
    
    def speech2text(self, list_pinyin):
        """ 语音转文本
            :param list_pinyin: 声学模型返回的拼音列表结果
        """
        if len(list_pinyin) == 0:
            return ''
        str_result = ''                 # 返回最终结果
        list_pinyin_remain = []         # 存储剩余的pinyin
        list_pinyin_tmp = list_pinyin  
        # 解码
        while len(list_pinyin_tmp) > 0:
            list_result_tmp = self.decode(list_pinyin_tmp)
            
            if(len(list_result_tmp) > 0): # 解码有结果
                str_result = str_result + list_result_tmp[0][0]
                
            while(len(list_result_tmp) == 0): # 解码没结果
                # 插入最后一个拼音
                list_pinyin_remain.insert(0, list_pinyin_tmp[-1])
                # 删除最后一个拼音
                list_pinyin_tmp = list_pinyin_tmp[:-1]
                # 再次进行拼音转汉字解码
                list_result_tmp = self.decode(list_pinyin_tmp)
                
                if(len(list_result_tmp) > 0):
                    # 将得到的结果加入进来
                    str_result = str_result + list_result_tmp[0][0]
                
            # 将剩余的结果补回来
            list_pinyin_tmp = list_pinyin_remain
            list_pinyin_remain = [] # 清空

        #print(str_result)
        return str_result
        
        
    def decode(self, list_pinyin):
        """ 拼音汉字解码：viterbi
            :param list_pinyin: 声学模型返回的拼音列表结果
        """
        # 文本结果
        list_words = []
        num_pinyin = len(list_pinyin)
        
        for i in range(num_pinyin):
            charlist_ch = ''
            if list_pinyin[i] in self.dict_pinyin:
                # 获取该拼音对应的所有中文字
                charlist_ch = self.dict_pinyin[list_pinyin[i]]
            else:
                break
            
            # 处理第一个中文字
            if i == 0 :
                for j in range(len(charlist_ch)):
                    #charch_prob = ['', 0.0]
                    # 设定马尔科夫模型状态初概率 1.0
                    charch_prob = [charlist_ch[j], 1.0]
                    # 添加可能的句子列表
                    list_words.append(charch_prob)
                continue
            else:
                list_words_2 = []
                for j in range(len(list_words)):
                    for k in range(len(charlist_ch)):
                        # 把现有的每一条短语取出来
                        charch_prob = list(list_words[j])
                        # 尝试按照下一个音可能对应的全部的字进行组合
                        charch_prob[0] = charch_prob[0] + charlist_ch[k]
                        # 取出用于计算的最后两个字
                        tmp_words = charch_prob[0][-2:]
                        # 判断它们是不是再状态转移表里
                        if tmp_words in self.model2 :
                            # n-gram 在当前概率上乘转移概率，公式化简后为第n-1和n个字出现的次数除以第n-1个字出现的次数
                            charch_prob[1] = charch_prob[1] * float(self.model2[tmp_words]) / float(self.model1[tmp_words[-2]])  
                        else:
                            charch_prob[1] = 0.0
                            continue
                        list_words_2.append(charch_prob)
                list_words = list_words_2
        
        # 查找最大概率中文串    
        for i in range(len(list_words)):
            for j in range(i+1, len(list_words)):
                if list_words[i][1] < list_words[j][1]:
                    tmp = list_words[i]
                    list_words[i] = list_words[j]
                    list_words[j] = tmp
                    
        #print(list_words)
        return list_words
        
        
        
if __name__ == '__main__':
    lm = LanguageModel('model_language/')
    lm.load_model()
    list_pinyin = ['kao3','yan2', 'ying1', 'yu3']
    list_pinyin = ['hong2','lei2', 'ba1', 'lei2']
    list_pinyin = ['zhong1','guo2', 'ren2', 'min2']
    list_pinyin = ['yun4', 'suan4', 'shi2', 'jian1', 'bu4', 'yi1', 'ding4', 'man4']
    list_pinyin = ['cai4', 'zuo4', 'hao3', 'le5', 'yi4', 'wan3', 'qing1',
                    'zheng1', 'wu3', 'chang1', 'yu2', 
                    'yi4', 'wan3', 'fan1', 'qie2', 'chao3', 'ji1', 'dan4',
                     'yi4', 'wan3', 'zha4', 'cai4', 'gan1', 'zi4', 'chao3', 'rou4', 'si1']

    result = lm.speech2text(list_pinyin)
    print(result)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        