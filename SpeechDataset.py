'''
Created on Apr 14, 2020

@author: monky
'''

import csv
import numpy as np

from feat import read_wav_data, get_spectrogram_feature 
from utils import get_symbollist


class SpeechDataset():
    
    def __init__(self, datapath=''):
        self.datapath = datapath
        self.list_symbol = get_symbollist()
        
        # 读取csv到list
        self.data_list, self.label_list = self.read_csv()
        
    def read_csv(self):
        """读取csv文件放入list
        
            :return data_list
            :return label_list
        """
        data_list = []
        label_list = []
        
        with open(self.datapath, 'r') as f:
            reader = csv.reader(f)
            row_index = 0
            for row in reader:
                # 表头不算
                if row_index == 0:
                    row_index += 1
                    continue
                data_list.append(row[0].split()[0])
                label_list.append(row[0].split()[2:])
                
        return data_list, label_list
        
    
    def generate_data(self, batch_size):
        """数据产生器
           https://www.it1352.com/949933.html
           :param batch_size:每批数据量
        """
        row_index = 0
        
        while True:
            X_input = np.zeros((batch_size, 1600, 200, 1), dtype = np.float)
            Y_ture = np.zeros((batch_size, 64), dtype=np.int16)
            input_length = []
            label_length = []
            
            for b in range(batch_size):
                # 如果一个epoch训练完成，则开始随机打乱一次.
                # 好像没有奏效，异步运行。结果是每一个epoch会从中间打乱一次
                if row_index == len(self.data_list):
                    row_index = 0
                    state = np.random.get_state()
                    np.random.shuffle(self.data_list)
                    np.random.set_state(state)
                    np.random.shuffle(self.label_list)
                    
                # 1个样本标签 wav: data_list[row_index] label: label_list[row_index]
                wave_data, fs = read_wav_data(self.data_list[row_index])
                spec_feat = get_spectrogram_feature(wave_data, fs)
                
                data_input = spec_feat.reshape(spec_feat.shape[0], spec_feat.shape[1], 1)
                data_label = []
                
                
                # 写入pinyin的label下标
                for i in self.label_list[row_index]:
                    data_label.append(self.list_symbol.index(i))
                
                # 将数据写入输入输出 (B, 1600, 200, 1)  (B, 64)
                X_input[b, 0:len(data_input)] = data_input
                Y_ture[b, 0:len(data_label)] = data_label
                
                input_length.append(data_input.shape[0] // 8 + data_input.shape[0] % 8)
                label_length.append(len(data_label))
                
                row_index += 1
                
            input_length = np.array([input_length]).T  # 列向量
            label_length = np.array([label_length]).T  
                
            yield [X_input, Y_ture, input_length, label_length ], Y_ture
                
