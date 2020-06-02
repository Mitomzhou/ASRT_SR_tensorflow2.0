'''
Created on Mar 25, 2020

@author: monky

model name 251
'''

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

import numpy as np
import random

from feat import read_wav_data, get_spectrogram_feature 
from utils import get_symbollist, get_editdistance
from SpeechDataset import SpeechDataset
from prepare_data import processor


class SpeechModel():
    def __init__(self, datapath):
        """ 初始化
            
            :param datapath:
        """
        self.AUDIO_LENGTH = 1600    # 输入语音长度，不够16s的0补齐
        self.AUDIO_FEATURE_LENGTH = 200 # 输入语音特征长度
        self.SM_OUTPUT_SIZE = 1424  # 输出的拼音的表示大小是1424，即1423个拼音+1个空白块
        self.LABEL_MAX_LENGTH = 64 # label的最大长度
        self.datapath = datapath
        
        # t_model:训练用  base_model:预测用
        self.t_model, self.base_model = self.create_model()
        
        # 当前训练好的模型-测试用
        self.current_model = ''
    
    
    def conv_layer(self, filters):
        return layers.Conv2D(filters, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')  
    
    def conv_layer_no_bias(self, filters):
        return layers.Conv2D(filters, (3,3), use_bias=False, activation='relu', padding='same', kernel_initializer='he_normal')    
    
    def maxpooling_layer(self, pool_size):
        return layers.MaxPooling2D(pool_size=pool_size, strides=None, padding="valid")
    
    def dense_layer(self, units, activation=None):
        return layers.Dense(units, activation=activation, use_bias=True, kernel_initializer='he_normal')
        
    def create_model(self):
        """构建model
            :return t_model: train model
            :return base_model: pred/test
        """
        input_data = layers.Input(shape=(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1))  # (1600, 200, 1)
        
        conv1 = self.conv_layer_no_bias(32)(input_data)
        drop1 = layers.Dropout(0.05)(conv1)
        conv2 = self.conv_layer(32)(drop1)
        pool1 = self.maxpooling_layer(2)(conv2)
        
        drop2 = layers.Dropout(0.05)(pool1)
        conv3 = self.conv_layer(64)(drop2)
        drop3 = layers.Dropout(0.1)(conv3)
        conv4 = self.conv_layer(64)(drop3)
        pool2 = self.maxpooling_layer(2)(conv4)
        
        drop4 = layers.Dropout(0.1)(pool2)
        conv5 = self.conv_layer(128)(drop4)
        drop5 = layers.Dropout(0.15)(conv5)
        conv6 = self.conv_layer(128)(drop5)
        pool3 = self.maxpooling_layer(2)(conv6)
       
        drop6 = layers.Dropout(0.15)(pool3)
        conv7 = self.conv_layer(128)(drop6)
        drop7 = layers.Dropout(0.2)(conv7)
        conv8 = self.conv_layer(128)(drop7)
        pool4 = self.maxpooling_layer(1)(conv8)
        
        drop8 = layers.Dropout(0.2)(pool4)
        conv9 = self.conv_layer(128)(drop8)
        drop9 = layers.Dropout(0.2)(conv9)
        conv10 = self.conv_layer(128)(drop9)
        pool5 = self.maxpooling_layer(1)(conv10)
        
        reshape_layer = layers.Reshape((200, 3200))(pool5)
        drop10 = layers.Dropout(0.3)(reshape_layer)
        dense1 = self.dense_layer(128, activation='relu')(drop10)
        drop11 = layers.Dropout(0.3)(dense1)
        dense2 = self.dense_layer(self.SM_OUTPUT_SIZE)(drop11)
        
        y_pred = layers.Activation('softmax')(dense2)
        
        # 测试用模型
        base_model = models.Model(inputs=input_data, outputs=y_pred, name='base_model')
        #base_model.summary()
        
        # ctc loss
        y_true = layers.Input(shape=[self.LABEL_MAX_LENGTH])
        y_pred = y_pred
        input_length = layers.Input(shape=[1], dtype='int64')
        label_length = layers.Input(shape=[1], dtype='int64')
        
        # func: ctc_lambda_func
        ctc_loss = layers.Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc_loss')([y_true, y_pred, input_length, label_length])
        
        # 训练用模型
        t_model = models.Model(inputs=[input_data, y_true, input_length, label_length], outputs=ctc_loss, name='t_model')
        t_model.summary()
        
        # optimizers
        opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        
        t_model.compile(loss={'ctc_loss': lambda y_true, y_pred: y_pred}, optimizer = opt)
        print('[*Info] Create Model Successful, Compiles Model Successful. ')
        
        return t_model, base_model
    
    
    def ctc_lambda_func(self, args):
        """CTCLoss Function: tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
           reference https://blog.csdn.net/Walter_0000/article/details/104471420
            :params y_true: 数字标记的tensor
            :params y_pred: 每个frame 各个class的概率
            :params input_length: y_pred的每个sample的序列长度
            :params label_length: y_true的序列长度
        """
        y_true, y_pred, input_length, label_length = args
        y_pred = y_pred[:, :, :]
        return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    
    
    def load_model(self, modelfile):
        """加载模型文件 tf.keras.models.load_weights()
            :param modelfile:
        """
        self.t_model.load_weights(modelfile)
        #self.base_model.load_weights(modelfile + '.base')
        
    
    def load_model_h5(self, modelfile):
        """加载模型文件 tf.keras.models.load_model()
           HDF5: conda install h5py
            :param modelfile:
        """
        #self.t_model = models.load_model(modelfile)
        self.base_model = models.load_model(modelfile)
        
    
    def recognize_speech(self, wavfile):
        """识别音频文件，输出拼音结果
            :param wavfile: 音频文件
        """
        # 加载wav
        wave_data, fs = read_wav_data(wavfile)
        # 语谱特征 
        spec_feat = get_spectrogram_feature(wave_data, fs)
        # (x,200)补全到(1600,200)
        input_data = np.zeros((self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH), dtype=np.float) 
        input_data[0:len(spec_feat)] = spec_feat
        
        # 维度变换  (1600,200) -> (1,1600,200,1)
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.expand_dims(input_data, axis=3)
        
        
        y_pred = self.base_model.predict(input_data)

        input_length = np.array([y_pred.shape[1]])   # 3 maxpooling -> 200
        
        d_result = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1)
        d_result = tf.keras.backend.get_value(d_result[0][0])[0]
        
        # 获取拼音列表
        list_symbol_dic = get_symbollist('') 
        
        r_str=[]
        for i in d_result:
            r_str.append(list_symbol_dic[i])
        #print('预测结果：')
        print(r_str)
        return r_str
        
        
    def train_speech(self, epoch=1, step_per_epochs=5, batch_size=2, path='', modelfile=''):
        """训练模型，采用 compile, fit_generator, save_weights
            :param epoch:
            :param step_per_epochs: 一个epoch的终止step数, 一般的是 len(data)//batch_size
        """
        data = SpeechDataset(self.datapath + '/train.csv')
        yielddata = data.generate_data(batch_size)
        for epoch in range(epoch):
            print('epoch: ', epoch)
            self.t_model.fit_generator(yielddata, step_per_epochs)
            #self.save_model(path + 'epoch_' + str(epoch) + '_' + filename)
            self.save_model(path + modelfile)
            self.test_model(modelfile=(path + modelfile), datapath=self.datapath, mode='train', datacount=32)
            self.test_model(modelfile=(path + modelfile), datapath=self.datapath, mode='test', datacount=32)
            
    
    def save_model(self, filename):    
        """保存模型，默认路径 
            :param filename: 模型文件名称
        """
        #self.t_model.save(filename + '.h5')
        #self.base_model.save(filename + '_base.h5')
        #self.current_model = filename + '_base.h5'
        
        self.base_model.save_weights(filename)
        print('save model file successful, ',filename)

    
    def test_model(self, modelfile='model_speech/train/weight.ckpt', datapath='', mode='test',datacount=100):
        """测试模型,计算字错率
            :param modelfile: 测试加载的模型
            :param datapath: csv所在目录
            :param mode: test, 测试模式,加载test.csv
                         dev,  验证模式,加载dev.csv
            :param datacount: 测试wav数量
        """
        # 加载模型
        self.load_model(modelfile)
        
        speech_dataset = SpeechDataset(datapath + '/' + mode + '.csv')
        set_len = len(speech_dataset.data_list)
        
        error_word = 0
        total_word = 0
        
        
        # 随机产生datacount个不重复整数
        rand_list = random.sample(range(0,set_len-1), datacount)
        for num in rand_list:
            wavfile = speech_dataset.data_list[num]
            label = speech_dataset.label_list[num]
            y_pred_index = []
            y_true_index = []
            
            # 写入pinyin的label下标
            for i in label:
                y_true_index.append(speech_dataset.list_symbol.index(i))
            
            # 预测结果
            y_pred = self.recognize_speech(wavfile)
            for i in y_pred:
                y_pred_index.append(speech_dataset.list_symbol.index(i))
                
            # 计算编辑距离
            error_word += get_editdistance(y_true_index, y_pred_index)
            # 统计总字长
            total_word += len(y_true_index)
            
        error_word_rate = error_word / total_word
        print(mode + ': error_word_rate: ', error_word_rate)
        
        return error_word_rate
        
        
#         print(self.current_model)
#         self.load_model('model_speech/train/weight.ckpt')
        

if __name__ == '__main__': 
    
#     root = '/raid/BH/mitom/data/data_thchs30'
#      
#     # 数据预处理，将data-label输出到cvs文件
#     SUBSETS = ["train", "dev", "test"]
#     for SUBSET in SUBSETS:
#         processor(root, SUBSET, True, root)
#     
#     # init model
#     model = SpeechModel(root)
#      
#     # train
#     model.train_speech(epoch=70, step_per_epochs=640, batch_size=16, path='model_speech/train/',modelfile='weight.ckpt')

        
    # wav test
    model = SpeechModel('')
    model.load_model('model_speech/train/weight.ckpt')
    model.recognize_speech("./data/A2_0.wav")
        
        
        
        
        
        
        
        
        
        
        
        
        
