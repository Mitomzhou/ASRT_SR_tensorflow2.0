'''
Created on Mar 25, 2020

@author: monky
-----------------
获取语音语谱图特征
-----------------
'''
import wave
import numpy as np
from scipy.fftpack import fft

import matplotlib.pyplot as plt
import matplotlib
import pylab as pl



def read_wav_data(filename):
    '''
    读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
    '''
    wav = wave.open(filename,"rb") # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes() # 获取帧数
    num_channel=wav.getnchannels() # 获取声道数
    framerate=wav.getframerate() # 获取帧速率
    str_data = wav.readframes(num_frame) # 读取全部的帧
    wav.close() # 关闭流
    wave_data = np.fromstring(str_data, dtype = np.short) # 将声音文件数据转换为数组矩阵形式
    wave_data.shape = -1, num_channel # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    wave_data = wave_data.T # 将矩阵转置
    #wave_data = wave_data 
    return wave_data, framerate  


matplotlib.use('Agg')
def plot_spectrogram(spec, file_name):
    '''
    Draw the spectrogram picture
    :param spec: 语谱图特征
    :param file_name: 保存的语谱图图片
    '''
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(file_name)
    

def get_spectrogram_feature(wav, fs):
    '''
    输入一维波形，如8s语音，输出约为 (8000ms/10ms, 200)的数据shape(800, 200)
    200是由窗长决定的，加汉明窗再fft后，得到的数据是对称的，所以 25ms -> 400，取一半得到200
    语谱图原理：https://www.cnblogs.com/BaroC/p/4283380.html
    @return: (T ms/10 ,200)
    '''
    if(16000 != fs):
        raise ValueError('[Error] ASRT currently only supports wav audio files with a sampling rate of 16000 Hz, but this audio is ' + str(fs) + ' Hz. ')
    
    # wav波形 加时间窗以及时移    10ms-160帧
    time_window = 25 #         25ms-400帧  
    window_length = fs / 1000 * time_window # 计算窗长度的公式，目前全部为400固定值
    
    wav_arr = np.array(wav)
    #wav_length = len(wavsignal[0])
    wav_length = wav_arr.shape[1]
    
    range0_end = int(len(wav[0])/fs*1000 - time_window) // 10 # 计算循环终止的位置，也就是最终生成的窗数
    data_input = np.zeros((range0_end, 200), dtype = np.float) # 用于存放最终的频率特征数据
    data_line = np.zeros((1, 400), dtype = np.float)
    
    x = np.linspace(0, 400 - 1, 400, dtype = np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1) ) # 汉明窗
    
    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400
        
        data_line = wav_arr[0, p_start:p_end]
        data_line = data_line * w # 加窗
        data_line = np.abs(fft(data_line)) / wav_length
        
        data_input[i]=data_line[0:200] # 设置为400除以2的值（即200）是取一半数据，因为是对称的
        
    #print(data_input.shape)
    data_input = np.log(data_input + 1)
    return data_input



if __name__ == '__main__':
    wave_data, fs = read_wav_data("./data/A2_33.wav")
    spec_feat = get_spectrogram_feature(wave_data, fs)
    plot_spectrogram(spec_feat.T,'spec_feat.png')
    print(spec_feat.shape)






    
