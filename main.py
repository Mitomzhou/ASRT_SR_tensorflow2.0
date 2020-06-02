'''
Created on Apr 14, 2020

@author: monky
'''
from prepare_data import processor
from SpeechModel import SpeechModel


if __name__ == '__main__':
   
    datapath = '/raid/BH/mitom/data/data_thchs30'
    #datapath = '/raid/BH/mitom/ASRT_SR_tensorflow2.0/data/data_thchs30'
    
    # 数据预处理，将data-label输出到cvs文件
    SUBSETS = ["train", "dev", "test"]
    for SUBSET in SUBSETS:
        processor(datapath, SUBSET, True, datapath)
    print('data csv done!')    
    
    model = SpeechModel(datapath)
    # test
    #model.load_model('model_speech/speech_model251_e_0_step_625000.model')
    #model.recognize_speech("./data/A2_0.wav")
    
    
    # train
    model.train_speech(epoch=5, step_per_epochs=100, batch_size=16, path='model_speech/train/',filename='weight')
    #model.load_model_h5('model_speech/train/weight_base.h5')
    #model.test_model()
    
    
        
