'''
Created on Jun 1, 2020

@author: monky
'''
from SpeechModel import SpeechModel
from LanguageModel import LanguageModel

# 声学模型
model = SpeechModel('')
model.load_model('model_speech/train/weight.ckpt')
result_pinyin = model.recognize_speech("./data/data_thchs30/data/A2_3.wav")

# 语言模型
lm = LanguageModel('model_language/')
lm.load_model()
result_chs = lm.speech2text(result_pinyin)
print(result_chs)


