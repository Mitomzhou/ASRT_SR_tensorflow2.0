# -*- coding: UTF-8 -*-
""" thchs30 dataset """

import os
import pandas
from absl import logging
from glob import glob

SUBSETS = ["train", "dev", "test"]


def convert_audio_and_split_transcript(dataset_dir, subset, out_csv_file, output_dir):
    """Convert tar.gz to WAV and split the transcript.

  Args:
    dataset_dir  : the directory which holds the input dataset.
    subset       : the name of the specified dataset. e.g. dev.
    out_csv_file : the resulting output csv file.
    output_dir   : Athena working directory.
  """

    logging.info("Processing audio and transcript for {}".format(subset))
    data_dir = os.path.join(dataset_dir, subset + "/")
    #trans_dir = os.path.join(dataset_dir, "transcript/")
    
    # 音频文件绝对路径
    wav_files  = []
    wav_files += glob('%s/*.%s' % (data_dir,'wav'))
    wav_files.sort()
    
    content = []  # csv的内容
    #files_size_dict = {} # 文件长度字典
    file_lables = []
    
    for wav_file in wav_files:
        # 长度存入字典
        #files_size_dict[wav_file] = 0
        
        # 处理 label, 映射到data目录下的lable文件
        labelpathfile = wav_file + '.trn'
        labelpath_split = labelpathfile.split("/")
        label_tail = labelpath_split[-2] + "/" + labelpath_split[-1]
        labelfile = labelpathfile[0:-len(label_tail)] + "data/" + labelpath_split[-1]
        
        label_obj=open(labelfile,'r',encoding='UTF-8') # 打开label并读入
        label_text=label_obj.read()
        label_lines=label_text.split('\n') # label_lines[0] 中文字 label_lines[1] pinyin
        items = label_lines[0].split(" ")
        pinyin_label = label_lines[1]
        chs_label = ""
        for item in items:
            chs_label += item
        file_lables.append((wav_file, chs_label, pinyin_label)) 
            
    for wav_filename, chs_label, pinyin_label in file_lables:  
        #filesize = files_size_dict[wav_filename]
        content.append((wav_filename, chs_label, pinyin_label))
    files = content
    
    #print(files)
    df = pandas.DataFrame(
        data=files, columns=["wav_filename", "CHS_transcript", "Pinyin_transcript"]
    )
    df.to_csv(out_csv_file, index=False, sep="\t")
    logging.info("Successfully generated csv file {}".format(out_csv_file))
        

def processor(dataset_dir, subset, force_process, output_dir):
    """ download and process """
    if subset not in SUBSETS:
        raise ValueError(subset, "is not in THCHS30")
    if force_process:
        logging.info("force process is set to be true")

    subset_csv = os.path.join(output_dir, subset + ".csv")
    #print(subset_csv)
    if not force_process and os.path.exists(subset_csv):
        logging.info("{} already exist".format(subset_csv))
        return subset_csv
    logging.info("Processing the THCHS30 subset {} in {}".format(subset, dataset_dir))
    convert_audio_and_split_transcript(dataset_dir, subset, subset_csv, output_dir)
    logging.info("Finished processing THCHS30 subset {}".format(subset))
    return subset_csv

# if __name__ == "__main__":
#     DATASET_DIR = '/home/monky/ASR/workspace/ASRT_SR_tensorflow2.0/data/data_thchs30'
#     OUTPUT_DIR = '/home/monky/ASR/workspace/ASRT_SR_tensorflow2.0/data/data_thchs30'
#     for SUBSET in SUBSETS:
#         processor(DATASET_DIR, SUBSET, True, OUTPUT_DIR)
