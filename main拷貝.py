
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from  transformer import evaluate,subword_encoder_zh,training_model,set_Hyperparams
import logging
import numpy as np
logging.basicConfig(level=logging.ERROR)

import argparse

np.set_printoptions(suppress=True)
# 要被翻譯的英文句子

num_layers_M = 4 
d_model_M = 128
dff_M = 512
num_heads_M = 8


def get_parser():
    parser = argparse.ArgumentParser(description='my description')
    parser.add_argument('--set', type=str , nargs="+")
    parser.add_argument('--train',type=int )
    parser.add_argument('--zh',type=str,nargs='+')

    return parser

def change_M(num_layers,d_model,dff,nums_heads):
    global num_layers_M
    global d_model_M
    global dff_M
    global num_heads_M
    num_layers_M = num_layers
    d_model_M = d_model
    dff_M = dff
    num_heads_M = nums_heads




EPOCHS = 30
sentence1 = "China, India, and others have enjoyed continuing economic growth."
def train(e = 30):
    set_Hyperparams(num_layers_M,d_model_M,dff_M,num_heads_M)
    global EPOCHS 
    EPOCHS= e
    training_model(e)
def translate(sentence):
    set_Hyperparams(num_layers_M,d_model_M,dff_M,num_heads_M)
    training_model(EPOCHS = EPOCHS)
    
    # 取得預測的中文索引序列
    predicted_seq, _ = evaluate(sentence)

    # 過濾掉 <start> & <end> tokens 並用中文的 subword tokenizer 幫我們將索引序列還原回中文句子
    target_vocab_size = subword_encoder_zh.vocab_size
    predicted_seq_without_bos_eos = [idx for idx in predicted_seq if idx < target_vocab_size]
    predicted_sentence = subword_encoder_zh.decode(predicted_seq_without_bos_eos)


    print(f"使用參數：{num_layers_M}layers_{d_model_M}d_{num_heads_M}heads_{dff_M}dff_{20}train_perc")
    print("翻譯結果：",predicted_sentence)

# change_M(4,128,512,1)
# train()
# translate(sentence1)

if __name__ == '__main__':
    
    
    parser = get_parser()
    args = parser.parse_args()

    if args.set:
        change_M(args.set[0],args.set[1],args.set[2],args.set[3])
    if args.train:
        train(args.train)
    if args.zh:
        sentence = ' '.join(args.zh)
        translate(sentence)
   
    




