#save checkpoint
#================Normal Liba=====================
import subprocess


from rouge_chinese import Rouge
import os 
import torch 
import pandas as pd 
from pathlib import Path 
from PIL import Image 
import json 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms 
import torch.nn.functional as F 
from torch.nn import CTCLoss 
import sys 
import torch.nn as nn 
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Sign2GPT/models/spatial_models/frame_models'))) 
from dino_adaptor_model import Model 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Sign2GPT/models/metaformer'))) 
from meta_model import MetaFormer 

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np

from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import time

#================Config=================
class Config:
    DEVICE = "cuda:6"
    with open('VocabDict.json', 'r') as file:
        DICT_LEM_TO_ID = json.load(file) # Load vocabulary mapping
    INDEX_TO_WORD = {v+1: k for k, v in DICT_LEM_TO_ID.items()} # Example: Transform vocabulary index to word
    INDEX_TO_WORD[0] = '<Zero>' #start from1

# Load checkpoint function
def load_checkpoint(checkpoint_path, model):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, 0, float('inf')

# Initialize DDP model

model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
tokenizer.src_lang = "zh_CN"  # source language

device = Config.DEVICE
model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

model, start_epoch, best_loss = load_checkpoint("LLM.pth", model)


sentnece = ""

subprocess.run("mkdir LLMReady", capture_output=True, text=True, shell=True)
model.eval()
while True:
    print("waiting")
    with torch.no_grad():
        if os.path.exists("LLMExit"):
            exit(-1)

        if os.path.exists("LLMGo"):
            with open("VisualOutput.txt", 'r') as file:
                sentence = file.readline()
            tokens = tokenizer(sentence, return_tensors='pt', max_length=380, padding='max_length', truncation=True)['input_ids']
            src_inputs = tokens.to(Config.DEVICE)    
            outputs = model(input_ids=src_inputs, labels=None)
            generated_tokens = model.generate(input_ids=src_inputs, num_beams=5, forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"])  # Access the generate method via .module
            preds = [tokenizer.decode(g, skip_special_tokens=True).replace(" ", "") for g in generated_tokens]
            

            with open("LLMOutput.txt", "w") as file:
                if sentence == "Error":
                    file.write("Error: 無法翻譯,請再嘗試")
                else:
                    file.write(preds[0])
            print(preds[0])
            time.sleep(1)

            subprocess.run("rm -r LLMGo", capture_output=True, text=True, shell=True)