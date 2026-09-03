#save checkpoint
#================Normal Liba=====================
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

sys.path.append(os.path.abspath(os.path.join("/project/fyp24_dy3/wsngam/GuideForTim", 'Sign2GPT/models/spatial_models/frame_models'))) 
from dino_adaptor_model import Model 

sys.path.append(os.path.abspath(os.path.join("/project/fyp24_dy3/wsngam/GuideForTim", 'Sign2GPT/models/metaformer'))) 
from meta_model import MetaFormer 

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np

from transformers import MBartForConditionalGeneration, MBart50Tokenizer




#================Config=================
class Config:
    BATCHSIZE = 1
    NUMWORKER = 8
    DEVICE = "cuda"
    with open('A2_dict_lem_to_id.json', 'r') as file:
        DICT_LEM_TO_ID = json.load(file) # Load vocabulary mapping
    INDEX_TO_WORD = {v+1: k for k, v in DICT_LEM_TO_ID.items()} # Example: Transform vocabulary index to word
    INDEX_TO_WORD[0] = '<Zero>' #start from1
    LOSS = nn.BCELoss()
    EPOCH = 100
    LR = 5e-5
    VALIDATION_FREQUENCY = 1
    CHECKPOINT_PATH = 'C1B_MBart.pth'
    LOADWEIGHT = False
    TRAIN = True
    WEIGTH_DECAY = 1e-3
    MIX_FEATURE = True
    SKIP_VAL_PRINT = False
    ACCSTEP = 8
    SAVEFREQ = 1

#=============Paths and Data=============
train_new_csv = "/project/fyp24_dy3/wsngam/GuideForTim/C1B_TrainReal.csv" 
test_new_csv = "/project/fyp24_dy3/wsngam/GuideForTim/C1B_TestReal.csv" 
dev_new_csv = "/project/fyp24_dy3/wsngam/GuideForTim/C1B_TestReal.csv" 

frames_root_dir = "/project/fyp24_dy3/utility/dataset/tvb-hksl-news/frames/"

#==========DataLoader=================

class VideoFrameDataset(Dataset): 
    def __init__(self, dataframe, tokenizer, max_length=80):  # Add max_length as a parameter
        self.dataframe = dataframe 
        self.tokenizer = tokenizer
        self.max_length = max_length  # Save it as an instance variable

    def __len__(self): 
        return len(self.dataframe) 

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx] 
        processed_glosses = " ".join(eval(row['processed_glosses']))
        processed_words = " ".join(eval(row['processed_words']))

        # Tokenize with specified max_length
        src_encoding = self.tokenizer(processed_glosses, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        tgt_encoding = self.tokenizer(processed_words, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        
        return src_encoding.input_ids.squeeze(0), tgt_encoding.input_ids.squeeze(0)

def custom_collate_fn(batch): 
    src_inputs, tgt_labels = zip(*batch)
    # Dynamic padding with pad_sequence
    src_padded = torch.nn.utils.rnn.pad_sequence(src_inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_labels, batch_first=True, padding_value=tokenizer.pad_token_id)
    return src_padded, tgt_padded





model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
tokenizer.src_lang = "zh_CN"  # source language


df_train_new = pd.read_csv(train_new_csv)
df_test_new = pd.read_csv(test_new_csv)
df_dev_new = pd.read_csv(dev_new_csv)

train_dataset = VideoFrameDataset(df_train_new, tokenizer)
test_dataset = VideoFrameDataset(df_test_new, tokenizer)
dev_dataset = VideoFrameDataset(df_dev_new, tokenizer)



train_loader = DataLoader(
    train_dataset,
    batch_size=Config.BATCHSIZE,
    num_workers=Config.NUMWORKER,
    pin_memory=True,
    collate_fn=custom_collate_fn
)


test_loader = DataLoader(
    test_dataset,
    batch_size=Config.BATCHSIZE,
    shuffle=False,
    num_workers=Config.NUMWORKER,
    pin_memory=True,
    collate_fn=custom_collate_fn
)


dev_loader = DataLoader(
    dev_dataset,
    batch_size=Config.BATCHSIZE,
    shuffle=False,
    num_workers=Config.NUMWORKER,
    pin_memory=True,
    collate_fn=custom_collate_fn
)



# Initialize DDP model
device = Config.DEVICE
model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)




optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)

# Manage learning rates with warmup and cosine annealing
warmup_epochs = 5
total_epochs = Config.EPOCH
max_lr = Config.LR
warmup_scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs
)
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=(total_epochs - warmup_epochs),
    eta_min=0
)

# Learning rate adjustment
def adjust_learning_rate(epoch):
    if epoch < warmup_epochs:
        warmup_scheduler.step() 
    else:
        cosine_scheduler.step(epoch - warmup_epochs)


# Evaluation function
def evaluate(loader, phase='Validation'):
    model.eval()
    total_loss = 0

    print("-" * 20 + f"Start {phase}" + "-" * 20)
    with torch.no_grad():
        for batch_idx, (src_inputs, tgt_labels) in enumerate(loader):
            src_inputs = src_inputs.to(Config.DEVICE)
            tgt_labels = tgt_labels.to(Config.DEVICE)

            # Perform forward pass to get predictions
            outputs = model(input_ids=src_inputs, labels=tgt_labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Decode predictions and targets into text
            generated_tokens = model.generate(input_ids=src_inputs, num_beams=5, forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"])  # Access the generate method via .module
            preds = [tokenizer.decode(g, skip_special_tokens=True).replace(" ", "") for g in generated_tokens]
            targets = [tokenizer.decode(tgt, skip_special_tokens=True).replace(" ", "") for tgt in tgt_labels] 

    mean_loss = total_loss / len(loader)
    print(f"Average Loss: {mean_loss}")
    return mean_loss
# Save checkpoint function
def save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_path=Config.CHECKPOINT_PATH):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")

# Load checkpoint function
def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #epoch = checkpoint['epoch']
        #best_loss = checkpoint['best_loss']
        #print(f"Checkpoint loaded: epoch {epoch}, best loss {best_loss}")
        #return model, optimizer, epoch, best_loss
    print(f"No checkpoint found at '{checkpoint_path}'")
    return model, optimizer, 0, float('inf')

# Main training loop
best_loss = float('inf')
start_epoch = 0

if Config.LOADWEIGHT:
    model, optimizer, start_epoch, best_loss = load_checkpoint("/project/fyp24_dy3/wsngam/GuideForTim/A1_MBart/70.pth", model, optimizer)

if Config.TRAIN:
    accumulation_steps = Config.ACCSTEP  # Define accumulation steps
    for epoch in range(start_epoch, Config.EPOCH):
        adjust_learning_rate(epoch)
        model.train()
        total_loss = 0
        optimizer.zero_grad()  # Ensure optimizer is reset before the loop

        for batch_idx, (src_inputs, tgt_labels) in enumerate(train_loader):
            print(f'Epoch : {epoch} - Batch : {batch_idx}')

            src_inputs = src_inputs.to(Config.DEVICE)
            tgt_labels = tgt_labels.to(Config.DEVICE)
            tgt_labels[tgt_labels == tokenizer.pad_token_id] = -100  # Ignore padding in target labels

            # Forward pass
            outputs = model(input_ids=src_inputs, labels=tgt_labels)
            loss = outputs.loss
            loss = loss / accumulation_steps  # Scale the loss by accumulation steps
            loss.backward()  # Backward pass

            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()  # Update weights
                optimizer.zero_grad()  # Reset gradients

            total_loss += loss.item() * accumulation_steps

        # Update for remaining gradients if accumulation steps don't align perfectly
        if len(train_loader) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        mean_loss = total_loss / len(train_loader)
        print(f"Mean Loss for Epoch {epoch + 1}: {mean_loss}")

        #evaluate(train_loader, phase='Validation')
        # Validation
        if (epoch + 1) % Config.VALIDATION_FREQUENCY == 0:
            thisLoss = evaluate(dev_loader, phase='Validation')
            print("Validation is Completed")
            if thisLoss < best_loss:
                best_loss = thisLoss
                save_checkpoint(model, optimizer, epoch + 1, best_loss)
                print("Model with best loss is saved!")

        print("=" * 60)

        if (epoch + 1) % Config.SAVEFREQ == 0: 
            save_checkpoint(model, optimizer, epoch + 1, best_loss, checkpoint_path=f"/project/fyp24_dy3/wsngam/GuideForTim/{Config.CHECKPOINT_PATH[0:-4]}/{epoch + 1}.pth")

evaluate(test_loader, phase='Test')


