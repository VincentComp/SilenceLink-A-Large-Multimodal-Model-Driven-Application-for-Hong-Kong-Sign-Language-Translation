
#================Normal Liba====================
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
import random
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import subprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Sign2GPT/models/spatial_models/frame_models'))) 
from dino_adaptor_model import Model 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Sign2GPT/models/metaformer'))) 
from meta_model import MetaFormer 


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np

from torch.nn import TransformerEncoder, TransformerEncoderLayer



#================Config=================
class Config:
    BATCHSIZE = 1
    NUMWORKER = 4
    with open('VocabDict.json', 'r') as file:
        DICT_LEM_TO_ID = json.load(file) # Load vocabulary mapping
    INDEX_TO_WORD = {v+1: k for k, v in DICT_LEM_TO_ID.items()} # Example: Transform vocabulary index to word
    INDEX_TO_WORD[0] = '<Zero>' #start from1
    LOSS = CTCLoss(blank=0)
    CHECKPOINT_PATH = 'VisualModel.pth'
    DEVICE ="cuda:5"

#=============Paths and Data=============
test_new_csv = "ReadImage.csv" 


#==========DataLoader=================
class VideoFrameDataset(Dataset): 
    def __init__(self, dataframe, transform=None): 
        self.dataframe = dataframe 
        self.transform = transform 


    def __len__(self): 
        return len(self.dataframe) 

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx] 
        video_path = Path(row['video_directory']) 
        label_sequence = row['label']
        label_length = int(row['label_length']) 
        num_frames = int(row['num_frames']) 
        glosses = row['glosses'] 
        words = row['words'] 
        processed_glosses = " ".join(eval(row['processed_glosses']))
        processed_words = " ".join(eval(row['processed_words']))

        
        imagePath = ["ImageBuffer/" + path for path in sorted(os.listdir(video_path))]
        frame_files = imagePath
        frame2_files = frame_files

        frames = []
        frames2 = []

        
        for frame_file, frame2_file in zip(frame_files,frame2_files):
            with open(frame_file, 'rb') as f, open(frame2_file, 'rb') as f2:
                image = Image.open(f)
                image2 = Image.open(f2)
                
                image = image.convert('RGB')
                image2 = image2.convert('RGB')
                
                if self.transform:
                    image = self.transform(image)
                    image2 = self.transform(image2)

            frames.append(image)
            frames2.append(image2)

        frames_tensor = torch.stack(frames) 
        frames_tensor2 = torch.stack(frames2) 
        
        label_tensor = torch.tensor(list(map(int, label_sequence.split())), dtype=torch.long)  
        label_tensor += 1 #shift the dictionary by 1 to provide space for empty tokens
        #label_tensor::start from 1

        return frames_tensor,frames_tensor2, label_tensor, label_length, num_frames, glosses, words, processed_glosses, processed_words 


 
        
def custom_collate_fn(batch): 
    frames,frames_tensor2, labels, label_lengths, num_frames, glosses, words, processed_glosses, processed_words  = zip(*batch) 
    labels_stacked = [label.clone().detach().long() for label in labels] 
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels_stacked, batch_first=True, padding_value=0) 
    return list(frames), list(frames_tensor2), labels_padded, torch.tensor(label_lengths), torch.tensor(num_frames), list(glosses), list(words), processed_glosses, processed_words 

    

if __name__ == "__main__":
    # Ensure init_distributed is called only once here
    device = Config.DEVICE

    # Dataloader with transformation and distributed sampler
    transformation = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    df_test_new = pd.read_csv(test_new_csv)
    test_dataset = VideoFrameDataset(df_test_new, transform=transformation)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCHSIZE,
        shuffle=False,
        num_workers=Config.NUMWORKER,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )


    #===========DinoV2===============
    vit_model1 = Model(
        ckpt_dir=None,
        trainable_names=[], 
        adaptor_layers=[7,8,9,10,11],
        adapt_params={
            "w_lora": True,
            "w_lora_ff": True,
            "lora_rank": 4,
            "lora_drop": 0.1,
            "lora_a": 4.0,
            "rng_init": False,
        },
        out_dim=512, 
    )

    vit_model2 = Model(
        ckpt_dir=None,
        trainable_names=[], 
        adaptor_layers=[7,8,9,10,11],
        adapt_params={
            "w_lora": True,
            "w_lora_ff": True,
            "lora_rank": 4,
            "lora_drop": 0.1,
            "lora_a": 4.0,
            "rng_init": False,
        },
        out_dim=512, 
    )

    # ResMLP Block with Spatial and Channel MLPs
    class LateralResMLPBlock(nn.Module):
        def __init__(self, dim, seq_length):
            super(LateralResMLPBlock, self).__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.spatial_mlp = nn.Sequential(
                nn.Linear(seq_length, seq_length),
                nn.GELU(),
                nn.Linear(seq_length, seq_length)
            )
            
            self.norm2 = nn.LayerNorm(dim)
            self.channel_mlp = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            )

            self.lateral_proj = nn.Linear(dim, dim)  # For lateral communication

        def forward(self, x, lateral_input=None):
            # Incorporate lateral input
            if lateral_input is not None:
                x = x + self.lateral_proj(lateral_input)


            # Spatial MLP
            residual = x
            x = self.norm1(x)
            x = x.transpose(1, 2)
            x = self.spatial_mlp(x)
            x = x.transpose(1, 2)
            x = x + residual

            # Channel MLP
            residual = x
            x = self.norm2(x)
            x = self.channel_mlp(x)
            x = x + residual

            return x

    # ResMLP with Lateral Communication Capabilities
    class LateralResMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, seq_length, num_layers=4):
            super(LateralResMLP, self).__init__()
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            self.res_mlp_layers = nn.ModuleList([LateralResMLPBlock(hidden_dim, seq_length) for _ in range(num_layers)])
            self.output_projection = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x, lateral_inputs=None):
            x = self.input_projection(x)
            if lateral_inputs is None:
                lateral_inputs = [None] * len(self.res_mlp_layers)
            
            for i, layer in enumerate(self.res_mlp_layers):
                x = layer(x, lateral_inputs[i])
            x = self.output_projection(x)
            return x

    def pad_sequence(sequence, max_length=380, embedding_dim=512):
        if sequence.size(1) < max_length:
            padding = torch.zeros((sequence.size(0), max_length - sequence.size(1), embedding_dim), device=sequence.device)
            sequence = torch.cat((sequence, padding), dim=1)
        return sequence



    # LMM Class with ResMLP
    class LMM(nn.Module):
        def __init__(self, vit_model1, resmlp1, vit_model2, resmlp2, dim = 512):
            super(LMM, self).__init__()
            
            self.vit_model1 = vit_model1
            self.resmlp1 = resmlp1

            self.vit_model2 = vit_model2
            self.resmlp2 = resmlp2

            self.output_projection = nn.Linear(dim, 6487)


        def forward(self, videos, frames_tensor2):

            videos = [video.to(device) for video in videos]
            numFrames_List = [video.size(0) for video in videos]
            y1, _, _ = self.vit_model1(videos, None)
            y1 = pad_sequence(y1)

            frames_tensor2 = [video.clone().to(device) for video in videos]
            y2, _, _ = self.vit_model2(frames_tensor2, None)
            y2 = pad_sequence(y2)


            # Calculate lateral inputs for each layer
            lateral_inputs1 = [layer(y2) for layer in self.resmlp2.res_mlp_layers]
            lateral_inputs2 = [layer(y1) for layer in self.resmlp1.res_mlp_layers]

            # Process through ResMLP with lateral communication and project to output size
            y1_transformed = self.resmlp1(y1, lateral_inputs=lateral_inputs1)
            y2_transformed = self.resmlp2(y2, lateral_inputs=lateral_inputs2)

            final_output = self.output_projection(y1_transformed + y2_transformed)
            return final_output, numFrames_List

    resmlp_model1 = LateralResMLP(512, 512, 380, 4)
    resmlp_model2 = LateralResMLP(512, 512, 380, 4)
    # Initialize DDP model
    lmm_model = LMM(vit_model1, resmlp_model1, vit_model2, resmlp_model2).to(device)


  
    def greedy_decoder(log_probs, blank_index=0):
        max_indices = torch.argmax(log_probs, dim=-1)
        sequences = []
        for batch in max_indices.permute(1, 0):
            prev_idx = None
            sequence = []
            for idx in batch:
                if idx != prev_idx and idx != blank_index:
                    sequence.append(idx.item())
                prev_idx = idx
            sequences.append(sequence)
        return sequences

    
    def convert_indices_to_words(index_sequences, index_to_word):
        word_sequences = []
        for sequence in index_sequences:
            word_sequence = [index_to_word.get(idx, "<UNK>") for idx in sequence]  # Use "<UNK>" if index not in the map
            word_sequences.append(word_sequence)
        return word_sequences


    # Evaluation function
    def evaluate(loader, phase='Validation', epoch = "Testing"):
        lmm_model.eval()
        with torch.no_grad():
            for batch_idx, (videos, frames_tensor2, labels, label_lengths, num_frames, glosses, words, processed_glosses, processed_words ) in enumerate(loader):
                final_output, numFrames_List = lmm_model(videos, frames_tensor2)
                softmax_scores = final_output
                log_probs = softmax_scores.log_softmax(2).permute(1, 0, 2)
                decoded_sequences = greedy_decoder(log_probs)
                word_sequences = convert_indices_to_words(decoded_sequences, Config.INDEX_TO_WORD)
                print(word_sequences)
                # subprocess.run("mkdir 1", capture_output=True, text=True, shell=True)
                with open("VisualOutput.txt", "w") as file:
                    if word_sequences and word_sequences[0]: 
                        file.write(eval(str(word_sequences))[0][0])
                    else:
                        file.write("Error")  # if no words

    # Load checkpoint function
    def load_checkpoint(checkpoint_path, model):
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            return model, epoch, best_loss
        return model, 0, float('inf')

    lmm_model, start_epoch, best_loss = load_checkpoint(Config.CHECKPOINT_PATH, lmm_model) #change here
    # subprocess.run("rm -r VisualGo", capture_output=True, text=True, shell=True)
                         
    subprocess.run("mkdir VisualModelReady", capture_output=True, text=True, shell=True)
    while True:
        print("waiting")
        if os.path.exists("VisualExit"):
            exit(-1)

        if os.path.exists("VisualGo"):
            evaluate(test_loader, phase='Test')
            subprocess.run("rm -r VisualGo; mkdir LLMGo", capture_output=True, text=True, shell=True)
            # os.makedirs("/csproject/fyp24_dy3/wsngam/Deploy/LLMGo", exist_ok=True)
            

