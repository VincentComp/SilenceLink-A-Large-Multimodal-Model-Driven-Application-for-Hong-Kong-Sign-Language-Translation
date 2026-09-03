#1: Change Checkpoint path
#2: Create directory
#3: Create bash file --> change file path

print("This is C1_2StreamResMLPWithBLCGELU")
#frame_files = frame2_files #### Changed
###With Gelu  + pose only  + Early lateral + (Removed the distributed) + Augmentation+++
#Aug: (translate + rotate + shear)+ temperal crop + speed variation
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
    with open('A2_dict_lem_to_id.json', 'r') as file:
        DICT_LEM_TO_ID = json.load(file) # Load vocabulary mapping
    INDEX_TO_WORD = {v+1: k for k, v in DICT_LEM_TO_ID.items()} # Example: Transform vocabulary index to word
    INDEX_TO_WORD[0] = '<Zero>' #start from1
    LOSS = CTCLoss(blank=0)
    EPOCH = 100
    LR = 5e-5
    VALIDATION_FREQUENCY = 1
    CHECKPOINT_PATH = 'C1_2StreamResMLPWithBLCGELU.pth'
    LOADWEIGHT = False #change here
    TRAIN = True
    WEIGTH_DECAY = 0.01
    MIX_FEATURE = True
    SKIP_VAL_PRINT = False
    ACCSTEP = 8
    SAVEFREQ = 1
    DEVICE = "cuda"

#=============Paths and Data=============
train_new_csv = "/project/fyp24_dy3/wsngam/GuideForTim/A2_trainTextCant.csv"
test_new_csv = "/project/fyp24_dy3/wsngam/GuideForTim/A2_testTextCant.csv" 
dev_new_csv = "/project/fyp24_dy3/wsngam/GuideForTim/A2_devTextCant.csv" 

frames_root_dir = "/project/fyp24_dy3/utility/dataset/tvb-hksl-news/frames/"

#==========DataLoader=================
def convert_to_new_path(original_path):
    # Convert the input to a Path object for easy path manipulation
    original_path = Path(original_path)

    # Define the new base path
    new_base_path = Path('/project/fyp24_dy3/hylambf/Skeleton_upgrade')  


    # Extract parts of the path starting from the date (assuming the date is always at this position)
    remaining_parts = original_path.parts[-3:]

    # Combine the new base path with the remaining parts
    new_path = new_base_path.joinpath(*remaining_parts)

    return new_path

class RandomAffineTransform: #aug1
    def __init__(self, degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear = 10):
        self.affine = transforms.RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear
        )

    def __call__(self, img):
        return self.affine(img)

def speed_variation(frames, speed_factor_range=(0.8, 1.2)): #aug2
    """
    Speed up or slow down the sequence by resampling frames.
    """
    speed_factor = random.uniform(*speed_factor_range)
    original_length = len(frames)
    new_length = max(1, int(original_length / speed_factor))
    indices = np.linspace(0, original_length - 1, num=new_length).astype(int)
    
    if len(indices) > 380:
        indices = indices[0:380] # added to fixed

    return [frames[i] for i in indices]

def temporal_cropping(frames, crop_ratio=0.9): #aug3
    """
    Crop a random temporal segment from the frames.
    """
    if len(frames) <= 1:
        return frames
    crop_length = max(1, int(len(frames) * crop_ratio))
    start = random.randint(0, len(frames) - crop_length)
    return frames[start:start + crop_length]        

def positional_noise(skeleton, noise_level=0.02): #aug4
    """
    Add Gaussian noise to joint coordinates.
    """
    noise = np.random.normal(0, noise_level, skeleton.shape)
    return skeleton + noise


class VideoFrameDataset(Dataset): 
    def __init__(self, dataframe, transform=None, augment=False): 
        self.dataframe = dataframe 
        self.transform = transform 
        self.augment = augment

        if self.augment:
            self.spatial_transform = transforms.Compose([
                RandomAffineTransform(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ])
        else:
            self.spatial_transform = transforms.Compose([])  # No augmentation


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


        #===========================Test=============================
        frame_files = sorted(video_path.glob('*'))
        frame2_files = [convert_to_new_path(file) for file in frame_files]
        frame_files = frame2_files #### Changed

        frames = []
        frames2 = []
        for frame_file, frame2_file in zip(frame_files, frame2_files):
            with open(frame_file, 'rb') as f, open(frame2_file, 'rb') as f2:
                image = Image.open(f)
                image2 = Image.open(f2)
                
                image = image.convert('RGB')
                image2 = image2.convert('RGB')
                
                if self.transform:
                    image = self.transform(image)
                    image2 = self.transform(image2)

                if self.augment:
                    image = self.spatial_transform(image) #aug1
                    image2 = self.spatial_transform(image2)

            frames.append(image)
            frames2.append(image2)

        
        if self.augment:
            frames = speed_variation(frames, speed_factor_range=(0.8, 1.2)) #aug2
            frames = temporal_cropping(frames, crop_ratio=0.9) #aug3
            frames2 = speed_variation(frames2, speed_factor_range=(0.8, 1.2))
            frames2 = temporal_cropping(frames2, crop_ratio=0.9)

        
        frames_tensor = torch.stack(frames) 
        frames_tensor2 = torch.stack(frames2) 

        # if self.augment:
        #     # Apply skeleton-based augmentations
        #     frames_tensor = positional_noise(frames_tensor, noise_level=0.02).float() #aug4
        #     frames_tensor2 = positional_noise(frames_tensor2, noise_level=0.02).float()
        


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

    df_train_new = pd.read_csv(train_new_csv)
    df_test_new = pd.read_csv(test_new_csv)
    df_dev_new = pd.read_csv(dev_new_csv)

    train_dataset = VideoFrameDataset(df_train_new, transform=transformation, augment=True)
    test_dataset = VideoFrameDataset(df_test_new, transform=transformation)
    dev_dataset = VideoFrameDataset(df_dev_new, transform=transformation)


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

    #===========DinoV2===============
    ckpt_path = "/project/fyp24_dy3/utility/modelWeight/dinov2/DinoV2Small.pth"
    vit_model1 = Model(
        ckpt_dir=ckpt_path,
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
        ckpt_dir=ckpt_path,
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


            #print("videos",torch.stack(videos).shape) 

            videos = [video.to(device) for video in videos]
            numFrames_List = [video.size(0) for video in videos]
            y1, _, _ = self.vit_model1(videos, None)
            y1 = pad_sequence(y1)

            frames_tensor2 = [video.clone().to(device) for video in videos]
            y2, _, _ = self.vit_model2(frames_tensor2, None)
            y2 = pad_sequence(y2)


            #print("y1",y1.shape)
            #print("y2",y2.shape)

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


    optimizer = torch.optim.AdamW(lmm_model.parameters(), lr=Config.LR)

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

    def beam_search_decoder(log_probs, beam_width=3, blank_index=0):
        time_steps, batch_size, num_classes = log_probs.shape
                    
        all_decoded_sequences = []

        # Iterate over the batch size
        for batch_idx in range(batch_size):
            beam = [(list(), 0.0)]  # Initialize the beam for each batch element

            # Iterate over each time step
            for t in range(time_steps):
                all_candidates = []
                probs = log_probs[t, batch_idx, :]  # Access once reduced by batch_idx
                
                # Explore hypotheses in the beam
                for seq, score in beam:
                    for char_index in range(num_classes):
                        new_score = score + probs[char_index].item()  # Get the log probability for char index
                        candidate = (seq + [char_index], new_score)
                        all_candidates.append(candidate)

                # Sort all candidates by score and keep top beam_width
                ordered_candidates = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
                beam = ordered_candidates[:beam_width]

            # Extract the best sequence
            best_seq, _ = max(beam, key=lambda tup: tup[1])

            # Remove duplicates and blanks
            best_sequence = []
            prev_char = None
            for char_index in best_seq:
                if char_index != prev_char and char_index != blank_index:
                    best_sequence.append(char_index)
                prev_char = char_index

            all_decoded_sequences.append(best_sequence)

        return all_decoded_sequences

    def convert_indices_to_words(index_sequences, index_to_word):
        word_sequences = []
        for sequence in index_sequences:
            word_sequence = [index_to_word.get(idx, "<UNK>") for idx in sequence]  # Use "<UNK>" if index not in the map
            word_sequences.append(word_sequence)
        return word_sequences

    # Learning rate adjustment
    def adjust_learning_rate(epoch):
        if epoch < warmup_epochs:
            warmup_scheduler.step() 
        else:
            cosine_scheduler.step(epoch - warmup_epochs)


    # Evaluation function
    def evaluate(loader, phase='Validation', epoch = "Testing"):
        #lmm_model.train()
        lmm_model.eval()
        total_loss = 0

        print("-" * 20 + f"Start {phase} : Epoch {epoch}" + "-" * 20)
        with torch.no_grad():
            for batch_idx, (videos, frames_tensor2, labels, label_lengths, num_frames, glosses, words, processed_glosses, processed_words ) in enumerate(loader):
                final_output, numFrames_List = lmm_model(videos, frames_tensor2)
                softmax_scores = final_output
                log_probs = softmax_scores.log_softmax(2).permute(1, 0, 2)
                
                #print(log_probs) #Debug
                #print("labels",labels) #Debug
                #print("pred", torch.argmax(log_probs, dim = -1).squeeze()) #Debug

                # Calculate loss
                input_lengths = torch.tensor(numFrames_List, dtype=torch.int32)
                loss = Config.LOSS(log_probs, labels, input_lengths, label_lengths.clone().detach().to(torch.int32))
                total_loss += loss.item()
                


                #decoded_sequences = beam_search_decoder(log_probs, beam_width=5)
                decoded_sequences = greedy_decoder(log_probs)
                
                word_sequences = convert_indices_to_words(decoded_sequences, Config.INDEX_TO_WORD)
                if not Config.SKIP_VAL_PRINT:
                    print("Target : ",processed_glosses)
                    print("Predicted : ",word_sequences) #!!!
            
        #     ================================print================================

            mean_loss = total_loss / len(train_loader)
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
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            print(f"Checkpoint loaded: epoch {epoch}, best loss {best_loss}")
            return model, optimizer, epoch, best_loss
        print(f"No checkpoint found at '{checkpoint_path}'")
        return model, optimizer, 0, float('inf')

    # Main training loop
    best_loss = float('inf')
    start_epoch = 0

    if Config.LOADWEIGHT:
        lmm_model, optimizer, start_epoch, best_loss = load_checkpoint(Config.CHECKPOINT_PATH, lmm_model, optimizer) #change here

    if Config.TRAIN:
        accumulation_steps = Config.ACCSTEP  # Define the number of steps to accumulate gradients
        for epoch in range(start_epoch, Config.EPOCH):
            adjust_learning_rate(epoch)
            lmm_model.train()
            total_loss = 0
            optimizer.zero_grad()  # Move zero_grad() outside the loop to allow accumulation

            for batch_idx, (videos, frames_tensor2, labels, label_lengths, num_frames, glosses, words, processed_glosses, processed_words) in enumerate(train_loader):
                

                print(f'Epoch : {epoch} - Batch : {batch_idx}')

                final_output, numFrames_List = lmm_model(videos, frames_tensor2)
                
                softmax_scores = final_output
                log_probs = softmax_scores.log_softmax(2).permute(1, 0, 2)
                input_lengths = torch.tensor(numFrames_List, dtype=torch.int32)
                target_lengths = label_lengths.clone().detach().to(torch.int32)
                loss = Config.LOSS(log_probs, labels.to(device).float(), torch.tensor(380), target_lengths)
                #loss = Config.LOSS(log_probs, labels.to(device).float(), input_lengths, target_lengths)

                loss = loss / accumulation_steps  # Scale loss by accumulation steps
                loss.backward()  # Backward pass with scaled loss

                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()  # Update weights
                    optimizer.zero_grad()  # Reset gradients after accumulation

                total_loss += loss.item() * accumulation_steps  # Accumulate loss

            # If the number of batches is not a multiple of `accumulation_steps`, step and zero gradients for remaining
            if len(train_loader) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

            mean_loss = total_loss / len(train_loader)
            print(f"Mean Loss for Epoch {epoch + 1}: {mean_loss}")


            if (epoch + 1) % Config.VALIDATION_FREQUENCY == 0:
                thisLoss = evaluate(dev_loader, phase='Validation', epoch = epoch)
                print("Validation is Completed")
                if thisLoss < best_loss:
                    best_loss = thisLoss
                    save_checkpoint(lmm_model, optimizer, epoch + 1, best_loss)
                    print("Model with best loss is saved!")

            print("=" * 60)

            if (epoch + 1) % Config.SAVEFREQ == 0:
                save_checkpoint(lmm_model, optimizer, epoch + 1, best_loss, checkpoint_path = f"/project/fyp24_dy3/wsngam/GuideForTim/{Config.CHECKPOINT_PATH[0:-4]}/{epoch+1}.pth")

    evaluate(test_loader, phase='Test')



#hidden_state