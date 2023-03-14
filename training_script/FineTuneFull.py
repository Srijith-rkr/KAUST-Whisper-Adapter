
# Mounting Drive and imports
"""
Printing GPU infromation
"""

# gpu_info = !nvidia-smi
# gpu_info = '\n'.join(gpu_info)
# if gpu_info.find('failed') >= 0:
#   print('Not connected to a GPU')
# else:
#   print(gpu_info)

"""Installing Libraries """

# #Custom OpenAI model implementation and dependency
#!git clone https://github.com/Srijith-rkr/Whisper_low_resource_arabic_adaptation.git  
# uncomment above line to initially clone the repo

#The Whisper conda environment already contains requrirements of Whisper implementation from OpenAI
import Whisper_low_resource_arabic_adaptation.whisper as whisper

#prefer using the environment file for the imports. or else you would need to use import requirements from https://github.com/openai/whisper and pip install the below pakages
# !pip install jiwer
# !pip install pytorch-LightningModule #  works with version 1.6.5
# !pip install evaluate

"""Imports"""

import IPython.display
from pathlib import Path

import os
import numpy as np
import pandas as pd
import wandb
import argparse
from tqdm.notebook import tqdm
import evaluate

try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass
#pytorch
import torch
from torch import nn
import torchaudio
import torchaudio.transforms as at
#lightining
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
#from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
#transformers
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

"""# Configurations"""

parser = argparse.ArgumentParser()
parser.add_argument('--project_name',help = 'vanilla_whisper or reprogramming',default = 'FullModel')
parser.add_argument('--run_name',help = 'alloted run name under project',required=True)
parser.add_argument('--num_train_epochs',help = 'Number of train epochs',default=50,type = int)
parser.add_argument('--weight_decay',help = 'of AdamW, default is 0.01',default=0.1,type = float)
parser.add_argument('--learning_rate',help = 'calculated on fly as ',default=1e-4,type=float)
parser.add_argument('--dev_fraction',help = 'an array of two elementes conntaining st,et of audio clips',default=[20,30000],nargs="+",type=int)
parser.add_argument('--train_fraction',help = 'an array of two elementes conntaining st,et of audio clips',default=[500],nargs="+",type=int)
parser.add_argument('--num_worker',help = 'set on the fly to int(os.cpu_count()*0.75)',type = int)
parser.add_argument('--batch_size',type=int,help = 'V100 with 32GB ram can handel a batch size of 128',default = 16)
parser.add_argument('--num_workers_percentage',type=float,help = 'Will have to reduce when using 8 gpus',default = 0.25)
parser.add_argument('--ga',type=int,help = 'gradient accumulation',default = 1)

args = parser.parse_args()

class Config:
    learning_rate = args.learning_rate
    weight_decay =args.weight_decay
    adam_epsilon = 1e-8
    warmup_steps = 0
    batch_size =  args.batch_size #* torch.cuda.device_count()
    num_worker = int(os.cpu_count()*args.num_workers_percentage)
    num_train_epochs = args.num_train_epochs
    gradient_accumulation_steps = args.ga
    train_fraction = args.train_fraction
    dev_fraction = args.dev_fraction#given in seaconds
    project_name = args.project_name
    run_name = args.run_name
    

"""# Dataloader

To refer torch dataset :https://pytorch.org/tutorials/beginner/basics/data_tutorial.html 

To refer torch dataloader : https://pytorch.org/docs/stable/data.html
"""

# The dataset class, must contain implementation for init, len and get_item method
# We read the metatrain.csv/metatest.csv in the datafiles directory 

class ArabicDialectDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, sample_rate = 16000, split='train',fraction=[500]) -> None:
        super().__init__()
        
        """ Below part enclosed in '-> <-' is no longer relavent as we downlaoded the full train set 
        -> OLD dfThe metatrain.csv and meta[test|dev].csv have different folder structures since train data was downloaded from 
        youtube-dlp as '.mp3' whereas test and dev data where downloaded from arabicspeech.org as '.wav'.
        The csv columns also differ <-
        
        fraction: for the train split: specifies how many datapoints per class. eg: [500] :means 500 files per dialect
                   for the test/dev split: specifies the minimum and maximmum  length of each audio clip. eg: [0,20] :means all audio clips
                   between 0 and 20s will be included"""


        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.split = split

        df = pd.read_csv('datafiles/meta'+split+'.csv')   
        
        #Filtering the dataframe according to the split and fraction parameters
        if (self.split == 'test') or  (self.split == 'dev'):
            df = df[df['len'] >= fraction[0]]
            df = df[df['len'] <= fraction[1]]
            
        elif self.split == 'train':
            to_concat_dfs = []
            classes = df['class'].unique()
            
            for cls in classes:
                tempdf = df[df['class']==cls].copy()
                tempdf.sort_values("len", ascending = False,inplace = True, na_position ='last')
                entries_above_20s = tempdf[tempdf['len']>20].copy()
                while (tempdf.shape[0] < fraction[0]):
                    tempdf = pd.concat([tempdf,entries_above_20s])
                to_concat  = tempdf.head(fraction[0])
                to_concat_dfs.append(to_concat)
            df = pd.concat(to_concat_dfs)
            
        else:
            print("Split is not one among 'train', 'test' or 'dev'")

        self.audio_info_df = df
        

    def __len__(self):
        return len(self.audio_info_df)
    
    def __getitem__(self, id):
        
        _,_,_,_,_,_,dialect,_,path = tuple(self.audio_info_df.iloc[id, :]) # train

        
        if (self.split == 'test') or  (self.split == 'dev'):
            path =  path.split('/',1)
            path= os.path.join(path[0],self.split,path[1])
            
        dialect = '<|'+dialect+'|>'

        # loading audio
        audio = whisper.load_audio(path)
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio)

        dec_input_ids = [self.tokenizer.sot] #+ self.tokenizer.encode(dialect)   #Decoder input does not have EOS  
        labels = self.tokenizer.encode(dialect) #+ [self.tokenizer.eot]          # labels does not have SOT

        return {
            "input_ids": mel,
            "dec_input_ids":dec_input_ids,
            "labels": labels,
            "class": dialect}
        
    def output_df(self):
        return self.audio_info_df

class WhisperDataCollator:
    def __call__(self, features):

      input_ids = [feature["input_ids"] for feature in features]
      input_ids = torch.concat([input_id[None, :] for input_id in input_ids])

      labels = [feature["labels"] for feature in features]
      labels = torch.Tensor(labels).to(torch.int64)                             # to(torch.int64)  same as .long()

      dialects = [feature["class"] for feature in features]
      #dialects =torch.Tensor(dialects) 

      dec_input_ids = [feature["dec_input_ids"] for feature in features]
      dec_input_ids = torch.Tensor(dec_input_ids).long()

      batch = {"input_ids": input_ids,'dec_input_ids':dec_input_ids, "labels": labels,"class":dialects}

      return batch

"""# DataLoader sanity checks"""

# wtokenizer = whisper.tokenizer.get_tokenizer(True)
# dataset = ArabicDialectDataset(wtokenizer,split='train',fraction=[30])
# loader = torch.utils.data.DataLoader(dataset, batch_size=4,collate_fn=WhisperDataCollator(),shuffle = True)

# # checking shapes 
# test = (next(iter(loader)))
# test.keys()
# print('input_ids shape:',test['input_ids'].shape)
# print('dec_input_ids shape:',test['dec_input_ids'].shape)
# print('labels shape :',test['labels'].shape)
# print("\nSeeing actual data\n")
# print('dec_input_ids:',test['dec_input_ids'])
# print('labels:',test['labels'])
# print('class:',test['class'])

# # decoding tokenizer
# for i in test['dec_input_ids']:
#           print(wtokenizer.tokenizer.convert_ids_to_tokens(i))

# for i in test['labels']:
#           print(wtokenizer.tokenizer.convert_ids_to_tokens(i))

#passing through the model 

# """ The model predicts NoSpeech or LanguageTag after SOT. The following code masks
# (sets logits to -inf) for non language tokens. Hence only language is predicted
# """

# woptions = whisper.DecodingOptions( without_timestamps=True)
# wmodel = whisper.load_model("base")


# for i in range(1):
#   test = (next(iter(loader)))
#   with torch.no_grad():
#     audio_features = wmodel.encoder(test['input_ids'].cuda())
#     x = torch.tensor([[wtokenizer.sot]] * 4).to(audio_features.device) # 1 for batch size
#     logits = wmodel.logits(x, audio_features)[:, 0]
#     mask = torch.ones(logits.shape[-1]+17, dtype=torch.bool)
#     mask[list(wtokenizer.all_language_tokens)] = False # a mask with true for all logits except language tokens
#     logits[:, mask[:logits.shape[-1]]] = -np.inf # - infinity for all logits except the languages

#     language_tokens = logits.argmax(dim=-1)
#     print(wtokenizer.tokenizer.convert_ids_to_tokens(language_tokens))

# wtokenizer.tokenizer.convert_ids_to_tokens(50272)

"""# Custom model"""

""" Modifying the decoder.token_embedding.weight key in state_dict as we added new dialect tokens """
def load_custom_model():
    
    model ,cpt= whisper.load_model("base",custom = True, num_dialects = 17)
    temp_states = torch.rand( (model.decoder.token_embedding.num_embeddings, cpt['dims']['n_text_state']) ,dtype = torch.float16, device = model.device)
    temp_states[ :cpt["model_state_dict"]['decoder.token_embedding.weight'].shape[0]   ,:] = cpt["model_state_dict"]['decoder.token_embedding.weight']
    cpt["model_state_dict"]['decoder.token_embedding.weight'] = temp_states
    model.load_state_dict(cpt["model_state_dict"]) 
    return model

#checking if the tensors are copied correctly  
# model = load_custom_model()
# print('Shape of nomal model ',wmodel.decoder.token_embedding.weight.shape)
# print('Shape of custom model ',model.decoder.token_embedding.weight.shape)

# print(wmodel.decoder.token_embedding.weight[51864][5])
# print(model.decoder.token_embedding.weight[51864][5])


if __name__=="__main__":
    """# Trainer"""
    
    cfg = Config()
    config_for_wandb={
    "learning_rate"  : cfg.learning_rate,
    "epochs"         : cfg.num_train_epochs,
    "weight_decay"   : cfg.weight_decay,
    "batch_size"     : cfg.batch_size,
    "train_fraction" : cfg.train_fraction,
    "dev_fraction"   : cfg.dev_fraction,
}
    
    # wandb.login() # you would need to create an account and login for the first time. Or use tensorboard logger or native pytorch lightining logger
    # wandb_logger = WandbLogger( project=cfg.project_name, name=cfg.run_name,config=config_for_wandb) # , config=config_for_wandb# can add config parameter also

    class MyWhisperModule(LightningModule):
        
        counter = 0
        
        def __init__(self,cfg:Config, model_name="base") -> None:# added batch_size and lr to use auto_scale_batch_size and auto tune lr
            super().__init__()
            
            self.options = whisper.DecodingOptions( without_timestamps=True)
            self.cfg = cfg
            self.batch_size = cfg.batch_size
            self.learning_rate = cfg.learning_rate
            self.model = load_custom_model()
            self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
            
            config_for_wandb={
                "learning_rate": self.cfg.learning_rate,
                "epochs"       : self.cfg.num_train_epochs,
                "weight_decay" : self.cfg.weight_decay,
                "batch_size"  : self.cfg.batch_size,
                "train_fraction" : self.cfg.train_fraction,
                "dev_fraction"   : self.cfg.dev_fraction,
        }

            wandb.login() # you would need to create an account and login for the first time. Or use tensorboard logger or native pytorch lightining logger
            wandb.init(project=self.cfg.project_name,name=(self.cfg.run_name+" gpu:"+str(MyWhisperModule.counter) )  ,config=config_for_wandb, group=self.cfg.run_name)
            MyWhisperModule.counter = MyWhisperModule.counter + 1
            wandb.log({"configuration_me": config_for_wandb})

            # only decoder training 
            # for p in self.model.encoder.parameters():
            #     p.requires_grad = False
                
            for n, p in self.model.named_parameters():
                if True:
                    p.requires_grad = True
                else: p.requires_grad = False
            
            self.loss_fn = nn.CrossEntropyLoss()

            
        def train_dataloader(self):
            dataset = ArabicDialectDataset(self.tokenizer,split='train',fraction =self.cfg.train_fraction)
            return torch.utils.data.DataLoader(dataset, 
                            batch_size=self.batch_size, 
                            drop_last=True, shuffle=True,
                            collate_fn=WhisperDataCollator(),
                            num_workers = self.cfg.num_worker
                            )

        def val_dataloader(self):
            dataset = ArabicDialectDataset(self.tokenizer,split='dev',fraction =self.cfg.dev_fraction)
            return torch.utils.data.DataLoader(dataset, 
                            batch_size=self.batch_size, 
                            drop_last=True, shuffle=False,
                            collate_fn=WhisperDataCollator(),
                            num_workers = self.cfg.num_worker
                            )
        
        def forward(self, x): # do not have to implement this - will use model directly for inference
            return self.model(x)

        def training_step(self, batch, batch_id):
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            dec_input_ids = batch["dec_input_ids"]

            #with torch.no_grad():
            audio_features = self.model.encoder(input_ids)

            out = self.model.decoder(dec_input_ids, audio_features)
            loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
            
            #if self.global_rank == 1:
            wandb.log({"train/loss": loss})
                #self.log("train/loss", loss, on_step=True,on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.batch_size)

            u=0; m=0
            target = labels.view(-1)
            pred = out.argmax(dim=-1)
            for i,j in zip(target,pred):
                
                if i==j:
                    
                    m = m+1
                else :
                    u = u +1 


            return {
                "loss": loss,
                'u':u,
                'm':m
            }
        
        def training_epoch_end(self, training_step_outputs):
    
            u=0; m=0
            for d in training_step_outputs:
                u = d['u'] + u
                m = d['m'] + m
            acc = m / (u+m)
            
            #wandb.log({"train/epoch/acc": acc})
            #if self.global_rank == 0:
            wandb.log({"train/epoch/acc":acc})
            wandb.log({"train/epochVsSteps":self.current_epoch})

        
        def validation_step(self, batch, batch_id):
            
            input_ids = batch["input_ids"]
            labels = batch["labels"].long()
            dec_input_ids = batch["dec_input_ids"].long()


            audio_features = self.model.encoder(input_ids)
            out = self.model.decoder(dec_input_ids, audio_features)

            loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

            
            u=0; m=0
            target = labels.view(-1)
            pred = out.argmax(dim=-1)
            for i,j in zip(target,pred):
                
                if i==j:
                    m = m+1
                else :
                    u = u +1 

            #if self.global_rank == 0:
            wandb.log({"val/loss": loss})

            return {
                "loss": loss,
                'u':u,
                'm':m
            }

        def validation_epoch_end(self, validation_step_outputs):
            u=0; m=0 ; 
            for d in validation_step_outputs:
                u = d['u'] + u
                m = d['m'] + m
                
            acc = m / (u+m)
            
            #if self.global_rank == 0:
            wandb.log({"val/epoch/acc": acc})# prog_bar=True, logger=True, batch_size=self.cfg.batch_size)


        def configure_optimizers(self):

            model = self.model
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() 
                                if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.cfg.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() 
                                if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, 
                            lr=self.learning_rate, 
                            eps=self.cfg.adam_epsilon)
            self.optimizer = optimizer

            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self.cfg.warmup_steps, 
                num_training_steps=self.t_total
            )
            self.scheduler = scheduler

            return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]


        def setup(self, stage=None):


            if stage == 'fit' or stage is None:
                
                dataset = ArabicDialectDataset(self.tokenizer,split='train')
                self.t_total = (
                ( dataset.__len__() // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )

    pwd = os.getcwd()
    check_output_dir = os.path.join(pwd,'logs',cfg.project_name+' '+cfg.run_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{check_output_dir}/checkpoint",
        filename="checkpoint-{epoch:04d}",
       # monitor="val/epoch/acc",
       # mode="max",
        save_top_k=-1
    )

    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]

    model = MyWhisperModule(cfg, model_name = 'base')

    SEED = 108
    seed_everything(SEED, workers=True)

    trainer = Trainer(
        precision=16,
        #accelerator='gpu', devices=1, # for one gpu
        gpus=-1,strategy='ddp',accelerator='gpu',# for multiple gpus
        
    # auto_scale_batch_size= True,  #{'scale_batch_size': 128} for V100 gpu with 32gb ram 
    #auto_lr_find=True,
    
        max_epochs=cfg.num_train_epochs,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
      #  logger=wandb_logger,
        callbacks=callback_list
    )
    # lr_finder = trainer.tune(model) 
    # lr_finder['lr_find'].results

    # # Plot with
    # fig = lr_finder['lr_find'].plot(suggest=True)
    # fig.show()

    # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder['lr_find'].suggestion()
    # print(new_lr)

    trainer.fit(model)
    wandb.finish()

