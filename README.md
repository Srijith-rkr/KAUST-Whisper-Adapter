# Train_Whisper

# Train_Whisper

## Introduction 
Train Whisper for your languages!!!

This repo contains the following functionality built on top of the Pytorch implementation of Whisper from OpenAI - https://github.com/openai/whisper

* Add new language to the Whisper model and train with multiple GPUs for language/dialect identification
* Plot your network architecture
* Visualize what parts of the input Whisper focuses on make predictions (perturbation bases neural saliency method) - https://github.com/ajsanjoaquin/mPerturb
* Additional comments to understand the code better.

## How to set up your environment
Use the .yaml file with conda to load your environment. 
Use Whisper.yaml for training and perturb.yaml for neural saliency methods

## How to add new languages. 

update the following parameters in the load model method to add your custom  languages to Whisper

model = whisper.load_model("base", custom = True, num_dialects = 3)

set custom to True if you want to add your own languages/dialects
set num_dialects to the number of languages/dialects you would like to add
And update the dialects list on whisper/tokenizer.py with your new languages. 

The code has been configured to add the new inputs to the tokenizer and enable the model to predict the new dialects. 

## Training script

The training script uses Pytorch lightning to train the model. 
```
python FineTuneWhisper.py
```
You can configure hyperparameters with the following flags

--project_name: wandb project_name
'--run_name: wandb run_name
--num_train_epochs': number of epochs
--weight_decay: weight_decay
--learning_rate: learning rate
--train_fraction: number of trainig samples to use per class
--batch_size: batch size
--num_workers_percentage: percentage of number of workers to use for dataloaders (reduce if memory error)
--ga: Number of gradient accumulation steps
