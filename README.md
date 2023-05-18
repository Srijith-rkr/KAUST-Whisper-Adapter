# Fine-tune OpenAI Whisper using Residual Adapters in Pytorch

## Introduction 
Train Whisper for your languages and own purpose!

This repo contains the following functionality built on top of the Pytorch implementation of Whisper from OpenAI - https://github.com/openai/whisper

* Add new language to the Whisper model and train with multiple GPUs for language/dialect identification
* Plot your network architecture
* Visualize what parts of the input Whisper focuses on make predictions (perturbation bases neural saliency method) - https://github.com/ajsanjoaquin/mPerturb
* Additional comments to understand the code better.
* Parameter Efficient Learning (PEL) methods to train Whisper (Adapter and Neural Reprogramming)

## How to set up your environment
Use the .yaml file with conda to load your environment. 
Use Whisper.yaml for training and perturb.yaml for neural saliency methods

## How to add new languages. 

update the following parameters in the load model method to add your custom  languages to Whisper
```
model = whisper.load_model("base", custom = True, num_dialects = 3)
```
set custom to True if you want to add your own languages/dialects \
set num_dialects to the number of languages/dialects you would like to add \
And update the dialects list on whisper/tokenizer.py with your new languages. 

The code has been configured to add the new inputs to the tokenizer and enable the model to predict the new dialects. 

## Training script

The training script uses Pytorch lightning to train the model. 
```
python FineTuneWhisper.py
```
You can configure hyperparameters with the following flags

--project_name: wandb project_name \
'--run_name: wandb run_name \
--num_train_epochs': number of epochs\
--weight_decay: weight_decay \
--learning_rate: learning rate
--train_fraction: number of trainig samples to use per class \
--batch_size: batch size \
--num_workers_percentage: percentage of number of workers to use for dataloaders (reduce if memory error) \
--ga: Number of gradient accumulation steps

## Fair warning

You will probably have to implement your own dataloaders, and a lot more. But the scripts might help go through the process quicker

# Reprogrammed whisper
The PEL methods for whisper are implemented by modifying methods the 'whisper' module in 'reprogrammed_whisper' \ 
Hence, please use 
```
import reprogrammed_whisper
```
The two PEL methods implemented are: 
## 1 Adapters [Paper](https://proceedings.mlr.press/v97/houlsby19a.html)
Adapters are small trainable blocks inserted between the layers of a transformer architecture. They down-project the latent dimension from the previous layer and apply a nonlinear activation function, followed by an up-projection. A residual connection surrounds the adapter layer. This setup encourages parameter sharing between the frozen components and localizes all the weight updates to the adapter modules as shown in the following figure. 

![Illustration of the transformer architecture embedded with adapter layers](https://github.com/Srijith-rkr/Train_Whisper/blob/main/Adapter_img.PNG)

## 2 Neural Reprogramming [Paper](https://arxiv.org/pdf/2106.09296.pdf)
Neural reprogramming can be used to repurpose a frozen pre-trained model to out-of-domain prediction tasks by adding trainable parameters to the input of the pre-trained model. The frozen model is followed by a label-mapping strategy to map the source labels to the outof-domain target labels. The trainable input noise aligns the latent distribution of the target domain with being more similar to that of the source domain, using the pre-existing decision boundaries of the frozen model. Neural reprogramming works well when the input size of the target data is comparably smaller than the source data.

In our implementation we add input noise to the input of the model (Log-Mel Spectrogram of a 30s audio clip), and implement a hard label mapping strategy to map the source labels to target domain. /
/
You can access the PEL methods using the load_model() methods in __init__.py as follows
```
model = whisper.load_model("base",adapter=True, adapter_dims = 64)
    """
    Load a Whisper ASR model with reprogramming features

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
    device : Union[str, torch.device]
        the PyTorch device to put the model into
    download_root: str
        path to download the model files; by default, it uses "~/.cache/whisper"
    in_memory: bool
        whether to preload the model weights into host memory
        
    Added parameters :
    custom : bool
        specifies weather you want the custom model with the added languages. (alters the final token_embedding layer accordingly)
        Use num_dialects to specify how many dialects you will be adding and specify the dialects in tokenizer.py
        if specified, returns checkpoint and un-initiliazed model.
        Set custom = False, if you want to use the reprogrammed model.
        
    adapter : bool
      specifies weather you want to add adapters to the model

    adapter :  int = 256
      specifies the adapter dimensions. (Default is 256)
      
      

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """
```

### Reference

If you consider this work may be useful for your research, please consider to cite this paper. Thank you!

```bib
@inproceedings{radhakrishnan2023parameter,
  title={A Parameter-Efficient Learning Approach to Arabic Dialect Identification with Pre-Trained General-Purpose Speech Model},
  author={Srijith Radhakrishnan, Chao-Han Huck Yang, Sumeer Ahmad Khan, Narsis A. Kiani, David Gomez-Cabrero, Jesper N. Tegner},
  booktitle={INTERSPEECH},
  year={2023}
}
```

