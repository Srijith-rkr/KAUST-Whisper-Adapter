import torch
import torch.nn.functional as F
from torch import nn

from PIL import Image
import cv2
import matplotlib.pyplot as plt

import os
from os.path import join, basename, splitext
import numpy as np
from tqdm import tqdm
import whisper


device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

def tv_norm(input, tv_beta):
    '''
    Computes the Total Variation (TV) denoising term
    '''
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
    return row_grad + col_grad

def postprocess(mask):
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))

    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    return 1 - mask

def upsample(image):
    return F.interpolate(image, size=(80, 3000), mode='bilinear', align_corners=False).to(device)

def perturb(image, model, #transforms, out_dir, \
    tv_beta=3, lr=1, max_iter=40, l1_coeff=0.01, tv_coeff=0.02, \
    plot=True,out_dir = 'masks_output'):
    
    # class CustomLoss(nn.Module):
    #     def __init__(self):
    #         super(CustomLoss, self).__init__()

    #     def forward(self, l1_coeff, tv_coeff, mask,tv_beta,masked_prob):
    #         ls = (l1_coeff * torch.mean(torch.abs(1 - mask)) + tv_coeff * tv_norm(mask, tv_beta) + masked_prob)
    #         return ls
    # customLoss = CustomLoss()

    original_img = image   #np.array(Image.open(image).convert('RGB').resize((224, 224)))

    blurred_img = cv2.GaussianBlur(np.float32(original_img.cpu()), (11, 11), 5) # (11,11) is kernal size , 5 is Gaussian kernel standard deviation in X & Y direction.
    # generate mask
    mask = torch.randn((1,1, 10, 375), dtype = torch.float32, requires_grad=True, device=device) # was 28,28 (8x smaller than original)

    img_tensor = (original_img).unsqueeze(0).to(device)
    blurred_tensor = torch.from_numpy(blurred_img).to(device)
    
    tokenizer = whisper.tokenizer.get_tokenizer(True)
    x = torch.tensor([[tokenizer.sot]] ).to(image.device)
    

    optimizer = torch.optim.Adam([mask], lr=lr)
    
    with torch.no_grad():
        logits = model.logits(x, model.encoder(img_tensor))[:, 0] 
    class_idx = (np.argmax(torch.nn.Softmax(dim=-1)(logits.cpu())).to(device))
    prob=   logits[0, class_idx]

    print(f'Predicted class index: {class_idx}. Probability before perturbation:  { torch.nn.Softmax(dim=-1)(logits)[0, class_idx]} Class {tokenizer.tokenizer.convert_ids_to_tokens([class_idx])[0]}')   #{ prob[0, class_idx]}')

    for i in range(max_iter):
        optimizer.zero_grad()
        upsampled_mask = upsample(mask)[0]       
        # perturb the image with mask
        perturbed_input = torch.mul(img_tensor, upsampled_mask) + torch.mul(blurred_tensor, 1-upsampled_mask)
        
        # add some noise to the perturbed image for the model to learn from multiple masks
        noise = (torch.randn(( 1, 80, 3000), device=device))
        perturbed_input = perturbed_input + noise

        logits = model.logits(x, model.encoder(perturbed_input))[:, 0] 
        masked_idx = torch.nn.Softmax(dim=-1)(logits)
        masked_prob= masked_idx[0, class_idx]
        

        loss = (l1_coeff * torch.mean(torch.abs(1 - mask)) + tv_coeff * tv_norm(mask, tv_beta) + masked_prob)  #customLoss(l1_coeff, tv_coeff, mask,tv_beta,masked_prob)   #(l1_coeff * torch.mean(torch.abs(1 - mask)) + tv_coeff * tv_norm(mask, tv_beta) + masked_prob)
        loss.backward()
        optimizer.step()
        print(mask.grad.mean())
        
        mask.data.clamp_(0, 1)
        if i% 20 == 0:
            print(f'Iteration {i}/{max_iter}, Loss: {loss}, mask_mean:{mask.mean()} Probability for target class {masked_prob}, Predicted label {tokenizer.tokenizer.convert_ids_to_tokens([torch.argmax(masked_idx[0, :])])[0]}, prob_pred {masked_idx[0, [torch.argmax(masked_idx[0, :])]]}')
    return upsample(mask) # squeezed mask of shape (n, m)
    
    
    
