# To generate masks of log mel spectrogram
import whisper
import librosa, librosa.display
import torch
import matplotlib.pyplot as plt
import numpy as np
from mPerturb import perturb
import cv2
from PIL import Image

def log_to_mel_spec(log):
    log = (4*log) - 4
    log = torch.pow(10,log)
    return log


model = whisper.load_model("base")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("tests/jfk.flac")  # converts the clip into a numpuy array after 16k samples per seacond
# audio = torch.Tensor(audio)
audio = whisper.pad_or_trim(audio)            # you pad the array with zeroes if the length is less

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)  # of shape (#channels =  80 ; #samples =3000 )
mel_ = log_to_mel_spec(mel)
librosa.display.specshow(mel.cpu().numpy() , sr=16000, hop_length=160, x_axis='time', y_axis='mel',n_fft = 400)

mask  = perturb.perturb(image = mel, model = model ) # mel is an image of [80,3000]
perturbed_input = torch.mul(mel, (mask)[0,0])
librosa.display.specshow(perturbed_input.cpu().detach().numpy() , sr=16000, hop_length=160, x_axis='time', y_axis='mel',n_fft = 400)
print('nell')

mask = mask.detach().numpy()
mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
mask = 1 - mask

heatmap = cv2.applyColorMap(np.uint8((255*mask[0,0])), cv2.COLORMAP_JET)

mask = cv2.cvtColor( np.float32(mask[0,0]) ,cv2.COLOR_GRAY2RGB)

heatmap = np.float32(heatmap) / 255
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

img  = (mel - np.min(mel.detach().numpy())) / (np.max(mel.detach().numpy()) - np.min(mel.detach().numpy()))
img = cv2.cvtColor( np.float32(img) ,cv2.COLOR_GRAY2RGB)
cam = 1.0 * heatmap + img
cam = cam / np.max(cam)

blurred = cv2.GaussianBlur(np.float32( mel.detach().numpy()), (11, 11), 5)
blurred = cv2.cvtColor( np.float32(blurred) , cv2.COLOR_GRAY2RGB)
perturbed = np.multiply(1 -mask[0], img) + np.multiply(mask, blurred) # THE ORDER OF 1-X AND X CHANGED BECAUSE YOU DID : M = 1 - M


# perturbed_img = Image.fromarray(np.uint8(255 * perturbed))
# perturbed_img.save('perturbed.png')

# heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
# heatmap_img.save( 'heatmap.png')

# mask = np.uint8(255 * mask)
# mask_img = Image.fromarray(mask)
# mask_img.save( 'mask.png')

# cam = Image.fromarray(np.uint8(255 * cam))
# cam.save('cam.png')

# plt.figure()

# plt.subplot(131)
# plt.title('Original')
# plt.imshow(np.uint8(img * 255))
# plt.axis('off')

# plt.subplot(132)
# plt.title('Mask')
# plt.imshow(mask, cmap='gray')
# plt.axis('off')

# plt.subplot(133)
# plt.title('Perturbed Image')
# plt.imshow(np.uint8(255 * perturbed))
# plt.axis('off')

# plt.tight_layout()
# plt.show()


    