import whisper
import torch
model = whisper.load_model("base")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("tests/jfk.flac")  # converts the clip into a numpuy array after 16k samples per seacond
# audio = torch.Tensor(audio)
audio = whisper.pad_or_trim(audio)            # you pad the array with zeroes if the length is less

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)  # of shape (#channels =  80 ; #samples = )

# detect the spoken language
x, probs = model.detect_language(mel)  # x is the language token id with the max probility
print(f"Detected language: {max(probs, key=probs.get)}")  

# decode the audio
options = whisper.DecodingOptions(fp16 = False, without_timestamps = False) #added language= max(probs, key=probs.get)language= max(probs, key=probs.get)
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)

# import whisper

# model = whisper.load_model("base")
# result = model.transcribe("tests/jfk.flac",verbose = True)
# print(result["text"])