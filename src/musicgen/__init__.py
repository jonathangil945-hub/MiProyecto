import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor , MusicgenForConditionalGeneration

MODEL_ID="facebook/musicgen-small"
DEVICE="cpu"

PROMPT ="Traditional Colombian vallenato, joyful accordion lead, melodic guacharaca, subtle caja drum, passionate male folk vocals, romantic storytelling, warm acoustic tropical rhythm, 110 BPM "

print("Cargando Musicgen ...")

processor=AutoProcessor.from_pretrained(MODEL_ID)

model=MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)

model.eval()

print("Preparando prompt...")

inputs = processor(text= [PROMPT],padding=True,return_tensors="pt").to(DEVICE)

with torch.inference_mode():
    audio = model.generate(**inputs,do_sample=True,guidance_scale=3.0,max_new_tokens=1000)

sample_rate = model.config.audio_encoder.sampling_rate
waveform = audio[0,0].cpu().float().numpy()

write("musica.wav",sample_rate,waveform )

print("Musica generada")


