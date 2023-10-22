import os
import torch
from fastapi import FastAPI, Request
from transformers import AutoProcessor, BarkModel
from scipy.io.wavfile import write as write_wav
from playsound import playsound

app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
model = model.to(device)
model = model.to_bettertransformer()


sample_rate = model.generation_config.sample_rate


@app.get("/")
async def root():
    return "ok"


@app.post('/generate')
async def generate(request: Request, gender: str | None = None, lang: str | None = None):
    body = await request.body()
    text_prompt = body.decode('utf-8')

    voice_preset = "v2/en_speaker_9"
    if gender == "male" and lang == "english":
        voice_preset = "v2/en_speaker_6"
    if gender == "female" and lang == "english":
        voice_preset = "v2/en_speaker_9"
    if gender == "male" and lang == "japanese":
        voice_preset = "v2/ja_speaker_2"
    if gender == "female" and lang == "japanese":
        voice_preset = "v2/ja_speaker_8"
    if gender == "male" and lang == "spanish":
        voice_preset = "v2/es_speaker_0"
    if gender == "female" and lang == "spanish":
        voice_preset = "v2/es_speaker_8"

    inputs = processor(text_prompt, voice_preset=voice_preset).to(device)

    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    write_wav("bark_generation.wav", sample_rate, audio_array)

    playsound(os.path.dirname(__file__) + "\\bark_generation.wav")

    return {"status": "ok"}
