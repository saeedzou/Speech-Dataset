import argparse
import os
import torch
import torchaudio
import librosa
import numpy as np
import json
import uuid
import wave
from pydub import AudioSegment

def parse_args():
    parser = argparse.ArgumentParser(description="ASR transcription tool.")
    parser.add_argument("audio_file", help="Path to the audio file.")
    parser.add_argument(
        "--model", 
        choices=["wav2vec_v3", "wav2vec_fa", "hezar", "vosk", "whisper"], 
        required=True, 
        help="Choose the ASR model to use."
    )
    return parser.parse_args()

# Helper function for resampling audio
def load_and_resample_audio(path, processor):
    speech_array, sampling_rate = torchaudio.load(path)
    speech_array = speech_array.squeeze().numpy()
    return librosa.resample(np.asarray(speech_array), orig_sr=sampling_rate, target_sr=processor.feature_extractor.sampling_rate)

# Wav2Vec2 model transcription
def wav2vec_transcript(audio_file_path, processor, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    speech = load_and_resample_audio(audio_file_path, processor)
    features = processor(speech, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    input_values, attention_mask = features.input_values.to(device), features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]

# Hezar transcription
def hezar_transcript(audio_file_path):
    from hezar.models import Model as HezarModel

    model = HezarModel.load("hezarai/whisper-small-fa").to("cuda:0" if torch.cuda.is_available() else "cpu")
    transcript = model.predict(audio_file_path)
    return transcript[0]['text'].strip()

# Vosk transcription
def vosk_transcript(audio_file_path):
    from vosk import Model as VoskModel, KaldiRecognizer, SetLogLevel
    import pyaudioconvert as pac

    SetLogLevel(0)
    model = VoskModel(model_name="vosk-model-fa-0.5")
    temp_output_file_name = f"{uuid.uuid4()}.{audio_file_path.split('.')[-1]}"
    AudioSegment.from_mp3(audio_file_path).export(temp_output_file_name, format="wav")
    pac.convert_wav_to_16bit_mono(temp_output_file_name, temp_output_file_name)

    wf = wave.open(temp_output_file_name, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    rec.SetPartialWords(True)

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data) if rec.AcceptWaveform(data) else rec.PartialResult()

    os.remove(temp_output_file_name)
    return json.loads(rec.FinalResult())['text']

# Whisper transcription
def whisper_transcript(audio_file_path):
    from speechbrain.inference.ASR import WhisperASR

    model = WhisperASR.from_hparams(source="speechbrain/asr-whisper-large-v2-commonvoice-fa", run_opts={"device": "cuda:0"})
    return model.transcribe_file(audio_file_path)

def main():
    args = parse_args()
    audio_file_path = args.audio_file

    if args.model == "wav2vec_v3":
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        processor = Wav2Vec2Processor.from_pretrained("m3hrdadfi/wav2vec2-large-xlsr-persian-v3")
        model = Wav2Vec2ForCTC.from_pretrained("m3hrdadfi/wav2vec2-large-xlsr-persian-v3").to("cuda:0" if torch.cuda.is_available() else "cpu")
        transcript = wav2vec_transcript(audio_file_path, processor, model)

    elif args.model == "wav2vec_fa":
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        processor = Wav2Vec2Processor.from_pretrained("masoudmzb/wav2vec2-xlsr-multilingual-53-fa")
        model = Wav2Vec2ForCTC.from_pretrained("masoudmzb/wav2vec2-xlsr-multilingual-53-fa").to("cuda:0" if torch.cuda.is_available() else "cpu")
        transcript = wav2vec_transcript(audio_file_path, processor, model)

    elif args.model == "hezar":
        transcript = hezar_transcript(audio_file_path)

    elif args.model == "vosk":
        transcript = vosk_transcript(audio_file_path)

    elif args.model == "whisper":
        transcript = whisper_transcript(audio_file_path)

    print(f"Transcript:\n{transcript}")

if __name__ == "__main__":
    main()