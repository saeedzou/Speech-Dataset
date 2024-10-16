import argparse
import os
import torch
import torchaudio
import librosa
import numpy as np
import uuid
import subprocess

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
    """Transcribe audio using the Vosk CLI command."""
    output_txt_path = f"{uuid.uuid4()}.txt"  # Temporary output file
    try:
        # Run the Vosk CLI command
        subprocess.run(
            ["vosk", "-l", "fa", "-i", audio_file_path, "-o", output_txt_path],
            check=True,
        )

        # Read the transcription from the output file
        with open(output_txt_path, "r", encoding="utf-8") as f:
            transcript = f.read().strip()

    finally:
        # Clean up the temporary file
        if os.path.exists(output_txt_path):
            os.remove(output_txt_path)

    return transcript

# Whisper transcription
def whisper_transcript(audio_file_path):
    from speechbrain.inference.ASR import WhisperASR

    model = WhisperASR.from_hparams(source="speechbrain/asr-whisper-large-v2-commonvoice-fa", run_opts={"device": "cuda:0"})
    return model.transcribe_file(audio_file_path)

def transcribe_file(audio_file, model_name):
    """Transcribe a single audio file based on the selected model."""
    if model_name == "wav2vec_v3":
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        processor = Wav2Vec2Processor.from_pretrained("m3hrdadfi/wav2vec2-large-xlsr-persian-v3")
        model = Wav2Vec2ForCTC.from_pretrained("m3hrdadfi/wav2vec2-large-xlsr-persian-v3").to("cuda:0" if torch.cuda.is_available() else "cpu")
        return wav2vec_transcript(audio_file, processor, model)

    elif model_name == "wav2vec_fa":
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        processor = Wav2Vec2Processor.from_pretrained("masoudmzb/wav2vec2-xlsr-multilingual-53-fa")
        model = Wav2Vec2ForCTC.from_pretrained("masoudmzb/wav2vec2-xlsr-multilingual-53-fa").to("cuda:0" if torch.cuda.is_available() else "cpu")
        return wav2vec_transcript(audio_file, processor, model)

    elif model_name == "hezar":
        return hezar_transcript(audio_file)

    elif model_name == "vosk":
        return vosk_transcript(audio_file)

    elif model_name == "whisper":
        return whisper_transcript(audio_file)

def transcribe_directory(directory, model_name):
    """Transcribe all WAV files in the directory and its subdirectories."""
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                audio_path = os.path.join(subdir, file)
                print(f"Transcribing: {audio_path}")
                transcript = transcribe_file(audio_path, model_name)
                print(f"Transcript:\n{transcript}\n")


def main():
    args = parse_args()
    input_path = args.input_path

    if os.path.isfile(input_path) and input_path.endswith(".wav"):
        print(f"Transcribing file: {input_path}")
        transcript = transcribe_file(input_path, args.model)
        print(f"Transcript:\n{transcript}")

    elif os.path.isdir(input_path):
        print(f"Transcribing all WAV files in directory: {input_path}")
        transcribe_directory(input_path, args.model)

    else:
        print(f"Invalid input: {input_path}. Please provide a valid WAV file or directory.")

if __name__ == "__main__":
    main()