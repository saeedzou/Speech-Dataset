import silero_vad


SAMPLING_RATE = 16000
USE_ONNX = False # change this to True if you want to test onnx model

class VAD:
    def __init__(self):
        self.model = silero_vad.load_silero_vad(onnx=USE_ONNX)
        self.sampling_rate = SAMPLING_RATE
        self.threshold = 0.5
        self.min_speech_duration_ms = 250
        self.min_silence_duration_ms = 700

    def get_speech_timestamps(self, wav):
        timestamps = silero_vad.get_speech_timestamps(wav, 
                                       self.model, 
                                       sampling_rate=self.sampling_rate,
                                       threshold=self.threshold,
                                       min_speech_duration_ms=self.min_speech_duration_ms,
                                       min_silence_duration_ms=self.min_silence_duration_ms
                                       )
        return timestamps
    
    def collect_chunks(self, wav, timestamps, audio_path, extension='_only_speech.wav'):
        chunks = silero_vad.collect_chunks(timestamps, wav)
        new_path = f'{audio_path.split(".")[0]}{extension}'
        silero_vad.save_audio(new_path, chunks, self.sampling_rate)
        return chunks, new_path
        

