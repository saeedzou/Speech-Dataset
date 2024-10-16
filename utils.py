import os
import argparse
from pydub import AudioSegment

def convert_mp3_to_mono_wav(mp3_path, target_sample_rate=44100):
    # Load MP3 file
    sound = AudioSegment.from_mp3(mp3_path)
    
    # Convert to mono and set the sample rate
    sound = sound.set_channels(1)
    sound = sound.set_frame_rate(target_sample_rate)
    
    # Generate output path with same filename but .wav extension
    output_path = os.path.splitext(mp3_path)[0] + ".wav"
    
    # Export the final audio file
    sound.export(output_path, format="wav")
    
    return output_path

def convert_directory_mp3_to_wav(root_dir, target_sample_rate=44100):
    # Traverse the directory and its subdirectories
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            # Process only MP3 files
            if file.endswith(".mp3"):
                mp3_path = os.path.join(subdir, file)
                print(f"Processing: {mp3_path}")
                
                # Convert the MP3 to mono WAV with the same name
                output_wav_path = convert_mp3_to_mono_wav(mp3_path, target_sample_rate)
                
                print(f"Converted file saved at: {output_wav_path}")

def process_input(path, target_sample_rate=44100):
    """Determine if the input is a file or directory and process accordingly."""
    if os.path.isfile(path) and path.endswith(".mp3"):
        print(f"Processing file: {path}")
        output_path = convert_mp3_to_mono_wav(path, target_sample_rate)
        print(f"Converted file saved at: {output_path}")
    elif os.path.isdir(path):
        print(f"Processing directory: {path}")
        convert_directory_mp3_to_wav(path, target_sample_rate)
    else:
        print(f"Invalid input: {path}. Please provide a valid MP3 file or directory.")


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Convert MP3 files to mono WAV with a specified sample rate.")

    # Arguments for input path and sample rate
    parser.add_argument("input_path", type=str, help="The MP3 file or directory to convert.")
    parser.add_argument("--sample_rate", type=int, default=44100, help="The target sample rate for WAV files. Default is 44100 Hz.")

    # Parse arguments
    args = parser.parse_args()

    # Process the input (either file or directory)
    process_input(args.input_path, args.sample_rate)