import os
import sys
import shutil
import argparse
import subprocess
import logging
import time
from pathlib import Path
from typing import List, Literal, Optional
from dataclasses import dataclass
from pydub import AudioSegment

@dataclass
class ProcessingConfig:
    chunk_duration_minutes: float = 10
    output_bitrate: str = "128k"
    log_level: str = "INFO"
    log_file: Optional[str] = "audio_processing.log"

class AudioProcessor:
    def __init__(self, config: ProcessingConfig = ProcessingConfig()):
        self.config = config
        self._setup_logging()
        self._setup_directories()
        self.logger.info("AudioProcessor initialized with config: %s", config)

    def _setup_logging(self):
        """Configure logging with both file and console handlers."""
        self.logger = logging.getLogger('AudioProcessor')
        self.logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Remove existing handlers if any
        self.logger.handlers = []

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler (if log_file is specified)
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        for dir_name in ['spleeter/temp_output', 'demucs/temp_output', 'demucs/output']:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
        self.logger.debug("Directory setup completed")

    def _process_chunk(self, chunk_info: tuple) -> tuple:
        """Process a single chunk of audio using either Spleeter or Demucs."""
        chunk, chunk_path, idx, input_file_name, mode = chunk_info
        start_time = time.time()
        
        try:
            self.logger.debug(f"Processing chunk {idx}: Duration={len(chunk)}ms, Mode={mode}")
            
            # Export chunk to temporary file
            chunk.export(chunk_path, format='wav')
            self.logger.debug(f"Chunk {idx} exported to temporary file: {chunk_path}")
            
            if mode == 'spleeter':
                processed = self._spleeter_process_chunk(chunk_path, f'{input_file_name}_temp_chunk_{idx}')
            else:
                _ = self._demucs_process_chunk(chunk_path, f'{input_file_name}_temp_chunk_{idx}')
                spleeter_input_path = f'demucs/temp_output/htdemucs_ft/{input_file_name}_temp_chunk_{idx}/vocals.wav'
                processed = self._spleeter_process_chunk(spleeter_input_path, 'vocals')

                
            
            processing_time = time.time() - start_time
            self.logger.info(f"Chunk {idx} processed successfully in {processing_time:.2f} seconds")
            
            return idx, processed
            
        except Exception as e:
            self.logger.error(f"Error processing chunk {idx}: {str(e)}", exc_info=True)
            raise
        finally:
            # Clean up temporary chunk file
            Path(chunk_path).unlink(missing_ok=True)
            self.logger.debug(f"Temporary file for chunk {idx} cleaned up")

    def _spleeter_process_chunk(self, input_path: str, basename: str) -> AudioSegment:
        """Process a chunk using Spleeter."""
        output_dir = 'spleeter/temp_output'
        cmd = [
            'python', '-m', 'spleeter', 'separate',
            input_path, '-o', output_dir, '-c', 'wav',
            '-b', self.config.output_bitrate
        ]
        
        self.logger.debug(f"Running Spleeter command: {' '.join(cmd)}")
        return self._run_separation_command(cmd, f'{output_dir}/{basename}/vocals.wav')

    def _demucs_process_chunk(self, input_path: str, basename: str, model='htdemucs_ft') -> AudioSegment:
        """Process a chunk using Demucs."""
        output_dir = 'demucs/temp_output'
        cmd = [
            'python', '-m', 'demucs.separate',
            '-n', model, '-o', output_dir,
            '--two-stems', 'vocals',
            '--other-method', 'none',
            input_path
        ]
        
        self.logger.debug(f"Running Demucs command: {' '.join(cmd)}")
        return self._run_separation_command(cmd, f'{output_dir}/{model}/{basename}/vocals.wav')


    def _run_separation_command(self, cmd: List[str], output_path: str, max_retries: int = 3) -> AudioSegment:
        """Execute separation command with retries and return processed audio."""
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Attempt {attempt + 1}/{max_retries}")
                
                # Create process with pipe for real-time output
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )

                # Handle output in real-time
                while True:
                    # Read stdout
                    stdout_line = process.stdout.readline()
                    if stdout_line:
                        stdout_line = stdout_line.rstrip()
                        print(stdout_line, flush=True)  # Print to console
                        self.logger.debug(f"Command stdout: {stdout_line}")  # Log to file

                    # Read stderr
                    stderr_line = process.stderr.readline()
                    if stderr_line:
                        stderr_line = stderr_line.rstrip()
                        print(stderr_line, file=sys.stderr, flush=True)  # Print to console
                        self.logger.debug(f"Command stderr: {stderr_line}")  # Log to file

                    # Check if process has finished
                    if process.poll() is not None:
                        break

                # Get return code
                return_code = process.wait()
                
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, cmd)

                # Check if output file exists
                if not os.path.exists(output_path):
                    raise FileNotFoundError(f"Output file not found: {output_path}")

                return AudioSegment.from_file(output_path).set_channels(1)
                
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All {max_retries} attempts failed", exc_info=True)
                    raise RuntimeError(f"Failed to process chunk after {max_retries} attempts") from e
                continue


    def remove_background_music(
        self,
        input_file_path: str,
        mode: Literal['spleeter', 'demucs'] = 'demucs'
    ) -> str:
        """Remove background music from audio file using specified method."""
        start_time = time.time()
        self.logger.info(f"Starting background music removal: {input_file_path}")
        
        assert mode in ['spleeter', 'demucs'], "Mode must be either 'spleeter' or 'demucs'"
        
        input_path = Path(input_file_path)
        output_path = input_path.with_stem(f"{input_path.stem}_vocals")  # Save with _vocals extension in the same directory
        
        # Check if output already exists
        if output_path.exists():
            self.logger.info(f"Skipping file {input_path.stem}. Vocals file exists.")
            return str(output_path)

        # Load and prepare audio
        self.logger.info("Loading input audio file")
        audio = AudioSegment.from_file(input_file_path)
        chunk_duration = int(self.config.chunk_duration_minutes * 60 * 1000)
        
        # Prepare chunks for processing
        chunks_info = [
            (
                audio[i:min(i + chunk_duration, len(audio))],
                f'{mode}/{input_path.stem}_temp_chunk_{idx}.wav',
                idx,
                input_path.stem,
                mode
            )
            for idx, i in enumerate(range(0, len(audio), chunk_duration))
        ]
        
        total_chunks = len(chunks_info)
        self.logger.info(f"Audio split into {total_chunks} chunks")

        try:
            # Process chunks sequentially
            processed_chunks = []
            for chunk_info in chunks_info:
                idx, processed_chunk = self._process_chunk(chunk_info)
                processed_chunks.append(processed_chunk)
                self.logger.info(f"Chunk {idx + 1}/{total_chunks} completed")

            # Concatenate and export
            self.logger.info("Concatenating processed chunks")
            final_audio = sum(processed_chunks[1:], processed_chunks[0])
            
            self.logger.info(f"Exporting final audio to {output_path}")
            final_audio.set_channels(1)
            final_audio.export(output_path, format='wav')
            
            total_time = time.time() - start_time
            self.logger.info(f"Processing completed in {total_time:.2f} seconds")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error("Processing failed", exc_info=True)
            raise
        finally:
            # Clean up temporary files
            self._cleanup_temp_files(mode)

    def _cleanup_temp_files(self, mode: str):
        """Clean up all temporary files and directories."""
        self.logger.debug("Cleaning up temporary files")
        temp_output = Path(f'{mode}/temp_output')
        
        if temp_output.exists():
            shutil.rmtree(temp_output)
        
        temp_files = Path(mode).glob('*.wav')
        for temp_file in temp_files:
            temp_file.unlink(missing_ok=True)
        
        self.logger.debug("Temporary files cleaned up")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Audio background music removal tool using Spleeter or Demucs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'input',
        type=str,
        help='Input audio file path or directory. If directory, processes all .wav files'
    )

    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['spleeter', 'demucs'],
        default='demucs',
        help='Processing mode: spleeter or demucs'
    )

    parser.add_argument(
        '--chunk-duration',
        type=float,
        default=5,
        help='Duration of each chunk in minutes'
    )

    parser.add_argument(
        '--bitrate',
        type=str,
        default='128k',
        help='Output audio bitrate'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level'
    )

    parser.add_argument(
        '--log-file',
        type=str,
        default='audio_processing.log',
        help='Log file path (disable with "none")'
    )

    return parser.parse_args()

def process_files(processor: AudioProcessor, input_path: str, mode: str) -> List[str]:
    """Process single file or directory of files."""
    input_path = Path(input_path)
    processed_files = []

    if input_path.is_file():
        if input_path.suffix.lower() != '.wav':
            processor.logger.warning(f"Skipping non-WAV file: {input_path}")
            return processed_files
        
        output_path = processor.remove_background_music(
            str(input_path),
            mode
        )
        processed_files.append(output_path)
    
    elif input_path.is_dir():
        wav_files = list(input_path.rglob('*.wav'))
        processor.logger.info(f"Found {len(wav_files)} WAV files in directory")
        
        for wav_file in wav_files:
            try:
                output_path = processor.remove_background_music(
                    str(wav_file),
                    mode
                )
                processed_files.append(output_path)
            except Exception as e:
                processor.logger.error(f"Failed to process {wav_file}: {str(e)}")
                continue
    
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    return processed_files

def main():
    """Main entry point with command line argument handling."""
    args = parse_args()

    # Configure processor
    config = ProcessingConfig(
        chunk_duration_minutes=args.chunk_duration,
        output_bitrate=args.bitrate,
        log_level=args.log_level,
        log_file=None if args.log_file.lower() == 'none' else args.log_file
    )

    try:
        processor = AudioProcessor(config)
        start_time = time.time()

        # Process files
        processed_files = process_files(
            processor,
            args.input,
            args.mode
        )

        # Print summary
        total_time = time.time() - start_time
        print("\nProcessing Summary:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Files processed: {len(processed_files)}")
        print("\nProcessed files:")
        for file_path in processed_files:
            print(f"- {file_path}")

    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()