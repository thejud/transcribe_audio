#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.silence import split_on_silence


def setup_logging(debug: bool = False, stdout_output: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if debug else logging.INFO
    # Use stderr when outputting to stdout to avoid mixing with transcription output
    stream = sys.stderr if stdout_output else None
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=stream
    )


def load_environment() -> OpenAI:
    """Load environment variables from .env file and return OpenAI client."""
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logging.error("OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def chunk_audio_by_silence(audio_path: Path, min_silence_len: int = 1000, 
                          silence_thresh: int = -40, max_chunk_size: int = 24) -> List[AudioSegment]:
    """
    Split audio into chunks based on silence detection.
    
    Args:
        audio_path: Path to the audio file
        min_silence_len: Minimum length of silence to split on (ms)
        silence_thresh: Silence threshold in dBFS
        max_chunk_size: Maximum chunk size in MB
        
    Returns:
        List of audio chunks
    """
    logging.info(f"Loading audio file: {audio_path}")
    audio = AudioSegment.from_mp3(str(audio_path))
    
    logging.info("Splitting audio on silence...")
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=500  # Keep 500ms of silence at the beginning and end
    )
    
    # Combine small chunks to avoid too many API calls
    combined_chunks = []
    current_chunk = AudioSegment.empty()
    max_size_bytes = max_chunk_size * 1024 * 1024  # Convert MB to bytes
    
    for chunk in chunks:
        # Estimate size (rough approximation)
        estimated_size = len(chunk.raw_data)
        
        if len(current_chunk) == 0:
            current_chunk = chunk
        elif len(current_chunk.raw_data) + estimated_size < max_size_bytes:
            current_chunk += chunk
        else:
            combined_chunks.append(current_chunk)
            current_chunk = chunk
    
    if len(current_chunk) > 0:
        combined_chunks.append(current_chunk)
    
    logging.info(f"Created {len(combined_chunks)} audio chunks")
    return combined_chunks


def transcribe_chunk(chunk: AudioSegment, chunk_index: int, temp_dir: Path, client: OpenAI, model: str = "gpt-4o-transcribe", prompt: Optional[str] = None) -> Optional[dict]:
    """
    Transcribe a single audio chunk using OpenAI transcription API.
    
    Args:
        chunk: Audio chunk to transcribe
        chunk_index: Index of the chunk
        temp_dir: Temporary directory for chunk files
        client: OpenAI client instance
        model: OpenAI transcription model to use
        prompt: Context or guidance for transcription
        
    Returns:
        Transcription result with timestamps
    """
    chunk_path = temp_dir / f"chunk_{chunk_index}.mp3"
    
    try:
        # Export chunk to temporary file
        chunk.export(str(chunk_path), format="mp3")
        
        logging.info(f"Transcribing chunk {chunk_index + 1}")
        
        with open(chunk_path, "rb") as audio_file:
            # Use appropriate response format based on model
            if model.startswith("gpt-4o"):
                # Prepare parameters for GPT-4o models
                params = {
                    "model": model,
                    "file": audio_file,
                    "response_format": "json"
                }
                if prompt:
                    params["prompt"] = prompt
                response = client.audio.transcriptions.create(**params)
            else:
                # Prepare parameters for Whisper model
                params = {
                    "model": model,
                    "file": audio_file,
                    "response_format": "verbose_json"
                }
                if prompt:
                    params["prompt"] = prompt
                response = client.audio.transcriptions.create(**params)
        
        # Clean up temporary file
        chunk_path.unlink()
        
        response_dict = response.model_dump()
        
        # For GPT-4o models that return simple JSON, create verbose-style structure
        if model.startswith("gpt-4o") and "segments" not in response_dict:
            # Create a single segment for the entire chunk
            response_dict["segments"] = [{
                "id": 0,
                "start": 0.0,
                "end": len(chunk) / 1000.0,  # Convert ms to seconds
                "text": response_dict.get("text", ""),
                "avg_logprob": -0.5,  # Default values since not provided
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
                "seek": 0,
                "temperature": 0.0,
                "tokens": []
            }]
        
        return response_dict
        
    except Exception as e:
        logging.error(f"Error transcribing chunk {chunk_index}: {str(e)}")
        if chunk_path.exists():
            chunk_path.unlink()
        return None


def combine_transcriptions(transcriptions: List[dict], chunks: List[AudioSegment]) -> Tuple[str, dict]:
    """
    Combine multiple transcription results with adjusted timestamps.
    
    Args:
        transcriptions: List of transcription results
        chunks: List of audio chunks (for timing calculations)
        
    Returns:
        Tuple of (plain_text, json_with_timestamps)
    """
    full_text = []
    full_segments = []
    current_time_offset = 0.0
    
    for i, (transcription, chunk) in enumerate(zip(transcriptions, chunks)):
        if transcription is None:
            logging.warning(f"Skipping chunk {i} due to transcription error")
            current_time_offset += len(chunk) / 1000.0  # Convert ms to seconds
            continue
            
        chunk_text = transcription.get('text', '').strip()
        if chunk_text:
            full_text.append(chunk_text)
        
        # Adjust timestamps for segments
        segments = transcription.get('segments', [])
        for segment in segments:
            adjusted_segment = segment.copy()
            adjusted_segment['start'] += current_time_offset
            adjusted_segment['end'] += current_time_offset
            full_segments.append(adjusted_segment)
        
        current_time_offset += len(chunk) / 1000.0  # Convert ms to seconds
    
    plain_text = ' '.join(full_text)
    json_result = {
        'text': plain_text,
        'segments': full_segments
    }
    
    return plain_text, json_result


def transcribe_audio_file(audio_path: Path, client: OpenAI, debug: bool = False, force: bool = False, model: str = "gpt-4o-transcribe", output_txt: bool = False, output_json: bool = False, prompt: Optional[str] = None) -> None:
    """
    Transcribe a single audio file and save results.
    
    Args:
        audio_path: Path to the audio file
        client: OpenAI client instance
        debug: Enable debug logging
        force: Force overwrite existing output files
        model: OpenAI transcription model to use
        output_txt: Print text output to stdout instead of writing files
        output_json: Print JSON output to stdout instead of writing files
        prompt: Context or guidance for transcription
    """
    if not audio_path.exists():
        logging.error(f"Audio file not found: {audio_path}")
        return
    
    logging.info(f"Processing audio file: {audio_path}")
    
    # Create temporary directory for chunks
    temp_dir = Path("temp_chunks")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Split audio into chunks
        chunks = chunk_audio_by_silence(audio_path)
        
        # Transcribe each chunk
        transcriptions = []
        for i, chunk in enumerate(chunks):
            result = transcribe_chunk(chunk, i, temp_dir, client, model, prompt)
            transcriptions.append(result)
        
        # Combine results
        plain_text, json_result = combine_transcriptions(transcriptions, chunks)
        
        # Handle output based on flags
        if output_txt:
            print(plain_text)
        elif output_json:
            print(json.dumps(json_result, indent=2, ensure_ascii=False))
        else:
            # Save results to files
            base_name = audio_path.stem
            txt_path = audio_path.parent / f"{base_name}.txt"
            json_path = audio_path.parent / f"{base_name}.json"
            
            # Check if files already exist and force flag is not set
            if not force and (txt_path.exists() or json_path.exists()):
                existing_files = []
                if txt_path.exists():
                    existing_files.append(str(txt_path))
                if json_path.exists():
                    existing_files.append(str(json_path))
                logging.warning(f"Output files already exist: {', '.join(existing_files)}. Use --force to overwrite.")
                return
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(plain_text)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_result, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Transcription completed. Output files: {txt_path}, {json_path}")
        
    except Exception as e:
        logging.error(f"Error processing {audio_path}: {str(e)}")
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            for file in temp_dir.glob("*"):
                file.unlink()
            temp_dir.rmdir()


def main() -> None:
    """Main function to handle command line arguments and process files."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using OpenAI Whisper API"
    )
    parser.add_argument(
        "audio_files",
        nargs="+",
        help="Path(s) to audio file(s) to transcribe"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force overwrite existing output files"
    )
    
    # Model selection options
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model",
        default="gpt-4o-transcribe",
        choices=["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"],
        help="OpenAI transcription model to use (default: gpt-4o-transcribe)"
    )
    model_group.add_argument(
        "--4o",
        action="store_const",
        const="gpt-4o-transcribe",
        dest="model",
        help="Use gpt-4o-transcribe model"
    )
    model_group.add_argument(
        "--mini",
        action="store_const",
        const="gpt-4o-mini-transcribe",
        dest="model",
        help="Use gpt-4o-mini-transcribe model"
    )
    
    # Output format options
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--txt",
        action="store_true",
        help="Print text output to stdout instead of writing files"
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output to stdout instead of writing files (whisper-1 only)"
    )
    
    # Context/prompt options
    parser.add_argument(
        "--prompt",
        type=str,
        help="Provide context or guidance to improve transcription accuracy (e.g., names, terminology). If not provided, uses DEFAULT_PROMPT from .env file"
    )
    
    args = parser.parse_args()
    
    # Validate JSON output option with model compatibility
    if args.json and args.model.startswith("gpt-4o"):
        logging.error(f"JSON output with detailed segments is not available for model '{args.model}'. ")
        logging.error("GPT-4o models only provide simple JSON format without detailed segments.")
        logging.error("Use --txt for text output or switch to --model whisper-1 for detailed JSON.")
        sys.exit(1)
    
    setup_logging(args.debug, args.txt or args.json)
    client = load_environment()
    
    # Use default prompt from .env if no prompt provided
    prompt = args.prompt
    if not prompt:
        prompt = os.getenv('DEFAULT_PROMPT')
        if prompt:
            logging.info(f"Using default prompt from .env: {prompt[:50]}...")
    
    for audio_file_path in args.audio_files:
        audio_path = Path(audio_file_path)
        transcribe_audio_file(audio_path, client, args.debug, args.force, args.model, args.txt, args.json, prompt)


if __name__ == "__main__":
    main()