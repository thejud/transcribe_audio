#!/usr/bin/env python3

"""
Audio Transcription Tool using OpenAI API

This script provides intelligent audio transcription capabilities using OpenAI's
transcription models (Whisper-1, GPT-4o-transcribe, GPT-4o-mini-transcribe).

Features:
- Automatic audio chunking based on silence detection
- Multiple output formats (text files, optional JSON with timestamps, stdout)
- Context-aware transcription using prompts for improved accuracy
- Support for multiple OpenAI transcription models
- Batch processing of multiple audio files
- Cost-effective model selection options

The tool is specifically designed for transcribing voicemails and personal audio
recordings with family-specific context for improved name and terminology recognition.

Usage:
    # Basic transcription with default settings (creates .txt file only)
    python transcribe.py audio/file.mp3

    # Include JSON output with timestamps
    python transcribe.py audio/file.mp3 --complex-json

    # Text output to stdout
    python transcribe.py audio/file.mp3 --txt

    # Use different model
    python transcribe.py audio/file.mp3 --mini

    # Custom context prompt
    python transcribe.py audio/file.mp3 --prompt "Names: Jud (not Judge)"

    # Batch processing
    python transcribe.py audio/*.mp3

Environment Variables:
    OPENAI_API_KEY: Required OpenAI API key
    DEFAULT_PROMPT: Default context prompt for improved transcription accuracy

Author: Jud Dagnall (with Claude Code)
Version: 1.0
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
from pydub.silence import split_on_silence


def setup_logging(debug: bool = False, stdout_output: bool = False) -> None:
    """
    Configure logging for the application.

    Args:
        debug: Enable debug-level logging
        stdout_output: If True, redirect logs to stderr to avoid mixing with stdout output
    """
    level = logging.DEBUG if debug else logging.INFO
    # Use stderr when outputting to stdout to avoid mixing with transcription output
    stream = sys.stderr if stdout_output else None
    logging.basicConfig(
        level=level, format="%(asctime)s - %(levelname)s - %(message)s", stream=stream
    )


def load_environment() -> OpenAI:
    """
    Load environment variables from .env file and return OpenAI client.

    Returns:
        OpenAI: Configured OpenAI client instance

    Raises:
        SystemExit: If OPENAI_API_KEY is not found in environment
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def get_cache_dir() -> Path:
    """
    Get or create the cache directory for audio chunks.

    Returns:
        Path: Cache directory path in /tmp/transcribe_cache
    """
    cache_dir = Path("/tmp/transcribe_cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def get_cache_key(audio_path: Path, max_chunk_duration: int = 1200) -> str:
    """
    Generate a cache key based on file path, modification time, and chunking parameters.

    Args:
        audio_path: Path to the audio file
        max_chunk_duration: Maximum chunk duration in seconds

    Returns:
        str: Unique cache key for the file and chunking parameters
    """
    stat = audio_path.stat()
    content = (
        f"{audio_path.absolute()}_{stat.st_mtime}_{stat.st_size}_{max_chunk_duration}"
    )
    return hashlib.md5(content.encode()).hexdigest()


def save_chunks_to_cache(chunks: List[AudioSegment], cache_key: str) -> None:
    """
    Save audio chunks to cache.

    Args:
        chunks: List of audio chunks to cache
        cache_key: Unique cache key for the chunks
    """
    cache_dir = get_cache_dir()
    cache_file = cache_dir / f"{cache_key}.pkl"

    # Convert AudioSegments to serializable format
    chunk_data = []
    for chunk in chunks:
        chunk_data.append(
            {
                "raw_data": chunk.raw_data,
                "frame_rate": chunk.frame_rate,
                "sample_width": chunk.sample_width,
                "channels": chunk.channels,
            }
        )

    with open(cache_file, "wb") as f:
        pickle.dump(chunk_data, f)

    logging.info(f"Cached {len(chunks)} chunks to {cache_file}")


def load_chunks_from_cache(cache_key: str) -> Optional[List[AudioSegment]]:
    """
    Load audio chunks from cache.

    Args:
        cache_key: Unique cache key for the chunks

    Returns:
        List of AudioSegment objects or None if cache miss
    """
    cache_dir = get_cache_dir()
    cache_file = cache_dir / f"{cache_key}.pkl"

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "rb") as f:
            chunk_data = pickle.load(f)

        # Reconstruct AudioSegments from cached data
        chunks = []
        for data in chunk_data:
            chunk = AudioSegment(
                data=data["raw_data"],
                sample_width=data["sample_width"],
                frame_rate=data["frame_rate"],
                channels=data["channels"],
            )
            chunks.append(chunk)

        logging.info(f"Loaded {len(chunks)} chunks from cache ({cache_file})")
        return chunks

    except Exception as e:
        logging.warning(f"Failed to load chunks from cache: {str(e)}")
        # Remove corrupted cache file
        try:
            cache_file.unlink()
        except:
            pass
        return None


def chunk_audio_by_silence(
    audio_path: Path,
    min_silence_len: int = 1000,
    silence_thresh: int = -40,
    max_chunk_size: int = 24,
    max_chunk_duration: int = 1200,
) -> List[AudioSegment]:
    """
    Split audio into chunks based on silence detection to optimize API calls.

    This function intelligently splits audio at natural breaks (silence) while
    ensuring chunks don't exceed OpenAI's file size AND duration limits. Small chunks are
    combined to reduce API call overhead. Results are cached to avoid
    re-processing large audio files.

    Args:
        audio_path: Path to the input audio file
        min_silence_len: Minimum length of silence to split on (milliseconds)
        silence_thresh: Silence threshold in dBFS (lower = more sensitive)
        max_chunk_size: Maximum chunk size in MB
        max_chunk_duration: Maximum chunk duration in seconds (OpenAI limit: 1500s)

    Returns:
        List of AudioSegment objects representing audio chunks

    Raises:
        Exception: If audio file cannot be loaded or processed
    """
    # Check cache first
    cache_key = get_cache_key(audio_path, max_chunk_duration)
    cached_chunks = load_chunks_from_cache(cache_key)

    if cached_chunks is not None:
        logging.info(f"Using cached chunks for {audio_path}")
        return cached_chunks

    logging.info(f"Building chunks for {audio_path} (not in cache)")
    logging.info(f"Loading audio file: {audio_path}")
    audio = AudioSegment.from_file(str(audio_path))

    audio_duration = len(audio) / 1000.0  # Convert to seconds
    logging.info(
        f"Loaded audio: {audio_duration:.1f} seconds, analyzing silence patterns..."
    )
    logging.info(
        f"Silence detection parameters: min_silence={min_silence_len}ms, threshold={silence_thresh}dBFS"
    )
    logging.info(
        "Splitting audio on silence (this may take several minutes for large files)..."
    )

    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=500,  # Keep 500ms of silence at the beginning and end
    )

    logging.info(f"Silence detection completed, found {len(chunks)} initial segments")

    # Combine small chunks to avoid too many API calls and stay under size AND duration limits
    logging.info(
        f"Combining segments into chunks (max {max_chunk_size}MB, {max_chunk_duration}s each)..."
    )
    combined_chunks = []
    current_chunk = AudioSegment.empty()
    max_size_bytes = max_chunk_size * 1024 * 1024  # Convert MB to bytes
    max_duration_ms = max_chunk_duration * 1000  # Convert seconds to milliseconds

    for i, chunk in enumerate(chunks):
        # Estimate size (rough approximation)
        estimated_size = len(chunk.raw_data)
        # Calculate duration in milliseconds
        current_duration = len(current_chunk)
        chunk_duration = len(chunk)

        if len(current_chunk) == 0:
            current_chunk = chunk
        elif (
            len(current_chunk.raw_data) + estimated_size < max_size_bytes
            and current_duration + chunk_duration < max_duration_ms
        ):
            current_chunk += chunk
        else:
            # Current chunk is full (by size or duration), start a new one
            combined_chunks.append(current_chunk)
            current_chunk = chunk

        # Log progress for large numbers of segments
        if len(chunks) > 50 and (i + 1) % 20 == 0:
            logging.info(f"Processing segment {i + 1}/{len(chunks)}...")

    if len(current_chunk) > 0:
        combined_chunks.append(current_chunk)

    # Check for and split any chunks that are still too long
    final_chunks = []
    for i, chunk in enumerate(combined_chunks):
        chunk_duration_seconds = len(chunk) / 1000.0
        if chunk_duration_seconds > max_chunk_duration:
            logging.warning(
                f"Chunk {i+1} is {chunk_duration_seconds:.1f}s (exceeds {max_chunk_duration}s limit)"
            )
            logging.info(f"Force-splitting chunk {i+1} into smaller segments...")

            # Force split long chunk into segments of max_chunk_duration
            sub_chunks_created = 0
            chunk_start = 0
            while chunk_start < len(chunk):
                chunk_end = min(chunk_start + max_duration_ms, len(chunk))
                sub_chunk = chunk[chunk_start:chunk_end]
                final_chunks.append(sub_chunk)
                sub_chunks_created += 1
                chunk_start = chunk_end

            logging.info(f"Split chunk {i+1} into {sub_chunks_created} sub-chunks")
        else:
            final_chunks.append(chunk)

    # Calculate total duration of final chunks for verification
    total_chunk_duration = sum(len(chunk) for chunk in final_chunks) / 1000.0
    max_chunk_dur = (
        max(len(chunk) / 1000.0 for chunk in final_chunks) if final_chunks else 0
    )
    logging.info(
        f"Created {len(final_chunks)} audio chunks (total: {total_chunk_duration:.1f}s, max: {max_chunk_dur:.1f}s)"
    )

    # Cache the chunks for future use
    save_chunks_to_cache(final_chunks, cache_key)

    return final_chunks


def transcribe_chunk(
    chunk: AudioSegment,
    chunk_index: int,
    temp_dir: Path,
    client: OpenAI,
    model: str = "gpt-4o-transcribe",
    prompt: Optional[str] = None,
) -> Optional[dict]:
    """
    Transcribe a single audio chunk using OpenAI transcription API.

    This function handles model-specific API parameters and response formats.
    GPT-4o models use 'json' format while Whisper uses 'verbose_json' format.

    Args:
        chunk: Audio chunk to transcribe
        chunk_index: Index of the chunk (for file naming)
        temp_dir: Temporary directory for chunk files
        client: OpenAI client instance
        model: OpenAI transcription model to use
        prompt: Context or guidance for transcription accuracy

    Returns:
        Dictionary containing transcription results with text and segments,
        or None if transcription fails

    Note:
        Temporary files are automatically cleaned up after transcription.
    """
    chunk_path = temp_dir / f"chunk_{chunk_index}.mp3"

    try:
        # Export chunk to temporary file
        chunk.export(str(chunk_path), format="mp3")

        logging.info(f"Transcribing chunk {chunk_index + 1}")

        with open(chunk_path, "rb") as audio_file:
            # Use appropriate response format based on model
            if model.startswith("gpt-4o"):
                # GPT-4o models support json format
                params = {"model": model, "file": audio_file, "response_format": "json"}
                if prompt:
                    params["prompt"] = prompt
                response = client.audio.transcriptions.create(**params)
            else:
                # Whisper model supports verbose_json format
                params = {
                    "model": model,
                    "file": audio_file,
                    "response_format": "verbose_json",
                }
                if prompt:
                    params["prompt"] = prompt
                response = client.audio.transcriptions.create(**params)

        # Clean up temporary file
        chunk_path.unlink()

        response_dict = response.model_dump()

        # For GPT-4o models that return simple JSON, create verbose-style structure
        # This ensures consistent output format across all models
        if model.startswith("gpt-4o") and "segments" not in response_dict:
            response_dict["segments"] = [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": len(chunk) / 1000.0,  # Convert ms to seconds
                    "text": response_dict.get("text", ""),
                    "avg_logprob": -0.5,  # Default values since not provided
                    "compression_ratio": 1.0,
                    "no_speech_prob": 0.0,
                    "seek": 0,
                    "temperature": 0.0,
                    "tokens": [],
                }
            ]

        return response_dict

    except Exception as e:
        logging.error(f"Error transcribing chunk {chunk_index}: {str(e)}")
        if chunk_path.exists():
            chunk_path.unlink()
        return None


def combine_transcriptions(
    transcriptions: List[dict], chunks: List[AudioSegment]
) -> Tuple[str, dict]:
    """
    Combine multiple transcription results with properly adjusted timestamps.

    This function merges transcriptions from multiple audio chunks, ensuring
    that timestamps are adjusted to reflect the correct position in the
    original audio file.

    Args:
        transcriptions: List of transcription results from individual chunks
        chunks: List of corresponding audio chunks (for timing calculations)

    Returns:
        Tuple containing:
        - str: Combined plain text transcription
        - dict: Combined JSON with properly adjusted timestamps and segments

    Note:
        Failed transcriptions (None values) are skipped with appropriate
        time offset adjustments to maintain accurate timestamps.
    """
    full_text = []
    full_segments = []
    current_time_offset = 0.0

    for i, (transcription, chunk) in enumerate(zip(transcriptions, chunks)):
        if transcription is None:
            logging.warning(f"Skipping chunk {i} due to transcription error")
            current_time_offset += len(chunk) / 1000.0  # Convert ms to seconds
            continue

        chunk_text = transcription.get("text", "").strip()
        if chunk_text:
            full_text.append(chunk_text)

        # Adjust timestamps for segments to reflect position in original audio
        segments = transcription.get("segments", [])
        for segment in segments:
            adjusted_segment = segment.copy()
            adjusted_segment["start"] += current_time_offset
            adjusted_segment["end"] += current_time_offset
            full_segments.append(adjusted_segment)

        current_time_offset += len(chunk) / 1000.0  # Convert ms to seconds

    plain_text = " ".join(full_text)
    json_result = {"text": plain_text, "segments": full_segments}

    return plain_text, json_result


def transcribe_audio_file(
    audio_path: Path,
    client: OpenAI,
    debug: bool = False,
    force: bool = False,
    model: str = "gpt-4o-transcribe",
    output_txt: bool = False,
    output_json: bool = False,
    complex_json: bool = False,
    output_dir: Optional[Path] = None,
    prompt: Optional[str] = None,
) -> None:
    """
    Transcribe a single audio file and save or output results.

    This is the main transcription function that orchestrates the entire process:
    chunking, transcription, combination, and output handling.

    Args:
        audio_path: Path to the audio file to transcribe
        client: OpenAI client instance
        debug: Enable debug-level logging
        force: Force overwrite existing output files
        model: OpenAI transcription model to use
        output_txt: Print text output to stdout instead of writing files
        output_json: Print JSON output to stdout instead of writing files
        complex_json: Write JSON file with timestamps and segments (in addition to txt)
        output_dir: Optional output directory for result files (default: same as input)
        prompt: Context or guidance for transcription accuracy

    Note:
        - Creates temporary directory for audio chunks (automatically cleaned up)
        - Supports both file output and stdout output modes
        - Prevents accidental overwrites unless force=True
        - Handles errors gracefully with appropriate logging
    """
    if not audio_path.exists():
        logging.error(f"Audio file not found: {audio_path}")
        return

    # Log file size and audio length
    file_size = audio_path.stat().st_size
    logging.info(f"Processing audio file: {audio_path}")
    logging.info(f"File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")

    # Load audio to get duration
    try:
        audio = AudioSegment.from_file(str(audio_path))
        duration_seconds = len(audio) / 1000.0
        logging.info(
            f"Audio length: {duration_seconds:.1f} seconds ({duration_seconds/60:.1f} minutes)"
        )
    except Exception as e:
        logging.warning(f"Could not determine audio length: {str(e)}")

    # Create temporary directory for chunks
    temp_dir = Path("temp_chunks")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Split audio into manageable chunks
        chunks = chunk_audio_by_silence(audio_path)

        # Transcribe each chunk
        transcriptions = []
        for i, chunk in enumerate(chunks):
            result = transcribe_chunk(chunk, i, temp_dir, client, model, prompt)
            transcriptions.append(result)

        # Combine results with proper timestamp adjustments
        plain_text, json_result = combine_transcriptions(transcriptions, chunks)

        # Handle output based on command-line flags
        if output_txt:
            print(plain_text)
        elif output_json:
            print(json.dumps(json_result, indent=2, ensure_ascii=False))
        else:
            # Save results to files
            base_name = audio_path.stem
            output_directory = output_dir if output_dir else audio_path.parent
            # Ensure output directory exists
            output_directory.mkdir(parents=True, exist_ok=True)

            txt_path = output_directory / f"{base_name}.txt"
            json_path = output_directory / f"{base_name}.json" if complex_json else None

            # Check if files already exist and force flag is not set
            existing_files = []
            if not force:
                if txt_path.exists():
                    existing_files.append(str(txt_path))
                if json_path and json_path.exists():
                    existing_files.append(str(json_path))
                if existing_files:
                    logging.warning(
                        f"Output files already exist: {', '.join(existing_files)}. "
                        "Use --force to overwrite."
                    )
                    return

            # Write transcription results
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(plain_text)

            if complex_json:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(json_result, f, indent=2, ensure_ascii=False)

            output_files = str(txt_path)
            if complex_json:
                output_files += f", {json_path}"
            logging.info(f"Transcription completed. Output files: {output_files}")

    except Exception as e:
        logging.error(f"Error processing {audio_path}: {str(e)}")
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            for file in temp_dir.glob("*"):
                file.unlink()
            temp_dir.rmdir()


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the transcription tool.

    Returns:
        argparse.Namespace: Parsed command line arguments

    Raises:
        SystemExit: For invalid argument combinations
    """
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using OpenAI transcription API",
        epilog="""
Simple Examples:
  %(prog)s audio/voicemail.mp3                    # Basic transcription with cost-effective model
  %(prog)s audio/*.mp3 --mini                     # Process multiple files (cheapest model)
  %(prog)s audio/file.mp3 --prompt "Names: Jud"   # Custom context for better accuracy
  %(prog)s audio/*.mp3 -O transcripts/            # Save to specific directory

For voice memo pipeline, consider using: python transcribe_pipeline.py audio/*.mp3

Advanced Examples (use --advanced for full options):
  %(prog)s audio/file.mp3 --advanced --model whisper-1 --complex-json
  %(prog)s audio/*.mp3 --advanced --txt           # Stdout output
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional arguments
    parser.add_argument(
        "audio_files", nargs="+", help="Path(s) to audio file(s) to transcribe"
    )

    # Core options (always visible)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite existing output files",
    )

    # Simple model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--mini",
        action="store_const",
        const="gpt-4o-mini-transcribe",
        dest="model",
        help="Use cost-effective model (gpt-4o-mini-transcribe)",
    )
    model_group.add_argument(
        "--4o",
        action="store_const",
        const="gpt-4o-transcribe",
        dest="model",
        help="Use balanced model (gpt-4o-transcribe)",
    )

    # Output directory option
    parser.add_argument(
        "-O",
        "--output-dir",
        type=str,
        help="Output directory for result files (default: same directory as input audio)",
    )

    # Context/prompt options
    parser.add_argument(
        "--prompt",
        type=str,
        help="Provide context or guidance to improve transcription accuracy "
        "(e.g., names, terminology). If not provided, uses DEFAULT_PROMPT from .env file",
    )

    # Advanced mode toggle
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Enable advanced options (model selection, output formats, JSON)",
    )

    # Advanced options (only shown with --advanced)
    if "--advanced" in sys.argv or "-h" in sys.argv or "--help" in sys.argv:
        # Model selection options (advanced)
        model_group.add_argument(
            "--model",
            default="gpt-4o-mini-transcribe",
            choices=["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"],
            help="OpenAI transcription model to use (default: gpt-4o-mini-transcribe)",
        )

        # Output format options (advanced)
        output_group = parser.add_mutually_exclusive_group()
        output_group.add_argument(
            "--txt",
            action="store_true",
            help="Print text output to stdout instead of writing files",
        )
        output_group.add_argument(
            "--json",
            action="store_true",
            help="Print JSON output to stdout instead of writing files (whisper-1 only)",
        )

        # Complex JSON output option (advanced)
        parser.add_argument(
            "--complex-json",
            action="store_true",
            help="Write JSON file with timestamps and segments (in addition to txt file)",
        )

    # Parse arguments
    args = parser.parse_args()

    # Set default model if not specified
    if not hasattr(args, "model") or args.model is None:
        args.model = "gpt-4o-mini-transcribe"

    # Set default values for advanced options if not in advanced mode
    if not args.advanced:
        args.txt = False
        args.json = False
        args.complex_json = False

    # Validate JSON output option with model compatibility (advanced mode only)
    if hasattr(args, "json") and args.json and args.model.startswith("gpt-4o"):
        logging.error(
            f"JSON output with detailed segments is not available for model '{args.model}'."
        )
        logging.error(
            "GPT-4o models only provide simple JSON format without detailed segments."
        )
        logging.error(
            "Use --txt for text output or switch to --model whisper-1 for detailed JSON."
        )
        sys.exit(1)

    return args


def main() -> None:
    """
    Main function to handle command line arguments and process audio files.

    This function sets up argument parsing, validates input combinations,
    loads environment configuration, and processes each audio file.

    Command-line interface supports:
    - Multiple audio file inputs
    - Model selection (whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe)
    - Output format selection (files, stdout text, stdout JSON)
    - Custom context prompts for improved accuracy
    - Debug logging and force overwrite options
    - Automatic chunk caching for faster re-processing of large files

    Environment variables:
    - OPENAI_API_KEY: Required for API access
    - DEFAULT_PROMPT: Default context prompt if none provided

    Raises:
        SystemExit: For invalid argument combinations or missing API key
    """
    args = parse_args()

    # Set up logging and load environment
    setup_logging(args.debug, args.txt or args.json)
    client = load_environment()

    # Use default prompt from .env if no prompt provided
    prompt = args.prompt
    if not prompt:
        prompt = os.getenv("DEFAULT_PROMPT")
        if prompt:
            logging.info(f"Using default prompt from .env: {prompt[:50]}...")

    # Process each audio file
    for audio_file_path in args.audio_files:
        audio_path = Path(audio_file_path)
        # Convert output_dir to Path if provided
        out_dir = Path(args.output_dir) if args.output_dir else None

        transcribe_audio_file(
            audio_path,
            client,
            args.debug,
            args.force,
            args.model,
            args.txt,
            args.json,
            args.complex_json,
            out_dir,
            prompt,
        )


if __name__ == "__main__":
    main()
