#!/usr/bin/env python3

"""
Streamlined Audio Processing Pipeline

A simple, one-command tool for processing voice memos and audio files from
raw audio to final cleaned text output. Designed for ease of use with
sensible defaults while hiding intermediate files and complex options.

This pipeline:
1. Transcribes audio using cost-effective OpenAI models
2. Post-processes transcripts using voice memo summarization
3. Outputs final cleaned text files matching input filenames
4. Organizes outputs by date and cleans up intermediate files

Usage:
    # Process single voice memo
    python pipeline.py voicemail.mp3

    # Process multiple files
    python pipeline.py *.mp3 *.m4a

    # Specify output directory
    python pipeline.py audio/*.mp3 -O processed/

    # Use higher quality models
    python pipeline.py memo.mp3 --quality high

    # Custom context for better transcription
    python pipeline.py meeting.mp3 --prompt "Meeting about project timeline"

Author: Jud Dagnall (with Claude Code)
Version: 1.0
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def setup_logging(debug: bool = False) -> None:
    """
    Configure logging for the pipeline.

    Args:
        debug: Enable debug-level logging
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )


def get_default_output_dir() -> Path:
    """
    Get default output directory organized by date.

    Returns:
        Path: Default output directory (processed/YYYY-MM-DD/)
    """
    today = datetime.now().strftime("%Y-%m-%d")
    return Path("processed") / today


def run_transcription(
    audio_file: Path,
    output_dir: Path,
    model: str = "gpt-4o-mini-transcribe",
    prompt: Optional[str] = None,
    force: bool = False,
) -> Optional[Path]:
    """
    Run transcription on an audio file.

    Args:
        audio_file: Path to audio file
        output_dir: Directory for output
        model: OpenAI transcription model
        prompt: Custom prompt for better accuracy
        force: Force overwrite existing files

    Returns:
        Path to transcribed text file, or None if failed
    """
    cmd = [
        "python",
        "transcribe.py",
        str(audio_file),
        "--model",
        model,
        "--output-dir",
        str(output_dir),
    ]

    if prompt:
        cmd.extend(["--prompt", prompt])
    if force:
        cmd.append("--force")

    try:
        logging.info(f"Transcribing {audio_file.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            # Find the generated text file
            txt_file = output_dir / f"{audio_file.stem}.txt"
            if txt_file.exists():
                logging.debug(f"Transcription completed: {txt_file}")
                return txt_file
            else:
                logging.error(f"Expected transcription file not found: {txt_file}")
                return None
        else:
            logging.error(
                f"Transcription failed for {audio_file.name}: {result.stderr}"
            )
            return None

    except subprocess.TimeoutExpired:
        logging.error(f"Transcription timed out for {audio_file.name}")
        return None
    except Exception as e:
        logging.error(f"Error transcribing {audio_file.name}: {e}")
        return None


def run_post_processing(
    transcript_file: Path,
    output_dir: Path,
    model: str = "gpt-4.1-nano",
    mode: str = "memo",
    force: bool = False,
) -> Optional[Path]:
    """
    Run post-processing on a transcript file.

    Args:
        transcript_file: Path to transcript file
        output_dir: Directory for final output
        model: OpenAI chat model for processing
        mode: Processing mode ('memo' or 'transcript')
        force: Force overwrite existing files

    Returns:
        Path to processed text file, or None if failed
    """
    # Generate output filename
    output_file = output_dir / f"{transcript_file.stem}.txt"

    cmd = [
        "python",
        "post_process.py",
        str(transcript_file),
        "--model",
        model,
        "-o",
        str(output_file),
    ]

    if mode == "memo":
        cmd.append("--memo")
    if force:
        # Note: post_process.py doesn't have --force, but -o will overwrite
        pass

    try:
        logging.info(f"Post-processing {transcript_file.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            if output_file.exists():
                logging.debug(f"Post-processing completed: {output_file}")
                return output_file
            else:
                logging.error(f"Expected processed file not found: {output_file}")
                return None
        else:
            logging.error(
                f"Post-processing failed for {transcript_file.name}: {result.stderr}"
            )
            return None

    except subprocess.TimeoutExpired:
        logging.error(f"Post-processing timed out for {transcript_file.name}")
        return None
    except Exception as e:
        logging.error(f"Error post-processing {transcript_file.name}: {e}")
        return None


def process_audio_file(
    audio_file: Path,
    output_dir: Path,
    temp_dir: Path,
    transcription_model: str = "gpt-4o-mini-transcribe",
    processing_model: str = "gpt-4.1-nano",
    mode: str = "memo",
    prompt: Optional[str] = None,
    force: bool = False,
    keep_intermediate: bool = False,
) -> bool:
    """
    Process a single audio file through the complete pipeline.

    Args:
        audio_file: Path to audio file
        output_dir: Final output directory
        temp_dir: Temporary directory for intermediate files
        transcription_model: Model for transcription
        processing_model: Model for post-processing
        mode: Processing mode ('memo' or 'transcript')
        prompt: Custom prompt for transcription
        force: Force overwrite existing files
        keep_intermediate: Keep intermediate transcript files

    Returns:
        bool: True if successful, False otherwise
    """
    if not audio_file.exists():
        logging.error(f"Audio file not found: {audio_file}")
        return False

    # Check if final output already exists
    final_output = output_dir / f"{audio_file.stem}.txt"
    if final_output.exists() and not force:
        logging.info(
            f"Output already exists (use --force to overwrite): {final_output}"
        )
        return True

    # Step 1: Transcription
    transcript_file = run_transcription(
        audio_file, temp_dir, transcription_model, prompt, force
    )
    if not transcript_file:
        return False

    # Step 2: Post-processing
    processed_file = run_post_processing(
        transcript_file, output_dir, processing_model, mode, force
    )
    if not processed_file:
        return False

    # Step 3: Cleanup intermediate files (unless keeping them)
    if not keep_intermediate:
        try:
            transcript_file.unlink()
            logging.debug(f"Cleaned up intermediate file: {transcript_file}")
        except Exception as e:
            logging.warning(f"Failed to clean up {transcript_file}: {e}")

    logging.info(
        f"✅ Successfully processed: {audio_file.name} → {processed_file.name}"
    )
    return True


def get_quality_settings(quality: str) -> tuple[str, str]:
    """
    Get model settings based on quality preference.

    Args:
        quality: Quality level ('fast', 'balanced', 'high')

    Returns:
        tuple: (transcription_model, processing_model)
    """
    if quality == "fast":
        return "gpt-4o-mini-transcribe", "gpt-4.1-nano"
    elif quality == "balanced":
        return "gpt-4o-transcribe", "gpt-4.1-mini"
    elif quality == "high":
        return "gpt-4o-transcribe", "gpt-4o-mini"
    else:
        # Default to balanced
        return "gpt-4o-mini-transcribe", "gpt-4.1-nano"


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Streamlined audio processing pipeline for voice memos",
        epilog="""
Examples:
  %(prog)s voicemail.mp3                           # Process single file
  %(prog)s *.mp3 *.m4a                            # Process multiple files
  %(prog)s audio/*.mp3 -O processed/               # Custom output directory  
  %(prog)s memo.mp3 --quality high                 # Higher quality processing
  %(prog)s meeting.mp3 --prompt "Project meeting"  # Custom context
  %(prog)s *.mp3 --transcript                     # Transcript mode (no summarization)
  %(prog)s *.mp3 --keep-intermediate               # Keep transcription files
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional arguments
    parser.add_argument(
        "audio_files", nargs="+", help="Path(s) to audio file(s) to process"
    )

    # Quality and model options
    parser.add_argument(
        "--quality",
        choices=["fast", "balanced", "high"],
        default="fast",
        help="Processing quality/speed tradeoff (default: fast)",
    )

    # Processing mode
    parser.add_argument(
        "--transcript",
        action="store_true",
        help="Use transcript mode (preserve exact wording) instead of memo mode (summarize)",
    )

    # Output options
    parser.add_argument(
        "-O",
        "--output-dir",
        type=str,
        help="Output directory (default: processed/YYYY-MM-DD/)",
    )

    # Context and prompts
    parser.add_argument(
        "--prompt",
        type=str,
        help="Custom prompt for better transcription accuracy (e.g., names, context)",
    )

    # Control options
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite existing output files",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate transcription files (for debugging)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args()


def main() -> None:
    """
    Main pipeline function.
    """
    args = parse_args()

    # Set up logging
    setup_logging(args.debug)

    # Get default prompt from environment if not provided
    prompt = args.prompt
    if not prompt:
        prompt = os.getenv("DEFAULT_PROMPT")
        if prompt:
            logging.info(f"Using default prompt from .env: {prompt[:50]}...")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = get_default_output_dir()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    # Get quality settings
    transcription_model, processing_model = get_quality_settings(args.quality)
    logging.info(
        f"Quality: {args.quality} (transcription: {transcription_model}, processing: {processing_model})"
    )

    # Determine processing mode
    mode = "transcript" if args.transcript else "memo"
    logging.info(f"Processing mode: {mode}")

    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory(prefix="pipeline_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        logging.debug(f"Temporary directory: {temp_dir}")

        # Process each audio file
        audio_files = [Path(f) for f in args.audio_files]
        successful = 0
        failed = 0

        logging.info(f"Processing {len(audio_files)} audio file(s)...")

        for audio_file in audio_files:
            success = process_audio_file(
                audio_file,
                output_dir,
                temp_dir,
                transcription_model,
                processing_model,
                mode,
                prompt,
                args.force,
                args.keep_intermediate,
            )

            if success:
                successful += 1
            else:
                failed += 1

        # Print summary
        logging.info(f"\n{'='*50}")
        logging.info(f"PIPELINE COMPLETE")
        logging.info(f"{'='*50}")
        logging.info(f"Files processed: {len(audio_files)}")
        logging.info(f"Successful: {successful}")
        logging.info(f"Failed: {failed}")
        logging.info(f"Output directory: {output_dir.absolute()}")

        if failed > 0:
            logging.warning(f"{failed} files failed to process. Check logs above.")
            sys.exit(1)
        else:
            logging.info("All files processed successfully! 🎉")


if __name__ == "__main__":
    main()
