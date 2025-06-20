#!/usr/bin/env python3

"""
Copy Audio Files - Recursively copy/move audio files with directory flattening

This script recursively finds audio files in an input directory and copies them
to an output directory, flattening the directory structure. All files end up in
a single output directory regardless of their original nested location.

Features:
- Recursively scans for audio files (mp3, m4a, wav, flac, ogg, opus, aac, wma)
- Flattens directory structure (all files go to single output directory)
- Preserves file timestamps using shutil.copy2
- Handles duplicate filenames by appending _1, _2, etc.
- Skips existing files by default (unless --force is used)
- Can delete source files after copy (move operation) with --delete
- Reads default directories from .env file

Environment Variables (.env file):
    AUDIO_COPY_INPUT=~/Music/source     # Default input directory
    AUDIO_COPY_OUTPUT=~/Music/organized  # Default output directory

Examples:
    # Copy using defaults from .env
    python copy_audio_files.py
    
    # Copy from specific directories
    python copy_audio_files.py -i /music/raw -o /music/flat
    
    # Move files (delete after successful copy)
    python copy_audio_files.py --delete
    
    # Force overwrite existing files
    python copy_audio_files.py --force
    
    # Move with force overwrite
    python copy_audio_files.py -d -f

Duplicate Handling:
    - If file exists with same size: Skip (unless --force)
    - If file exists with different size: Rename to name_1.ext, name_2.ext, etc.
    - With --force: Always overwrite existing files

Safety:
    - Files are only deleted AFTER successful copy when using --delete
    - Clear warnings shown when delete mode is active
    - All operations preserve original file timestamps
"""

import os
import shutil
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv


def find_audio_files(directory: Path) -> list[Path]:
    """Recursively find all audio files in directory."""
    audio_extensions = {
        ".mp3",
        ".m4a",
        ".wav",
        ".flac",
        ".ogg",
        ".opus",
        ".aac",
        ".wma",
    }
    audio_files = []

    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
            audio_files.append(file_path)

    return audio_files


def copy_files(
    input_dir: Path, output_dir: Path, force: bool = False, delete: bool = False
) -> tuple[int, int]:
    """Copy all audio files from input to output directory, flattening the structure."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    audio_files = find_audio_files(input_dir)

    if not audio_files:
        print("No audio files found.")
        return 0, 0

    print(f"Found {len(audio_files)} audio files to copy.")

    copied = 0
    skipped = 0

    # Keep track of filenames to handle duplicates
    filename_counter = {}

    for file_path in audio_files:
        try:
            # Get base filename
            base_name = file_path.stem
            extension = file_path.suffix

            # Handle duplicate filenames by adding counter
            output_filename = f"{base_name}{extension}"
            output_path = output_dir / output_filename

            # Handle existing files
            if output_path.exists():
                if not force:
                    # Check if it's the same file (by size)
                    if output_path.stat().st_size == file_path.stat().st_size:
                        print(f"Skipped (already exists): {output_filename}")
                        skipped += 1
                        continue

                    # Different file with same name - create unique name
                    counter = filename_counter.get(base_name, 1)
                    while output_path.exists():
                        output_filename = f"{base_name}_{counter}{extension}"
                        output_path = output_dir / output_filename
                        counter += 1
                    filename_counter[base_name] = counter

            # Copy file (copy2 preserves timestamps)
            shutil.copy2(file_path, output_path)

            # Show relative path from input for context
            relative_input = file_path.relative_to(input_dir)

            # Delete source file if requested and copy was successful
            if delete:
                try:
                    file_path.unlink()
                    print(
                        f"Moved: {relative_input} -> {output_filename} (source deleted)"
                    )
                except Exception as e:
                    print(
                        f"Copied: {relative_input} -> {output_filename} (failed to delete source: {e})"
                    )
                    logging.error(f"Failed to delete source file {file_path}: {e}")
            else:
                print(f"Copied: {relative_input} -> {output_filename}")

            copied += 1

        except Exception as e:
            logging.error(f"Failed to copy {file_path}: {e}")

    return copied, skipped


def main():
    # Load environment variables
    load_dotenv()

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Copy audio files recursively from input to output directory (flattened structure)"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="Input directory (default: from .env AUDIO_COPY_INPUT)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory (default: from .env AUDIO_COPY_OUTPUT)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite existing files (default: skip existing)",
    )
    parser.add_argument(
        "-d",
        "--delete",
        action="store_true",
        help="Delete source files after successful copy (move operation)",
    )

    args = parser.parse_args()

    # Get directories from args or environment
    input_dir = args.input or Path(
        os.path.expanduser(os.getenv("AUDIO_COPY_INPUT", "./audio_input"))
    )
    output_dir = args.output or Path(
        os.path.expanduser(os.getenv("AUDIO_COPY_OUTPUT", "./audio_output"))
    )

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1

    print(f"Input directory: {input_dir.absolute()}")
    print(f"Output directory: {output_dir.absolute()}")

    # Warn about delete option
    if args.delete:
        print(
            "WARNING: Delete mode enabled - source files will be removed after copying!"
        )
        print()

    # Copy files
    copied, skipped = copy_files(input_dir, output_dir, args.force, args.delete)

    operation = "moved" if args.delete else "copied"
    print(f"\nCompleted:")
    print(f"  Files {operation}: {copied}")
    print(f"  Files skipped: {skipped}")
    print(f"  Total processed: {copied + skipped}")

    return 0


if __name__ == "__main__":
    exit(main())
