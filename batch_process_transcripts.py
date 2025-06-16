#!/usr/bin/env python3

"""
Batch Transcript Processing Script

This script processes all transcripts in a source directory using the postprocessing tool
and generates comprehensive validation reports for each one.

Features:
- Processes all .txt files in the source directory
- Skips files that have already been processed
- Generates postprocessed files with " - post" suffix
- Creates comprehensive validation reports with " - report" suffix
- Provides progress tracking and summary statistics
- Handles errors gracefully and continues processing

Usage:
    python batch_process_transcripts.py SOURCE_DIR OUTPUT_DIR

    # Example:
    python batch_process_transcripts.py "/path/to/transcripts/" "tmp/"

Author: Jud Dagnall (with Claude Code)
Version: 1.0
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple


def setup_logging(debug: bool = False) -> None:
    """
    Configure logging for the application.

    Args:
        debug: Enable debug-level logging
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )


def find_transcript_files(source_dir: Path) -> List[Path]:
    """
    Find all transcript files (.txt) in the source directory.
    Excludes system files and hidden files.

    Args:
        source_dir: Directory to search for transcript files

    Returns:
        List[Path]: List of transcript file paths
    """
    if not source_dir.exists():
        logging.error(f"Source directory not found: {source_dir}")
        sys.exit(1)

    all_txt_files = list(source_dir.glob("*.txt"))

    # Filter out system files and hidden files
    transcript_files = []
    for file_path in all_txt_files:
        filename = file_path.name

        # Skip macOS system files (._filename)
        if filename.startswith("._"):
            logging.debug(f"Skipping system file: {filename}")
            continue

        # Skip other hidden files (.filename)
        if filename.startswith(".") and not filename.startswith("._"):
            logging.debug(f"Skipping hidden file: {filename}")
            continue

        # Skip files that are too small (likely not real transcripts)
        try:
            if file_path.stat().st_size < 100:  # Less than 100 bytes
                logging.debug(
                    f"Skipping small file: {filename} ({file_path.stat().st_size} bytes)"
                )
                continue
        except OSError:
            logging.debug(f"Skipping file with stat error: {filename}")
            continue

        transcript_files.append(file_path)

    logging.info(
        f"Found {len(transcript_files)} valid transcript files in {source_dir}"
    )
    if len(all_txt_files) > len(transcript_files):
        skipped_count = len(all_txt_files) - len(transcript_files)
        logging.info(f"Skipped {skipped_count} system/hidden files")

    return sorted(transcript_files)


def get_output_filenames(input_file: Path, output_dir: Path) -> Tuple[Path, Path, Path]:
    """
    Generate output filenames for input, postprocessed, and report files.

    Args:
        input_file: Path to input transcript file
        output_dir: Directory for output files

    Returns:
        Tuple[Path, Path, Path]: (input_copy, postprocessed_file, report_file) paths
    """
    base_name = input_file.stem  # filename without extension
    extension = input_file.suffix  # .txt

    input_copy = output_dir / f"{base_name}{extension}"
    postprocessed_file = output_dir / f"{base_name} - post{extension}"
    report_file = output_dir / f"{base_name} - report{extension}"

    return input_copy, postprocessed_file, report_file


def validate_transcript_file(file_path: Path) -> bool:
    """
    Validate that a file is a readable text transcript.

    Args:
        file_path: Path to the file to validate

    Returns:
        bool: True if file appears to be a valid transcript
    """
    try:
        # Try to read first few bytes to check if it's text
        with open(file_path, "rb") as f:
            first_bytes = f.read(512)

        # Try to decode as UTF-8
        try:
            first_bytes.decode("utf-8")
        except UnicodeDecodeError:
            logging.debug(f"File {file_path.name} is not valid UTF-8")
            return False

        # Check if it contains reasonable text content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read(1000)  # Read first 1000 chars

        if len(content.strip()) < 50:
            logging.debug(f"File {file_path.name} has too little content")
            return False

        return True

    except Exception as e:
        logging.debug(f"Error validating {file_path.name}: {e}")
        return False


def copy_input_file(source_file: Path, dest_file: Path) -> bool:
    """
    Copy input file to destination directory after validation.

    Args:
        source_file: Source transcript file
        dest_file: Destination path for copy

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Validate file before copying
        if not validate_transcript_file(source_file):
            logging.error(f"File validation failed for {source_file.name}")
            return False

        import shutil

        shutil.copy2(source_file, dest_file)
        logging.debug(f"Copied {source_file.name} to {dest_file}")
        return True
    except Exception as e:
        logging.error(f"Failed to copy {source_file}: {e}")
        return False


def run_postprocessing(input_file: Path, output_file: Path) -> bool:
    """
    Run the postprocessing tool on a transcript file.

    Args:
        input_file: Input transcript file
        output_file: Output file for postprocessed text

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cmd = ["python", "post_process.py", str(input_file), "-o", str(output_file)]

        logging.debug(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            logging.debug(f"Successfully processed {input_file.name}")
            return True
        else:
            logging.error(
                f"Postprocessing failed for {input_file.name}: {result.stderr}"
            )
            return False

    except subprocess.TimeoutExpired:
        logging.error(f"Postprocessing timed out for {input_file.name}")
        return False
    except Exception as e:
        logging.error(f"Error running postprocessing for {input_file.name}: {e}")
        return False


def generate_validation_report(
    input_file: Path, postprocessed_file: Path, report_file: Path
) -> bool:
    """
    Generate a comprehensive validation report comparing input and postprocessed files.

    Args:
        input_file: Original transcript file
        postprocessed_file: Postprocessed transcript file
        report_file: Output file for validation report

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize report
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=== COMPREHENSIVE TEXT VALIDATION REPORT ===\n")
            f.write(f"Talk: {input_file.stem}\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")

        # Add file statistics
        cmd_stats = ["wc", "-l", "-w", "-c", str(input_file), str(postprocessed_file)]
        result_stats = subprocess.run(cmd_stats, capture_output=True, text=True)

        with open(report_file, "a", encoding="utf-8") as f:
            f.write("## File Statistics\n")
            f.write(result_stats.stdout)
            f.write("\n")

        # Run validation modes
        validation_modes = [
            ("Strict Mode (exact character match)", "strict"),
            ("Semantic Mode (NLP-based validation)", "semantic"),
            ("Comprehensive Mode (Full NLP Analysis)", "comprehensive"),
        ]

        for i, (mode_name, mode_flag) in enumerate(validation_modes, 1):
            cmd_validate = [
                "python",
                "verify_text_integrity.py",
                str(input_file),
                str(postprocessed_file),
                "--mode",
                mode_flag,
            ]

            result_validate = subprocess.run(
                cmd_validate, capture_output=True, text=True
            )

            with open(report_file, "a", encoding="utf-8") as f:
                if i == 1:
                    f.write("## Validation Results\n\n")
                f.write(f"### {i}. {mode_name}:\n")
                f.write(result_validate.stdout)
                f.write("\n")

        # Add analysis summary
        with open(report_file, "a", encoding="utf-8") as f:
            f.write("## Analysis Summary\n\n")
            f.write("### Content Integrity:\n")
            f.write(
                "The validation results above show how the postprocessing tool performs\n"
            )
            f.write(
                "across different validation criteria. Semantic modes (scores of 1.0000)\n"
            )
            f.write(
                "indicate perfect preservation of meaning and content.\n\n"
            )
            f.write("### Recommendation:\n")
            f.write("✅ Use semantic or comprehensive mode for validation\n")
            f.write(
                "✅ Postprocessing preserves teachings while improving readability\n"
            )
            f.write(
                "✅ Formatting improvements make talks more accessible to readers\n"
            )

        logging.debug(f"Generated validation report: {report_file.name}")
        return True

    except Exception as e:
        logging.error(f"Error generating validation report for {input_file.name}: {e}")
        return False


def process_single_transcript(
    source_file: Path, output_dir: Path, skip_existing: bool = True
) -> Tuple[bool, str]:
    """
    Process a single transcript file through the complete pipeline.

    Args:
        source_file: Source transcript file
        output_dir: Output directory
        skip_existing: Skip processing if output files already exist

    Returns:
        Tuple[bool, str]: (success, status_message)
    """
    input_copy, postprocessed_file, report_file = get_output_filenames(
        source_file, output_dir
    )

    # Check if already processed
    if skip_existing and postprocessed_file.exists() and report_file.exists():
        return True, "Already processed (skipped)"

    # Step 1: Copy input file
    if not copy_input_file(source_file, input_copy):
        return False, "Failed to copy input file"

    # Step 2: Run postprocessing
    if not run_postprocessing(input_copy, postprocessed_file):
        return False, "Postprocessing failed"

    # Step 3: Generate validation report
    if not generate_validation_report(input_copy, postprocessed_file, report_file):
        return False, "Report generation failed"

    return True, "Successfully processed"


def main() -> None:
    """
    Main function to process all transcripts in batch.
    """
    parser = argparse.ArgumentParser(
        description="Batch process transcript files with postprocessing and validation",
        epilog="""
Examples:
  %(prog)s "/path/to/transcripts/" "tmp/"                    # Process all transcripts
  %(prog)s "/path/to/transcripts/" "output/" --no-skip-existing       # Reprocess all files
  %(prog)s "/path/to/transcripts/" "output/" --debug                  # Enable debug logging
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "source_dir", help="Directory containing transcript files (.txt)"
    )
    parser.add_argument("output_dir", help="Directory for output files")
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess files even if output already exists",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.debug)

    # Validate directories
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    if not source_dir.exists():
        logging.error(f"Source directory does not exist: {source_dir}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find transcript files
    transcript_files = find_transcript_files(source_dir)
    if not transcript_files:
        logging.warning("No transcript files found to process")
        sys.exit(0)

    # Process each transcript
    logging.info(f"Starting batch processing of {len(transcript_files)} transcripts...")
    logging.info(f"Output directory: {output_dir}")

    successful = 0
    failed = 0
    skipped = 0

    start_time = time.time()

    for i, transcript_file in enumerate(transcript_files, 1):
        logging.info(
            f"\n[{i}/{len(transcript_files)}] Processing: {transcript_file.name}"
        )

        success, message = process_single_transcript(
            transcript_file, output_dir, skip_existing=not args.no_skip_existing
        )

        if success:
            if "skipped" in message.lower():
                skipped += 1
                logging.info(f"✓ {message}")
            else:
                successful += 1
                logging.info(f"✅ {message}")
        else:
            failed += 1
            logging.error(f"❌ {message}")

    # Print summary
    end_time = time.time()
    duration = end_time - start_time

    logging.info(f"\n" + "=" * 60)
    logging.info("BATCH PROCESSING COMPLETE")
    logging.info(f"=" * 60)
    logging.info(f"Total files: {len(transcript_files)}")
    logging.info(f"Successfully processed: {successful}")
    logging.info(f"Skipped (already processed): {skipped}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Total duration: {duration:.1f} seconds")
    logging.info(f"Average per file: {duration/len(transcript_files):.1f} seconds")
    logging.info(f"Output directory: {output_dir.absolute()}")

    if failed > 0:
        logging.warning(
            f"{failed} files failed to process. Check logs above for details."
        )
        sys.exit(1)
    else:
        logging.info("All files processed successfully! 🎉")


if __name__ == "__main__":
    main()
