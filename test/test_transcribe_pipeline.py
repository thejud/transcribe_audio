#!/usr/bin/env python3

"""
Test Suite for Transcribe Pipeline

This test suite validates the transcribe_pipeline.py tool using the existing
test audio files. It tests the complete pipeline from audio input to
processed text output.

Usage:
    cd test && python test_transcribe_pipeline.py
    cd test && python test_transcribe_pipeline.py --debug
    cd test && python test_transcribe_pipeline.py --quality high

Author: Jud Dagnall (with Claude Code)
Version: 1.0
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional


def setup_logging(debug: bool = False) -> None:
    """
    Configure logging for the test suite.

    Args:
        debug: Enable debug-level logging
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )


def get_test_audio_files() -> List[Path]:
    """
    Get list of test audio files from test_audio directory.

    Returns:
        List[Path]: Available test audio files
    """
    test_audio_dir = Path("test_audio")
    if not test_audio_dir.exists():
        logging.error("test_audio directory not found. Run from test/ directory.")
        sys.exit(1)

    # Get all audio files (mp3 and wav)
    audio_files = []
    for pattern in ["*.mp3", "*.wav"]:
        audio_files.extend(test_audio_dir.glob(pattern))

    if not audio_files:
        logging.error("No test audio files found in test_audio/")
        sys.exit(1)

    # Sort for consistent ordering
    return sorted(audio_files)


def run_pipeline_test(
    audio_files: List[Path],
    output_dir: Path,
    quality: str = "fast",
    transcript_mode: bool = False,
    debug: bool = False,
) -> tuple[bool, List[str]]:
    """
    Run the transcribe pipeline on test audio files.

    Args:
        audio_files: List of audio files to process
        output_dir: Output directory for results
        quality: Quality setting (fast, balanced, high)
        transcript_mode: Use transcript mode instead of memo mode
        debug: Enable debug logging

    Returns:
        tuple: (success, error_messages)
    """
    cmd = [
        "python",
        "../transcribe_pipeline.py",
        *[str(f) for f in audio_files],
        "-O",
        str(output_dir),
        "--quality",
        quality,
    ]

    if transcript_mode:
        cmd.append("--transcript")
    if debug:
        cmd.append("--debug")

    try:
        logging.info(f"Running pipeline with quality: {quality}")
        logging.info(f"Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            logging.info("Pipeline completed successfully")
            return True, []
        else:
            error_msg = f"Pipeline failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f": {result.stderr}"
            return False, [error_msg]

    except subprocess.TimeoutExpired:
        return False, ["Pipeline timed out after 5 minutes"]
    except Exception as e:
        return False, [f"Error running pipeline: {str(e)}"]


def validate_pipeline_output(
    audio_files: List[Path], output_dir: Path
) -> tuple[bool, List[str]]:
    """
    Validate that the pipeline produced expected output files.

    Args:
        audio_files: Original audio files
        output_dir: Output directory to check

    Returns:
        tuple: (success, error_messages)
    """
    errors = []

    # Check that output directory exists
    if not output_dir.exists():
        return False, [f"Output directory not found: {output_dir}"]

    # Check for expected output files
    for audio_file in audio_files:
        expected_txt = output_dir / f"{audio_file.stem}.txt"

        if not expected_txt.exists():
            errors.append(f"Missing output file: {expected_txt}")
            continue

        # Check file has content
        try:
            with open(expected_txt, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                errors.append(f"Empty output file: {expected_txt}")
            elif len(content) < 10:
                errors.append(
                    f"Output file too short: {expected_txt} ({len(content)} chars)"
                )
            else:
                logging.info(f"✓ Valid output: {expected_txt} ({len(content)} chars)")

        except Exception as e:
            errors.append(f"Error reading {expected_txt}: {str(e)}")

    return len(errors) == 0, errors


def test_pipeline_quality_settings() -> bool:
    """
    Test pipeline with different quality settings.

    Returns:
        bool: True if all tests pass
    """
    logging.info("=== Testing Pipeline Quality Settings ===")

    # Use a single test file for speed
    audio_files = [Path("test_audio/test_basic.mp3")]
    all_passed = True

    for quality in ["fast", "balanced", "high"]:
        logging.info(f"\n--- Testing quality: {quality} ---")

        with tempfile.TemporaryDirectory(
            prefix=f"pipeline_test_{quality}_"
        ) as temp_dir:
            output_dir = Path(temp_dir)

            # Run pipeline
            success, errors = run_pipeline_test(
                audio_files, output_dir, quality=quality
            )

            if not success:
                logging.error(f"Quality {quality} test failed:")
                for error in errors:
                    logging.error(f"  - {error}")
                all_passed = False
                continue

            # Validate output
            valid, validation_errors = validate_pipeline_output(audio_files, output_dir)

            if not valid:
                logging.error(f"Quality {quality} validation failed:")
                for error in validation_errors:
                    logging.error(f"  - {error}")
                all_passed = False
            else:
                logging.info(f"✅ Quality {quality} test passed")

    return all_passed


def test_pipeline_modes() -> bool:
    """
    Test pipeline memo vs transcript modes.

    Returns:
        bool: True if all tests pass
    """
    logging.info("=== Testing Pipeline Processing Modes ===")

    # Use a single test file for speed
    audio_files = [Path("test_audio/test_basic.mp3")]
    all_passed = True

    for mode_name, transcript_flag in [("memo", False), ("transcript", True)]:
        logging.info(f"\n--- Testing {mode_name} mode ---")

        with tempfile.TemporaryDirectory(
            prefix=f"pipeline_test_{mode_name}_"
        ) as temp_dir:
            output_dir = Path(temp_dir)

            # Run pipeline
            success, errors = run_pipeline_test(
                audio_files, output_dir, transcript_mode=transcript_flag
            )

            if not success:
                logging.error(f"Mode {mode_name} test failed:")
                for error in errors:
                    logging.error(f"  - {error}")
                all_passed = False
                continue

            # Validate output
            valid, validation_errors = validate_pipeline_output(audio_files, output_dir)

            if not valid:
                logging.error(f"Mode {mode_name} validation failed:")
                for error in validation_errors:
                    logging.error(f"  - {error}")
                all_passed = False
            else:
                logging.info(f"✅ Mode {mode_name} test passed")

    return all_passed


def test_pipeline_batch_processing() -> bool:
    """
    Test pipeline with multiple audio files.

    Returns:
        bool: True if test passes
    """
    logging.info("=== Testing Batch Processing ===")

    # Use multiple test files
    audio_files = [
        Path("test_audio/test_basic.mp3"),
        Path("test_audio/test_names.wav"),
    ]

    # Filter to only files that exist
    existing_files = [f for f in audio_files if f.exists()]
    if not existing_files:
        logging.warning("No test files found for batch processing test")
        return True

    logging.info(
        f"Testing with {len(existing_files)} files: {[f.name for f in existing_files]}"
    )

    with tempfile.TemporaryDirectory(prefix="pipeline_test_batch_") as temp_dir:
        output_dir = Path(temp_dir)

        # Run pipeline
        success, errors = run_pipeline_test(existing_files, output_dir)

        if not success:
            logging.error("Batch processing test failed:")
            for error in errors:
                logging.error(f"  - {error}")
            return False

        # Validate output
        valid, validation_errors = validate_pipeline_output(existing_files, output_dir)

        if not valid:
            logging.error("Batch processing validation failed:")
            for error in validation_errors:
                logging.error(f"  - {error}")
            return False

        logging.info("✅ Batch processing test passed")
        return True


def test_pipeline_output_organization() -> bool:
    """
    Test that pipeline organizes output correctly.

    Returns:
        bool: True if test passes
    """
    logging.info("=== Testing Output Organization ===")

    audio_files = [Path("test_audio/test_basic.mp3")]

    with tempfile.TemporaryDirectory(prefix="pipeline_test_org_") as temp_dir:
        base_output = Path(temp_dir)

        # Run pipeline (should create dated subdirectory)
        success, errors = run_pipeline_test(audio_files, base_output)

        if not success:
            logging.error("Output organization test failed:")
            for error in errors:
                logging.error(f"  - {error}")
            return False

        # Check that dated subdirectory was created
        from datetime import datetime

        today = datetime.now().strftime("%Y-%m-%d")
        expected_subdir = base_output / today

        if not expected_subdir.exists():
            logging.error(f"Expected dated subdirectory not found: {expected_subdir}")
            return False

        # Validate output in subdirectory
        valid, validation_errors = validate_pipeline_output(
            audio_files, expected_subdir
        )

        if not valid:
            logging.error("Output organization validation failed:")
            for error in validation_errors:
                logging.error(f"  - {error}")
            return False

        logging.info("✅ Output organization test passed")
        return True


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Test suite for transcribe_pipeline.py",
        epilog="""
Examples:
  %(prog)s                    # Run all tests
  %(prog)s --debug            # Run with debug logging  
  %(prog)s --quality high     # Test with high quality only
  %(prog)s --quick            # Run quick tests only
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--quality",
        choices=["fast", "balanced", "high"],
        help="Test specific quality level only",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only (skip batch processing)",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main test function.
    """
    args = parse_args()

    # Set up logging
    setup_logging(args.debug)

    # Verify we're in the test directory
    if not Path("test_audio").exists():
        logging.error("Must run from test/ directory")
        sys.exit(1)

    # Verify transcribe_pipeline.py exists
    pipeline_script = Path("../transcribe_pipeline.py")
    if not pipeline_script.exists():
        logging.error("transcribe_pipeline.py not found in parent directory")
        sys.exit(1)

    logging.info("Starting transcribe_pipeline.py test suite")
    logging.info(f"Testing with pipeline: {pipeline_script.absolute()}")

    # Get available test files
    test_files = get_test_audio_files()
    logging.info(f"Available test audio files: {[f.name for f in test_files]}")

    # Run tests
    all_tests_passed = True
    start_time = time.time()

    # Test 1: Quality settings
    if args.quality:
        # Test only specific quality
        logging.info(f"\n--- Testing quality: {args.quality} ---")
        audio_files = [Path("test_audio/test_basic.mp3")]
        with tempfile.TemporaryDirectory(
            prefix=f"pipeline_test_{args.quality}_"
        ) as temp_dir:
            output_dir = Path(temp_dir)
            success, errors = run_pipeline_test(
                audio_files, output_dir, quality=args.quality, debug=args.debug
            )
            if success:
                valid, validation_errors = validate_pipeline_output(
                    audio_files, output_dir
                )
                if valid:
                    logging.info(f"✅ Quality {args.quality} test passed")
                else:
                    all_tests_passed = False
                    for error in validation_errors:
                        logging.error(f"  - {error}")
            else:
                all_tests_passed = False
                for error in errors:
                    logging.error(f"  - {error}")
    else:
        # Test all quality settings
        if not test_pipeline_quality_settings():
            all_tests_passed = False

    # Test 2: Processing modes
    if not test_pipeline_modes():
        all_tests_passed = False

    # Test 3: Output organization
    if not test_pipeline_output_organization():
        all_tests_passed = False

    # Test 4: Batch processing (unless quick mode)
    if not args.quick:
        if not test_pipeline_batch_processing():
            all_tests_passed = False
    else:
        logging.info("Skipping batch processing test (quick mode)")

    # Print summary
    end_time = time.time()
    duration = end_time - start_time

    logging.info(f"\n{'='*60}")
    logging.info("TRANSCRIBE PIPELINE TEST SUMMARY")
    logging.info(f"{'='*60}")
    logging.info(f"Total duration: {duration:.1f} seconds")

    if all_tests_passed:
        logging.info("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        logging.error("❌ SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
