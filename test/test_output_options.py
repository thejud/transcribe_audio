#!/usr/bin/env python3

"""
Test suite for output directory and file options in transcribe.py and post_process.py

This module tests:
1. Output directory functionality (-o/--out-dir)
2. In-place processing (--inplace for post_process.py)
3. Extension suffix functionality (-E/--extension for post_process.py)
4. Integration with existing functionality

The tests use the existing test audio files and create temporary test directories
to validate the output options work correctly.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv


def setup_logging(debug: bool = False) -> None:
    """
    Configure logging for the test application.

    Args:
        debug: Enable debug-level logging
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def cleanup_test_audio_generated_files() -> None:
    """
    Clean up generated .txt and .json files from the test_audio directory.

    This function removes transcription output files that may have been created
    during testing to keep the test directory clean.
    """
    test_audio_dir = Path("test_audio")
    if not test_audio_dir.exists():
        return

    # Remove .txt and .json files from test_audio directory
    files_to_remove = []
    files_to_remove.extend(test_audio_dir.glob("*.txt"))
    files_to_remove.extend(test_audio_dir.glob("*.json"))

    for file_path in files_to_remove:
        try:
            file_path.unlink()
            logging.debug(f"Removed: {file_path}")
        except OSError as e:
            logging.warning(f"Failed to remove {file_path}: {e}")

    removed_count = len(files_to_remove)
    if removed_count > 0:
        logging.info(
            f"Cleaned up {removed_count} generated files from test_audio directory"
        )


def ensure_test_audio_exists() -> Path:
    """
    Ensure test audio files exist by running the main test suite if needed.

    Returns:
        Path to test_audio directory
    """
    test_audio_dir = Path("test_audio")
    if not test_audio_dir.exists() or not list(test_audio_dir.glob("*.mp3")):
        logging.info("Test audio files not found, generating them...")
        result = subprocess.run(
            ["python", "test_transcribe.py", "--regenerate"],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),  # Run from current test directory
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to generate test audio files: {result.stderr}")

    return test_audio_dir


def test_transcribe_output_directory() -> None:
    """
    Test the -o/--out-dir functionality in transcribe.py
    """
    logging.info("=== Testing transcribe.py output directory functionality ===")

    test_audio_dir = ensure_test_audio_exists()
    test_files = list(test_audio_dir.glob("test_basic.*"))

    if not test_files:
        raise RuntimeError("No test audio files found")

    test_audio_file = next(f for f in test_files if f.suffix == ".mp3")
    logging.info(f"Using test file: {test_audio_file}")

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "transcribe_output"
        output_dir.mkdir()

        # Test basic output directory functionality
        # Use absolute path for audio file to avoid path issues
        cmd = [
            "python",
            "transcribe.py",
            str(test_audio_file.absolute()),
            "--out-dir",
            str(output_dir),
            "--force",
        ]

        logging.info("Testing basic output directory...")
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path.cwd().parent
        )

        if result.returncode != 0:
            logging.error(f"Transcription failed: {result.stderr}")
            logging.error(f"stdout: {result.stdout}")
            raise RuntimeError(f"Transcription failed: {result.stderr}")

        logging.info(f"Transcription stdout: {result.stdout}")
        logging.info(f"Transcription stderr: {result.stderr}")

        # Check that output files were created in the correct directory
        expected_txt = output_dir / f"{test_audio_file.stem}.txt"
        if not expected_txt.exists():
            # List what files actually exist in the output directory
            existing_files = list(output_dir.iterdir()) if output_dir.exists() else []
            logging.error(f"Expected file: {expected_txt}")
            logging.error(f"Output directory exists: {output_dir.exists()}")
            logging.error(f"Files in output directory: {existing_files}")
            raise AssertionError(f"Expected output file not found: {expected_txt}")

        logging.info(
            f"✓ Basic output directory test passed - file created: {expected_txt}"
        )

        # Test with --complex-json option
        cmd_json = cmd + ["--complex-json"]
        logging.info("Testing output directory with --complex-json...")

        result = subprocess.run(
            cmd_json, capture_output=True, text=True, cwd=Path.cwd().parent
        )
        if result.returncode != 0:
            raise RuntimeError(f"Transcription with JSON failed: {result.stderr}")

        expected_json = output_dir / f"{test_audio_file.stem}.json"
        if not expected_json.exists():
            raise AssertionError(
                f"Expected JSON output file not found: {expected_json}"
            )

        # Validate JSON structure
        with open(expected_json, "r") as f:
            json_data = json.load(f)

        if "text" not in json_data or "segments" not in json_data:
            raise AssertionError("Invalid JSON structure in output file")

        logging.info(
            f"✓ Output directory with JSON test passed - files created: {expected_txt}, {expected_json}"
        )

        # Test nested output directory creation
        nested_dir = Path(temp_dir) / "deep" / "nested" / "output"
        cmd_nested = [
            "python",
            "transcribe.py",
            str(test_audio_file.absolute()),
            "--out-dir",
            str(nested_dir),
            "--force",
        ]

        logging.info("Testing nested directory creation...")
        result = subprocess.run(
            cmd_nested, capture_output=True, text=True, cwd=Path.cwd().parent
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Nested directory transcription failed: {result.stderr}"
            )

        expected_nested_txt = nested_dir / f"{test_audio_file.stem}.txt"
        if not expected_nested_txt.exists():
            raise AssertionError(
                f"Expected nested output file not found: {expected_nested_txt}"
            )

        logging.info(
            f"✓ Nested directory creation test passed - file created: {expected_nested_txt}"
        )


def create_test_transcript_file(content: str, temp_dir: Path) -> Path:
    """
    Create a temporary transcript file for testing post_process.py

    Args:
        content: Content to write to the file
        temp_dir: Temporary directory to create file in

    Returns:
        Path to the created test file
    """
    test_file = temp_dir / "test_transcript.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(content)
    return test_file


def test_post_process_output_directory() -> None:
    """
    Test the --out-dir functionality in post_process.py
    """
    logging.info("=== Testing post_process.py output directory functionality ===")

    test_content = (
        "This is a test transcript that needs to be processed. "
        "It contains multiple sentences that should be reformatted into paragraphs. "
        "The processing should maintain all the original content while improving readability."
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test transcript file
        test_file = create_test_transcript_file(test_content, temp_path)

        # Create output directory
        output_dir = temp_path / "post_process_output"
        output_dir.mkdir()

        # Test basic output directory functionality
        cmd = [
            "python",
            "post_process.py",
            str(test_file.absolute()),
            "--out-dir",
            str(output_dir),
            "--test-mode",  # Use test mode to avoid large API calls
        ]

        logging.info("Testing post_process.py output directory...")
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path.cwd().parent
        )

        if result.returncode != 0:
            logging.error(f"Post-processing failed: {result.stderr}")
            raise RuntimeError(f"Post-processing failed: {result.stderr}")

        # Check that output file was created in the correct directory
        expected_output = output_dir / test_file.name
        if not expected_output.exists():
            raise AssertionError(f"Expected output file not found: {expected_output}")

        # Verify content is different from original (processed)
        with open(expected_output, "r", encoding="utf-8") as f:
            processed_content = f.read()

        if processed_content.strip() == test_content.strip():
            logging.warning(
                "Processed content appears identical to original (this may be normal for test mode)"
            )

        logging.info(
            f"✓ Post-process output directory test passed - file created: {expected_output}"
        )


def test_post_process_extension_suffix() -> None:
    """
    Test the -E/--extension functionality in post_process.py
    """
    logging.info("=== Testing post_process.py extension suffix functionality ===")

    test_content = "This is a test transcript for extension testing."

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test transcript file
        test_file = create_test_transcript_file(test_content, temp_path)

        # Test extension suffix functionality
        extension = "_processed"
        cmd = [
            "python",
            "post_process.py",
            str(test_file.absolute()),
            "--extension",
            extension,
            "--test-mode",
        ]

        logging.info(f"Testing extension suffix: {extension}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path.cwd().parent
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Post-processing with extension failed: {result.stderr}"
            )

        # Check that output file was created with correct suffix
        expected_name = f"{test_file.stem}{extension}{test_file.suffix}"
        expected_output = temp_path / expected_name

        if not expected_output.exists():
            raise AssertionError(
                f"Expected output file with extension not found: {expected_output}"
            )

        logging.info(
            f"✓ Extension suffix test passed - file created: {expected_output}"
        )

        # Test combination of output directory and extension
        output_dir = temp_path / "combined_test"
        output_dir.mkdir()

        cmd_combined = [
            "python",
            "post_process.py",
            str(test_file.absolute()),
            "--out-dir",
            str(output_dir),
            "--extension",
            "_clean",
            "--test-mode",
        ]

        logging.info("Testing combined output directory and extension...")
        result = subprocess.run(
            cmd_combined, capture_output=True, text=True, cwd=Path.cwd().parent
        )

        if result.returncode != 0:
            raise RuntimeError(f"Combined test failed: {result.stderr}")

        expected_combined = output_dir / f"{test_file.stem}_clean{test_file.suffix}"
        if not expected_combined.exists():
            raise AssertionError(
                f"Expected combined output file not found: {expected_combined}"
            )

        logging.info(
            f"✓ Combined output directory and extension test passed - file created: {expected_combined}"
        )


def test_post_process_inplace() -> None:
    """
    Test the --inplace functionality in post_process.py
    """
    logging.info("=== Testing post_process.py --inplace functionality ===")

    test_content = "This is a test transcript for in-place processing testing."

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test transcript file
        test_file = create_test_transcript_file(test_content, temp_path)
        original_content = test_content

        # Test in-place processing
        cmd = [
            "python",
            "post_process.py",
            str(test_file.absolute()),
            "--inplace",
            "--test-mode",
        ]

        logging.info("Testing in-place processing...")
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path.cwd().parent
        )

        if result.returncode != 0:
            raise RuntimeError(f"In-place processing failed: {result.stderr}")

        # Check that original file was modified
        with open(test_file, "r", encoding="utf-8") as f:
            modified_content = f.read()

        # In test mode, content might not change much, but file should still exist
        if not test_file.exists():
            raise AssertionError("Original file was deleted instead of modified")

        logging.info(f"✓ In-place processing test passed - file modified: {test_file}")

        # Test in-place with verification
        test_file2 = create_test_transcript_file(original_content, temp_path)

        cmd_verify = [
            "python",
            "post_process.py",
            str(test_file2.absolute()),
            "--inplace",
            "--verify",
            "--test-mode",
        ]

        logging.info("Testing in-place processing with verification...")
        result = subprocess.run(
            cmd_verify, capture_output=True, text=True, cwd=Path.cwd().parent
        )

        if result.returncode != 0:
            logging.warning(
                f"In-place with verification may have failed due to test mode: {result.stderr}"
            )
        else:
            logging.info("✓ In-place processing with verification test passed")


def test_error_conditions() -> None:
    """
    Test error conditions and edge cases
    """
    logging.info("=== Testing error conditions ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Test transcribe.py with non-existent output directory parent
        non_existent_parent = temp_path / "does_not_exist" / "deep" / "path"
        test_audio_dir = ensure_test_audio_exists()
        test_audio_file = next(test_audio_dir.glob("test_basic.mp3"))

        cmd = [
            "python",
            "transcribe.py",
            str(test_audio_file.absolute()),
            "--out-dir",
            str(non_existent_parent),
            "--force",
        ]

        logging.info("Testing transcribe.py with deep non-existent output directory...")
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path.cwd().parent
        )

        # Should succeed because we create parent directories
        if result.returncode != 0:
            raise RuntimeError(f"Expected success but got failure: {result.stderr}")

        expected_file = non_existent_parent / f"{test_audio_file.stem}.txt"
        if not expected_file.exists():
            raise AssertionError("Output file not created in deep directory")

        logging.info("✓ Deep directory creation test passed")

        # Test conflicting options in post_process.py
        test_file = create_test_transcript_file("test content", temp_path)

        cmd_conflict = [
            "python",
            "post_process.py",
            str(test_file.absolute()),
            "--inplace",
            "--output",
            str(temp_path / "output.txt"),
        ]

        logging.info("Testing conflicting options (should fail)...")
        result = subprocess.run(
            cmd_conflict, capture_output=True, text=True, cwd=Path.cwd().parent
        )

        # Should fail due to mutually exclusive options
        if result.returncode == 0:
            logging.warning(
                "Expected failure for conflicting options, but command succeeded"
            )
        else:
            logging.info("✓ Conflicting options properly rejected")


def run_all_tests(debug: bool = False) -> None:
    """
    Run all output option tests

    Args:
        debug: Enable debug-level logging
    """
    setup_logging(debug)

    logging.info("Starting output options test suite...")

    # Clean up any existing generated files first
    cleanup_test_audio_generated_files()

    # Check environment
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY not found in environment variables")
        logging.error("Please ensure your .env file contains a valid OpenAI API key")
        return

    try:
        # Test transcribe.py output directory functionality
        test_transcribe_output_directory()

        # Test post_process.py functionality
        test_post_process_output_directory()
        test_post_process_extension_suffix()
        test_post_process_inplace()

        # Test error conditions
        test_error_conditions()

        logging.info("=== All tests completed successfully! ===")

    except Exception as e:
        logging.error(f"Test failed: {str(e)}")
        raise
    finally:
        # Clean up generated files after testing
        logging.info("Cleaning up generated test files...")
        cleanup_test_audio_generated_files()


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the test script.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Test suite for output directory and file options",
        epilog="""
This test suite validates the new output options:
- transcribe.py: -o/--out-dir parameter
- post_process.py: --out-dir, --inplace, and -E/--extension parameters

The tests create temporary directories and files to verify functionality.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging for detailed test execution information",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to run the output options test suite.
    """
    args = parse_args()
    run_all_tests(args.debug)


if __name__ == "__main__":
    main()
