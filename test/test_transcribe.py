#!/usr/bin/env python3

"""
Test suite for the transcribe.py script.

This module tests the transcription functionality by:
1. Generating known test audio files using OpenAI's text-to-speech API
2. Running transcription on these files
3. Validating the output against expected content

Test audio files are kept as permanent fixtures to avoid regenerating them
and to provide consistent test data across runs.
"""

import argparse
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment


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


def setup_test_environment() -> OpenAI:
    """
    Set up the test environment by loading environment variables and creating OpenAI client.

    Returns:
        OpenAI client instance

    Raises:
        SystemExit: If OPENAI_API_KEY is not found
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY not found in environment variables")
        logging.error("Please ensure your .env file contains a valid OpenAI API key")
        exit(1)
    return OpenAI(api_key=api_key)


def get_test_content() -> List[Tuple[str, str]]:
    """
    Get test content for audio generation.

    Returns:
        List of tuples containing (filename_prefix, text_content)
    """
    return [
        (
            "test_basic",
            "Hello, this is a test message for the transcription system. "
            "My name is Jud, not Judge. I'm calling about scheduling a meeting "
            "for next Tuesday at three PM. Please call me back when you get this message.",
        ),
        (
            "test_names",
            "This message includes several names that might be challenging: "
            "Jud Dagnall, Catherine, and Michael. We're planning to meet at "
            "the San Francisco office on Market Street next week.",
        ),
        (
            "test_technical",
            "We need to discuss the API integration, SDK documentation, "
            "and the machine learning model performance metrics. The CPU usage "
            "has increased by twenty percent since the last deployment.",
        ),
    ]


def generate_test_audio_file(
    client: OpenAI, text: str, output_path: Path, voice: str = "nova"
) -> None:
    """
    Generate an audio file using OpenAI's text-to-speech API.

    Args:
        client: OpenAI client instance
        text: Text content to convert to speech
        output_path: Path where the audio file should be saved
        voice: Voice to use for speech synthesis
    """
    logging.info(f"Generating audio file: {output_path}")

    response = client.audio.speech.create(
        model="tts-1", voice=voice, input=text, response_format="mp3"
    )

    # Save the MP3 file
    with open(output_path, "wb") as f:
        f.write(response.content)
    logging.info(f"Created: {output_path}")


def convert_mp3_to_wav(mp3_path: Path, wav_path: Path) -> None:
    """
    Convert MP3 file to WAV format using pydub.

    Args:
        mp3_path: Path to source MP3 file
        wav_path: Path for output WAV file
    """
    logging.info(f"Converting {mp3_path} to {wav_path}")
    audio = AudioSegment.from_mp3(str(mp3_path))
    audio.export(str(wav_path), format="wav")
    logging.info(f"Created: {wav_path}")


def create_test_audio_files(
    force_regenerate: bool = False,
) -> Dict[str, Dict[str, Path]]:
    """
    Create test audio files if they don't exist or if force_regenerate is True.

    Args:
        force_regenerate: If True, regenerate files even if they exist

    Returns:
        Dictionary mapping test names to file paths and expected content
    """
    client = setup_test_environment()
    test_dir = Path("test_audio")
    test_dir.mkdir(exist_ok=True)

    test_files = {}
    test_content = get_test_content()

    for filename_prefix, text_content in test_content:
        mp3_path = test_dir / f"{filename_prefix}.mp3"
        wav_path = test_dir / f"{filename_prefix}.wav"

        # Generate files if they don't exist or if regeneration is forced
        if force_regenerate or not mp3_path.exists():
            generate_test_audio_file(client, text_content, mp3_path)
        else:
            logging.info(f"Using existing file: {mp3_path}")

        if force_regenerate or not wav_path.exists():
            convert_mp3_to_wav(mp3_path, wav_path)
        else:
            logging.info(f"Using existing file: {wav_path}")

        test_files[filename_prefix] = {
            "mp3_path": mp3_path,
            "wav_path": wav_path,
            "expected_content": text_content,
            "expected_words": set(
                word.lower().strip(".,!?") for word in text_content.split()
            ),
        }

    return test_files


def run_transcription(
    audio_path: Path,
    complex_json: bool = False,
    out_dir: Optional[Path] = None,
    use_temp_dir: bool = True,
) -> Tuple[str, Dict]:
    """
    Run transcription on an audio file and return the results.

    Args:
        audio_path: Path to the audio file to transcribe
        complex_json: Whether to use --complex-json flag
        out_dir: Optional output directory
        use_temp_dir: Whether to use a temporary output directory

    Returns:
        Tuple of (plain_text, json_data)
    """
    logging.info(
        f"Transcribing: {audio_path} (complex_json={complex_json}, out_dir={out_dir})"
    )

    # Use temporary directory if no specific out_dir provided and use_temp_dir is True
    temp_dir = None
    actual_out_dir = out_dir

    if use_temp_dir and not out_dir:
        import tempfile

        temp_dir = tempfile.mkdtemp()
        actual_out_dir = Path(temp_dir)

    # Build command with absolute path for audio file
    cmd = ["python", "transcribe.py", str(audio_path.absolute()), "--force"]
    if complex_json:
        cmd.append("--complex-json")
    if actual_out_dir:
        cmd.extend(["--out-dir", str(actual_out_dir)])

    # Run transcription from parent directory
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd().parent)

    if result.returncode != 0:
        logging.error(f"Transcription failed for {audio_path}")
        logging.error(f"stdout: {result.stdout}")
        logging.error(f"stderr: {result.stderr}")
        raise RuntimeError(f"Transcription failed: {result.stderr}")

    # Read output files
    base_name = audio_path.stem
    output_dir = actual_out_dir if actual_out_dir else Path.cwd().parent
    txt_path = output_dir / f"{base_name}.txt"
    json_path = output_dir / f"{base_name}.json"

    try:
        # Read text output
        if not txt_path.exists():
            raise FileNotFoundError(f"Expected text output file not found: {txt_path}")

        with open(txt_path, "r", encoding="utf-8") as f:
            plain_text = f.read().strip()

        # Read JSON output if complex_json was used
        json_data = {}
        if complex_json:
            if not json_path.exists():
                raise FileNotFoundError(
                    f"Expected JSON output file not found: {json_path}"
                )

            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

        return plain_text, json_data

    finally:
        # Clean up temporary directory if created
        if temp_dir:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


def calculate_word_accuracy(expected_words: set, transcribed_text: str) -> float:
    """
    Calculate word-level accuracy between expected and transcribed text.

    Args:
        expected_words: Set of expected words (lowercase, no punctuation)
        transcribed_text: Transcribed text to compare

    Returns:
        Accuracy score as a float between 0 and 1
    """
    transcribed_words = set(
        word.lower().strip(".,!?") for word in transcribed_text.split()
    )

    if not expected_words:
        return 1.0 if not transcribed_words else 0.0

    correct_words = expected_words.intersection(transcribed_words)
    accuracy = len(correct_words) / len(expected_words)

    return accuracy


def validate_transcription_output(
    audio_path: Path,
    expected_content: str,
    expected_words: set,
    plain_text: str,
    json_data: Dict,
    complex_json: bool,
) -> Dict:
    """
    Validate transcription output against expected content.

    Args:
        audio_path: Path to the source audio file
        expected_content: Expected text content
        expected_words: Set of expected words
        plain_text: Transcribed plain text
        json_data: Transcribed JSON data (if complex_json=True)
        complex_json: Whether complex JSON output was requested

    Returns:
        Dictionary containing validation results
    """
    results = {
        "file": str(audio_path),
        "format": audio_path.suffix,
        "complex_json": complex_json,
        "success": True,
        "errors": [],
        "warnings": [],
        "metrics": {},
    }

    # Basic validation: ensure we got some text
    if not plain_text:
        results["success"] = False
        results["errors"].append("No transcribed text returned")
        return results

    # Calculate word accuracy
    word_accuracy = calculate_word_accuracy(expected_words, plain_text)
    results["metrics"]["word_accuracy"] = word_accuracy
    results["metrics"]["transcribed_length"] = len(plain_text)
    results["metrics"]["expected_length"] = len(expected_content)

    # Validate accuracy threshold
    if word_accuracy < 0.7:  # 70% minimum word accuracy
        results["warnings"].append(f"Low word accuracy: {word_accuracy:.2%}")

    # Validate JSON structure if complex_json was used
    if complex_json:
        if not json_data:
            results["success"] = False
            results["errors"].append(
                "No JSON data returned despite --complex-json flag"
            )
        else:
            # Check required JSON fields
            required_fields = ["text", "segments"]
            for field in required_fields:
                if field not in json_data:
                    results["errors"].append(f"Missing required JSON field: {field}")

            # Validate segments structure
            if "segments" in json_data:
                if not isinstance(json_data["segments"], list):
                    results["errors"].append("JSON segments field should be a list")
                else:
                    results["metrics"]["segment_count"] = len(json_data["segments"])

                    # Check segment structure
                    for i, segment in enumerate(json_data["segments"]):
                        if not isinstance(segment, dict):
                            results["errors"].append(
                                f"Segment {i} should be a dictionary"
                            )
                            continue

                        segment_fields = ["start", "end", "text"]
                        for field in segment_fields:
                            if field not in segment:
                                results["errors"].append(
                                    f"Segment {i} missing field: {field}"
                                )

    # Check for specific expected content (names, etc.)
    expected_names = ["Jud", "Catherine", "Michael"]
    transcribed_lower = plain_text.lower()

    for name in expected_names:
        if (
            name.lower() in expected_content.lower()
            and name.lower() not in transcribed_lower
        ):
            results["warnings"].append(
                f"Expected name '{name}' not found in transcription"
            )

    if results["errors"]:
        results["success"] = False

    return results


def cleanup_generated_files() -> None:
    """
    Clean up generated .txt and .json files from the test_audio directory.

    This function removes transcription output files that may have been created
    during testing to keep the test directory clean.
    """
    test_dir = Path("test_audio")
    if not test_dir.exists():
        return

    # Remove .txt and .json files from test_audio directory
    files_to_remove = []
    files_to_remove.extend(test_dir.glob("*.txt"))
    files_to_remove.extend(test_dir.glob("*.json"))

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


def test_transcription_accuracy(force_regenerate: bool = False) -> None:
    """
    Main test function that creates audio files and validates transcription accuracy.

    Args:
        force_regenerate: Force regeneration of test audio files
    """
    logging.info("=== Transcription Test Suite ===")

    # Clean up any existing generated files first
    cleanup_generated_files()

    # Create test audio files
    logging.info("Setting up test audio files...")
    test_files = create_test_audio_files(force_regenerate)
    logging.info(f"Created {len(test_files)} test cases")

    # Test results storage
    all_results = []

    # Test each file with both formats and both output modes
    for test_name, file_info in test_files.items():
        logging.info(f"Testing: {test_name}")
        logging.info("-" * 50)

        for format_name, audio_path in [
            ("MP3", file_info["mp3_path"]),
            ("WAV", file_info["wav_path"]),
        ]:
            for complex_json in [False, True]:
                mode_desc = "with JSON" if complex_json else "text only"
                logging.info(f"  {format_name} format ({mode_desc})")

                try:
                    # Run transcription
                    plain_text, json_data = run_transcription(audio_path, complex_json)

                    # Validate results
                    validation_result = validate_transcription_output(
                        audio_path,
                        file_info["expected_content"],
                        file_info["expected_words"],
                        plain_text,
                        json_data,
                        complex_json,
                    )

                    all_results.append(validation_result)

                    # Log results
                    status = "✓ PASS" if validation_result["success"] else "✗ FAIL"
                    accuracy = validation_result["metrics"].get("word_accuracy", 0)
                    logging.info(f"    {status} - Word accuracy: {accuracy:.1%}")

                    if validation_result["errors"]:
                        for error in validation_result["errors"]:
                            logging.error(f"    ERROR: {error}")

                    if validation_result["warnings"]:
                        for warning in validation_result["warnings"]:
                            logging.warning(f"    WARNING: {warning}")

                    logging.info(
                        f"    Transcribed: '{plain_text[:100]}{'...' if len(plain_text) > 100 else ''}'"
                    )

                except Exception as e:
                    logging.error(f"    ✗ FAIL - Exception: {str(e)}")
                    all_results.append(
                        {
                            "file": str(audio_path),
                            "format": audio_path.suffix,
                            "complex_json": complex_json,
                            "success": False,
                            "errors": [str(e)],
                            "warnings": [],
                            "metrics": {},
                        }
                    )

    # Summary
    logging.info("=== Test Summary ===")
    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results if r["success"])
    failed_tests = total_tests - passed_tests

    logging.info(f"Total tests: {total_tests}")
    logging.info(f"Passed: {passed_tests}")
    logging.info(f"Failed: {failed_tests}")

    if failed_tests > 0:
        logging.error("Failed tests:")
        for result in all_results:
            if not result["success"]:
                mode = "JSON" if result["complex_json"] else "TXT"
                logging.error(
                    f"  {result['file']} ({mode}): {', '.join(result['errors'])}"
                )

    # Calculate average accuracy
    accuracies = [
        r["metrics"].get("word_accuracy", 0) for r in all_results if r["success"]
    ]
    if accuracies:
        avg_accuracy = sum(accuracies) / len(accuracies)
        logging.info(f"Average word accuracy: {avg_accuracy:.1%}")

    logging.info(
        "Test audio files are preserved in the test_audio/ directory for future runs."
    )

    # Clean up generated files after testing
    logging.info("Cleaning up generated test files...")
    cleanup_generated_files()


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the test script.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Test suite for transcription accuracy validation",
        epilog="""
This test suite validates transcription accuracy by:
1. Generating known test audio files using OpenAI's text-to-speech API
2. Running transcription on these files (both MP3 and WAV formats)
3. Validating output against expected content and JSON structure

Test audio files are preserved as permanent fixtures in test_audio/ directory.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration of test audio files using text-to-speech API. "
        "By default, existing audio files are reused for faster testing.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging for detailed test execution information",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to run the transcription test suite.
    """
    args = parse_args()

    # Set up logging
    setup_logging(args.debug)

    if args.regenerate:
        logging.info("Regenerating test audio files...")

    # Run the test suite
    test_transcription_accuracy(args.regenerate)


if __name__ == "__main__":
    main()
