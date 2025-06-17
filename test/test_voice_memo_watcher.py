#!/usr/bin/env python3

import unittest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from voice_memo_watcher import VoiceMemoWatcher, AudioFile


class TestVoiceMemoWatcher(unittest.TestCase):
    """Test cases for VoiceMemoWatcher."""

    def setUp(self):
        """Set up test environment with temporary directories."""
        self.test_dir = tempfile.mkdtemp()
        self.audio_in = Path(self.test_dir) / "audio_in"
        self.audio_out = Path(self.test_dir) / "audio_out"
        self.transcript_out = Path(self.test_dir) / "transcript_out"

        # Create directories
        self.audio_in.mkdir(parents=True)
        self.audio_out.mkdir(parents=True)
        self.transcript_out.mkdir(parents=True)

        self.watcher = VoiceMemoWatcher(
            self.audio_in, self.audio_out, self.transcript_out
        )

    def tearDown(self):
        """Clean up temporary directories."""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)

    def test_directory_creation(self):
        """Test that directories are created if they don't exist."""
        # Remove directories
        shutil.rmtree(self.audio_in)
        shutil.rmtree(self.audio_out)
        shutil.rmtree(self.transcript_out)

        # Create new watcher - should create directories
        watcher = VoiceMemoWatcher(self.audio_in, self.audio_out, self.transcript_out)

        self.assertTrue(self.audio_in.exists())
        self.assertTrue(self.audio_out.exists())
        self.assertTrue(self.transcript_out.exists())

    def test_find_audio_files(self):
        """Test finding audio files in input directory."""
        # Copy test audio files
        test_audio_dir = Path(__file__).parent / "test_audio"
        test_files = ["test_basic.mp3", "test_basic.wav"]

        for test_file in test_files:
            src = test_audio_dir / test_file
            if src.exists():
                shutil.copy2(src, self.audio_in / test_file)

        # Add a non-audio file that should be ignored
        (self.audio_in / "test.txt").write_text("This should be ignored")

        files = self.watcher.find_audio_files()

        # Should find only audio files
        self.assertEqual(len(files), 2)
        self.assertTrue(all(isinstance(f, AudioFile) for f in files))
        self.assertTrue(all(f.path.suffix in [".mp3", ".wav"] for f in files))

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        test_cases = [
            ("Hello World", "Hello World"),
            ("File/Name:With*Bad?Chars", "File_Name_With_Bad_Chars"),
            ("Multiple   Spaces   Here", "Multiple Spaces Here"),
            ("..Leading.Dots..", "Leading.Dots"),
            ("  Trim Whitespace  ", "Trim Whitespace"),
            ("A" * 150, "A" * 100),  # Length limit
            ("", "untitled"),  # Empty string
            ("???", "___"),  # Only invalid chars become underscores
        ]

        for input_str, expected in test_cases:
            result = self.watcher.sanitize_filename(input_str)
            self.assertEqual(result, expected, f"Failed for input: {input_str}")

    def test_extract_title_from_transcript(self):
        """Test title extraction from transcript content."""
        # Create a test transcript
        transcript_path = self.transcript_out / "test.txt"

        # Test with good content
        transcript_path.write_text(
            "Meeting Notes for Project X\n\nThis is the content..."
        )
        title = self.watcher.extract_title_from_transcript(transcript_path)
        self.assertEqual(title, "Meeting Notes for Project X")

        # Test with short first line
        transcript_path.write_text(
            "Hi\n\nThis is a longer second line that could be a title\n\nMore content..."
        )
        title = self.watcher.extract_title_from_transcript(transcript_path)
        self.assertEqual(title, "This is a longer second line that could be a title")

        # Test with no suitable lines (fallback to first 50 chars)
        transcript_path.write_text(
            "Short\nLines\nOnly\nHere\n\nBut here is a very long line that exceeds our character limit and should be truncated appropriately"
        )
        title = self.watcher.extract_title_from_transcript(transcript_path)
        self.assertIsNotNone(title)
        self.assertLessEqual(len(title), 50)

    def test_process_file_simulation(self):
        """Test file processing workflow without actually running transcription."""
        # Copy a test file
        test_audio_dir = Path(__file__).parent / "test_audio"
        test_file = "test_basic.mp3"
        src = test_audio_dir / test_file

        if src.exists():
            dst = self.audio_in / test_file
            shutil.copy2(src, dst)

            # Create audio file object
            audio_file = AudioFile(path=dst, timestamp=datetime(2024, 1, 15, 10, 30, 0))

            # Simulate transcript creation (since we can't run full pipeline in test)
            transcript_file = (
                self.transcript_out / f"{test_file.rsplit('.', 1)[0]}_transcript.txt"
            )
            transcript_file.write_text(
                "Test Meeting Notes\n\nThis is a test transcript..."
            )

            # Test filename generation
            title = self.watcher.extract_title_from_transcript(transcript_file)
            timestamp_str = audio_file.timestamp.strftime("%Y%m%d_%H%M%S")

            expected_audio_name = f"{timestamp_str}_{title}.mp3"
            expected_transcript_name = f"{timestamp_str}_{title}.md"

            self.assertEqual(
                expected_audio_name, "20240115_103000_Test Meeting Notes.mp3"
            )
            self.assertEqual(
                expected_transcript_name, "20240115_103000_Test Meeting Notes.md"
            )

    def test_empty_directory(self):
        """Test behavior with empty input directory."""
        files = self.watcher.find_audio_files()
        self.assertEqual(len(files), 0)

        results = self.watcher.run()
        self.assertEqual(results["total"], 0)
        self.assertEqual(results["processed"], 0)
        self.assertEqual(results["failed"], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests using actual test files."""

    @classmethod
    def setUpClass(cls):
        """Set up integration test environment."""
        cls.test_root = Path(__file__).parent / "test_integration"
        cls.test_root.mkdir(exist_ok=True)

        # Create test directories
        cls.audio_in = cls.test_root / "audio_in"
        cls.audio_out = cls.test_root / "audio_out"
        cls.transcript_out = cls.test_root / "transcript_out"

        for dir_path in [cls.audio_in, cls.audio_out, cls.transcript_out]:
            dir_path.mkdir(exist_ok=True, parents=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up integration test environment."""
        if cls.test_root.exists():
            shutil.rmtree(cls.test_root)

    def setUp(self):
        """Clean directories before each test."""
        for dir_path in [self.audio_in, self.audio_out, self.transcript_out]:
            for file in dir_path.iterdir():
                if file.is_file():
                    file.unlink()

    def test_full_workflow_with_test_file(self):
        """Test the complete workflow with an actual test file."""
        # Copy a test file
        test_audio_dir = Path(__file__).parent / "test_audio"
        test_file = "test_basic.mp3"
        src = test_audio_dir / test_file

        if not src.exists():
            self.skipTest(f"Test file {src} not found")

        # Copy to input directory
        dst = self.audio_in / test_file
        shutil.copy2(src, dst)

        # Create watcher
        watcher = VoiceMemoWatcher(self.audio_in, self.audio_out, self.transcript_out)

        # Run processing
        results = watcher.run()

        # Check results
        self.assertEqual(results["total"], 1)

        # Check that input file was moved
        self.assertFalse(dst.exists(), "Input file should have been moved")

        # Check output directory
        audio_files = list(self.audio_out.glob("*.mp3"))
        transcript_files = list(self.transcript_out.glob("*.md"))

        if results["processed"] == 1:
            self.assertEqual(
                len(audio_files), 1, "Should have one audio file in output"
            )
            self.assertEqual(
                len(transcript_files), 1, "Should have one transcript file"
            )

            # Check filename format
            audio_name = audio_files[0].name
            self.assertTrue(
                audio_name.startswith("20"), "Filename should start with year"
            )
            self.assertIn("_", audio_name, "Filename should contain underscore")


def run_safe_test():
    """Run a safe test with copied files to ensure no data loss."""
    print("\n=== Running Safe Integration Test ===\n")

    # Create a safe test environment
    safe_test_dir = Path.cwd() / "test" / "safe_integration_test"
    safe_test_dir.mkdir(exist_ok=True, parents=True)

    audio_in = safe_test_dir / "audio_in"
    audio_out = safe_test_dir / "audio_out"
    transcript_out = safe_test_dir / "transcript_out"

    for dir_path in [audio_in, audio_out, transcript_out]:
        dir_path.mkdir(exist_ok=True)

    # Copy a test file
    test_audio_dir = Path(__file__).parent / "test_audio"
    test_file = "test_basic.mp3"
    src = test_audio_dir / test_file

    if src.exists():
        dst = audio_in / test_file
        shutil.copy2(src, dst)
        print(f"Copied test file to: {dst}")

        # Create and run watcher
        watcher = VoiceMemoWatcher(audio_in, audio_out, transcript_out)
        results = watcher.run()

        print(f"\nResults: {results}")

        # Show what happened
        print(f"\nAudio output files: {list(audio_out.glob('*'))}")
        print(f"Transcript files: {list(transcript_out.glob('*'))}")

        # Clean up
        response = input("\nClean up test directory? (y/n): ")
        if response.lower() == "y":
            shutil.rmtree(safe_test_dir)
            print("Test directory cleaned up.")
    else:
        print(f"Test file {src} not found!")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[""], exit=False, verbosity=2)

    # Offer to run safe integration test
    print("\n" + "=" * 50)
    response = input("\nRun safe integration test with file copying? (y/n): ")
    if response.lower() == "y":
        run_safe_test()
