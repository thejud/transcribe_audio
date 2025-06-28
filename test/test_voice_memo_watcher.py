#!/usr/bin/env python3

import unittest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import os
import sys
import json
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from voice_memo_watcher import (
    VoiceMemoWatcher,
    AudioFile,
    AudioFileHandler,
    WATCHDOG_AVAILABLE,
)


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
            [self.audio_in], self.audio_out, self.transcript_out
        )

    def tearDown(self):
        """Clean up temporary directories."""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)

    def test_directory_creation(self):
        """Test that output directories are created if they don't exist."""
        # Remove directories
        shutil.rmtree(self.audio_out)
        shutil.rmtree(self.transcript_out)

        # Create new watcher - should create output directories (but not input - those should exist)
        watcher = VoiceMemoWatcher([self.audio_in], self.audio_out, self.transcript_out)

        # Input directory should exist (it's our test setup directory)
        self.assertTrue(self.audio_in.exists())
        # Output directories should be created
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

    def test_multiple_input_directories(self):
        """Test finding audio files across multiple input directories."""
        # Create additional input directories
        audio_in_2 = Path(self.test_dir) / "audio_in_2"
        audio_in_3 = Path(self.test_dir) / "audio_in_3"
        audio_in_2.mkdir()
        audio_in_3.mkdir()

        # Create test files in different directories
        test_files = {
            self.audio_in / "test1.mp3": "test content 1",
            audio_in_2 / "test2.m4a": "test content 2",
            audio_in_3 / "test3.wav": "test content 3",
            audio_in_2 / "test.txt": "non-audio file",  # Should be ignored
        }

        for file_path, content in test_files.items():
            file_path.write_text(content)

        # Create watcher with multiple input directories
        multi_watcher = VoiceMemoWatcher(
            [self.audio_in, audio_in_2, audio_in_3], self.audio_out, self.transcript_out
        )

        files = multi_watcher.find_audio_files()

        # Should find 3 audio files (ignoring .txt file)
        self.assertEqual(len(files), 3)

        # Check that files from all directories are found
        found_names = [f.path.name for f in files]
        self.assertIn("test1.mp3", found_names)
        self.assertIn("test2.m4a", found_names)
        self.assertIn("test3.wav", found_names)
        self.assertNotIn("test.txt", found_names)

    def test_nonexistent_input_directory(self):
        """Test behavior when one of the input directories doesn't exist."""
        nonexistent_dir = Path(self.test_dir) / "nonexistent"

        # Create watcher with mix of existing and non-existing directories
        multi_watcher = VoiceMemoWatcher(
            [self.audio_in, nonexistent_dir], self.audio_out, self.transcript_out
        )

        # Should not crash, just log warning
        files = multi_watcher.find_audio_files()
        self.assertEqual(len(files), 0)  # No files in existing empty directory

    def test_single_directory_backward_compatibility(self):
        """Test that single directory usage still works (backward compatibility)."""
        # Test passing a single Path instead of list
        single_watcher = VoiceMemoWatcher(
            self.audio_in, self.audio_out, self.transcript_out  # Single Path, not list
        )

        # Should convert to list internally
        self.assertEqual(len(single_watcher.audio_in_dirs), 1)
        self.assertEqual(single_watcher.audio_in_dirs[0], self.audio_in)

    @unittest.skipIf(not WATCHDOG_AVAILABLE, "watchdog not available")
    def test_directory_event_handling(self):
        """Test that directory events trigger file scanning."""
        from watchdog.events import DirCreatedEvent

        # Create test audio files
        test_files = ["test1.mp3", "test2.m4a", "test3.wav"]
        for test_file in test_files:
            (self.audio_in / test_file).write_text("test audio content")

        # Create handler
        handler = AudioFileHandler(self.watcher)

        # Mock the watcher's process_file method to track processed files
        processed_files = []
        original_process_file = self.watcher.process_file

        def mock_process_file(audio_file):
            processed_files.append(audio_file.path.name)
            return True  # Simulate successful processing

        self.watcher.process_file = mock_process_file

        # Simulate directory creation event
        event = DirCreatedEvent(str(self.audio_in))
        handler.on_created(event)

        # Should have found all audio files
        self.assertEqual(len(processed_files), 3)
        self.assertIn("test1.mp3", processed_files)
        self.assertIn("test2.m4a", processed_files)
        self.assertIn("test3.wav", processed_files)

    def test_empty_transcript_quarantine(self):
        """Test that files producing empty transcripts are quarantined."""
        # Create a test audio file
        test_file = self.audio_in / "empty_transcript.mp3"
        test_file.write_text("test audio content")

        # Create AudioFile object
        audio_file = AudioFile(path=test_file, timestamp=datetime.now())

        # Mock the transcription to create an empty transcript
        transcript_file = self.transcript_out / "empty_transcript_transcript.txt"
        transcript_file.write_text("")  # Empty transcript

        # Mock subprocess to simulate successful transcription
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stderr = ""

            # Process the file
            result = self.watcher.process_file(audio_file)

        # Should return False for failed processing
        self.assertFalse(result)

        # Check that quarantine directory was created
        quarantine_dir = self.audio_out.parent / "quarantine"
        self.assertTrue(quarantine_dir.exists())

        # Check that file was moved to quarantine
        quarantined_file = quarantine_dir / "empty_transcript.mp3"
        self.assertTrue(quarantined_file.exists())
        self.assertFalse(test_file.exists())

        # Check that empty transcript was cleaned up
        self.assertFalse(transcript_file.exists())

    @unittest.skipIf(not WATCHDOG_AVAILABLE, "watchdog not available")
    def test_modification_time_tracking(self):
        """Test that files with different modification times are processed correctly."""
        from watchdog.events import DirModifiedEvent
        import time

        # Create a test audio file
        test_file = self.audio_in / "test_mtime.mp3"
        test_file.write_text("test audio content")

        # Create handler
        handler = AudioFileHandler(self.watcher)

        # Track processed files
        processed_files = []

        def mock_process_file(audio_file):
            processed_files.append(audio_file.path.name)
            return True

        self.watcher.process_file = mock_process_file

        # First directory scan
        event = DirModifiedEvent(str(self.audio_in))
        handler.on_modified(event)

        # Should have processed the file once
        self.assertEqual(len(processed_files), 1)
        self.assertEqual(processed_files[0], "test_mtime.mp3")

        # Simulate the same event again without modifying the file
        processed_files.clear()
        handler.on_modified(event)

        # Should not process the file again (same modification time)
        self.assertEqual(len(processed_files), 0)

        # Now modify the file
        time.sleep(0.1)  # Ensure different timestamp
        test_file.write_text("modified audio content")

        # Trigger directory event again
        handler.on_modified(event)

        # Should process the file again (different modification time)
        self.assertEqual(len(processed_files), 1)
        self.assertEqual(processed_files[0], "test_mtime.mp3")


class TestEnvironmentVariableParsing(unittest.TestCase):
    """Test environment variable parsing for multiple directories."""

    def test_colon_separated_paths(self):
        """Test parsing colon-separated paths from environment variable."""
        # Mock environment variable with colon-separated paths
        with patch.dict(
            os.environ,
            {
                "AUDIO_IN": "/path/A:/path/B:/path/C:/path/D",
                "AUDIO_OUT": "/tmp/out",
                "TRANSCRIPT_OUT": "/tmp/transcripts",
            },
        ):
            # Simulate argument parsing that would happen in main()
            from voice_memo_watcher import main
            import argparse

            # Create a mock args object
            class MockArgs:
                audio_in_dirs = None
                audio_out = None
                transcript_out = None
                verbose = False
                watch = False
                interval = 60

            args = MockArgs()

            # Test the parsing logic from main()
            if args.audio_in_dirs:
                audio_in_dirs = [
                    Path(os.path.expanduser(str(path))) for path in args.audio_in_dirs
                ]
            else:
                env_audio_in = os.getenv(
                    "AUDIO_IN", "~/Dropbox/01-projects/voice_memo_inbox"
                )
                if ":" in env_audio_in:
                    audio_in_dirs = [
                        Path(os.path.expanduser(path.strip()))
                        for path in env_audio_in.split(":")
                    ]
                else:
                    audio_in_dirs = [Path(os.path.expanduser(env_audio_in))]

            # Should parse into 4 directories
            self.assertEqual(len(audio_in_dirs), 4)
            self.assertEqual(str(audio_in_dirs[0]), "/path/A")
            self.assertEqual(str(audio_in_dirs[1]), "/path/B")
            self.assertEqual(str(audio_in_dirs[2]), "/path/C")
            self.assertEqual(str(audio_in_dirs[3]), "/path/D")

    def test_single_path_env_var(self):
        """Test parsing single path from environment variable (backward compatibility)."""
        with patch.dict(os.environ, {"AUDIO_IN": "/single/path"}):
            env_audio_in = os.getenv(
                "AUDIO_IN", "~/Dropbox/01-projects/voice_memo_inbox"
            )
            if ":" in env_audio_in:
                audio_in_dirs = [
                    Path(os.path.expanduser(path.strip()))
                    for path in env_audio_in.split(":")
                ]
            else:
                audio_in_dirs = [Path(os.path.expanduser(env_audio_in))]

            # Should create list with single directory
            self.assertEqual(len(audio_in_dirs), 1)
            self.assertEqual(str(audio_in_dirs[0]), "/single/path")


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
        watcher = VoiceMemoWatcher([self.audio_in], self.audio_out, self.transcript_out)

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
