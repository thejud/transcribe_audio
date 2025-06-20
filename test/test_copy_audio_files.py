#!/usr/bin/env python3

"""
Test script for copy_audio_files.py
Tests all major functionality including flattening, timestamps, duplicates, and delete.
"""

import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_test_structure(base_dir):
    """Create a test directory structure with audio files."""
    # Create nested directories
    dirs = [
        "music/rock",
        "music/jazz",
        "podcasts/tech",
        "podcasts/history",
        "audiobooks",
    ]
    
    for dir_path in dirs:
        (base_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create test files with different timestamps
    files = [
        ("music/rock/song1.mp3", "Rock song 1", datetime.now() - timedelta(days=10)),
        ("music/rock/song2.mp3", "Rock song 2", datetime.now() - timedelta(days=5)),
        ("music/jazz/song1.mp3", "Jazz song 1 - different content!", datetime.now() - timedelta(days=3)),  # Duplicate name, different content
        ("podcasts/tech/episode1.mp3", "Tech episode 1", datetime.now() - timedelta(days=2)),
        ("podcasts/history/episode1.wav", "History episode 1", datetime.now() - timedelta(days=1)),
        ("audiobooks/book1.m4a", "Audiobook 1", datetime.now() - timedelta(hours=12)),
        ("test.flac", "Test FLAC file", datetime.now() - timedelta(hours=6)),
    ]
    
    created_files = []
    for file_path, content, timestamp in files:
        full_path = base_dir / file_path
        full_path.write_text(content)
        
        # Set custom timestamp
        timestamp_epoch = timestamp.timestamp()
        os.utime(full_path, (timestamp_epoch, timestamp_epoch))
        
        created_files.append(full_path)
    
    return created_files


def run_copy_script(args):
    """Run the copy_audio_files.py script with given arguments."""
    script_path = Path(__file__).parent.parent / "copy_audio_files.py"
    cmd = [sys.executable, str(script_path)] + args
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def test_basic_copy():
    """Test basic copy functionality with flattening."""
    print("TEST 1: Basic copy with directory flattening")
    print("-" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        
        # Create test structure
        test_files = create_test_structure(input_dir)
        
        # Run copy
        result = run_copy_script(["-i", str(input_dir), "-o", str(output_dir)])
        
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Verify all files were copied to flat structure
        output_files = list(output_dir.glob("*"))
        print(f"\nFiles in output directory: {len(output_files)}")
        for f in sorted(output_files):
            print(f"  - {f.name}")
        
        # Check that duplicate names were handled
        assert (output_dir / "song1.mp3").exists(), "First song1.mp3 should exist"
        assert (output_dir / "song1_1.mp3").exists(), "Second song1.mp3 should be renamed to song1_1.mp3"
        
        print("\n✓ Test passed: Files copied and flattened correctly")
    
    print()


def test_timestamp_preservation():
    """Test that timestamps are preserved during copy."""
    print("TEST 2: Timestamp preservation")
    print("-" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        
        # Create a test file with specific timestamp
        input_dir.mkdir(parents=True)
        test_file = input_dir / "test.mp3"
        test_file.write_text("Test content")
        
        # Set specific timestamp (1 day ago)
        old_time = datetime.now() - timedelta(days=1)
        old_timestamp = old_time.timestamp()
        os.utime(test_file, (old_timestamp, old_timestamp))
        
        # Copy the file
        result = run_copy_script(["-i", str(input_dir), "-o", str(output_dir)])
        
        # Check timestamp was preserved
        output_file = output_dir / "test.mp3"
        assert output_file.exists(), "Output file should exist"
        
        output_stat = output_file.stat()
        time_diff = abs(output_stat.st_mtime - old_timestamp)
        assert time_diff < 1, f"Timestamp not preserved (diff: {time_diff} seconds)"
        
        print(f"Original timestamp: {datetime.fromtimestamp(old_timestamp)}")
        print(f"Copied timestamp: {datetime.fromtimestamp(output_stat.st_mtime)}")
        print("\n✓ Test passed: Timestamps preserved correctly")
    
    print()


def test_skip_existing():
    """Test that existing files are skipped by default."""
    print("TEST 3: Skip existing files")
    print("-" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        
        # Create test file
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        test_file = input_dir / "test.mp3"
        test_file.write_text("Original content")
        
        # Create existing file in output
        existing_file = output_dir / "test.mp3"
        existing_file.write_text("Original content")  # Same size
        
        # Run copy (should skip)
        result = run_copy_script(["-i", str(input_dir), "-o", str(output_dir)])
        
        print("STDOUT:", result.stdout)
        assert "Skipped (already exists): test.mp3" in result.stdout
        assert "Files skipped: 1" in result.stdout
        
        print("\n✓ Test passed: Existing files skipped correctly")
    
    print()


def test_force_overwrite():
    """Test force overwrite functionality."""
    print("TEST 4: Force overwrite")
    print("-" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        
        # Create test files
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        test_file = input_dir / "test.mp3"
        test_file.write_text("New content from input")
        
        existing_file = output_dir / "test.mp3"
        existing_file.write_text("Old content")
        
        # Run copy with force
        result = run_copy_script(["-i", str(input_dir), "-o", str(output_dir), "--force"])
        
        print("STDOUT:", result.stdout)
        
        # Verify file was overwritten
        assert existing_file.read_text() == "New content from input"
        assert "Files copied: 1" in result.stdout
        
        print("\n✓ Test passed: Force overwrite works correctly")
    
    print()


def test_delete_after_copy():
    """Test delete source files after copy."""
    print("TEST 5: Delete after copy (move operation)")
    print("-" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        
        # Create test structure
        input_dir.mkdir(parents=True)
        subdir = input_dir / "subdir"
        subdir.mkdir()
        
        file1 = input_dir / "test1.mp3"
        file2 = subdir / "test2.mp3"
        
        file1.write_text("Content 1")
        file2.write_text("Content 2")
        
        # Run copy with delete
        result = run_copy_script(["-i", str(input_dir), "-o", str(output_dir), "--delete"])
        
        print("STDOUT:", result.stdout)
        
        # Verify files were moved
        assert not file1.exists(), "Source file1 should be deleted"
        assert not file2.exists(), "Source file2 should be deleted"
        assert (output_dir / "test1.mp3").exists(), "Output file1 should exist"
        assert (output_dir / "test2.mp3").exists(), "Output file2 should exist"
        assert "WARNING: Delete mode enabled" in result.stdout
        assert "Files moved: 2" in result.stdout
        
        print("\n✓ Test passed: Files moved correctly (copied then deleted)")
    
    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing copy_audio_files.py")
    print("=" * 60)
    print()
    
    tests = [
        test_basic_copy,
        test_timestamp_preservation,
        test_skip_existing,
        test_force_overwrite,
        test_delete_after_copy,
    ]
    
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ Test failed: {e}")
            return 1
        except Exception as e:
            print(f"✗ Test error: {e}")
            return 1
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())