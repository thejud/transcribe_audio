#!/usr/bin/env python3

import os
import sys
import logging
import argparse
import subprocess
import json
import re
import time
import signal
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from dotenv import load_dotenv

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    WATCHDOG_AVAILABLE = True
except ImportError:
    Observer = None
    FileSystemEventHandler = None
    WATCHDOG_AVAILABLE = False

try:
    from mutagen.mp4 import MP4
except ImportError:
    print("Error: mutagen is required. Install with: uv pip install mutagen")
    sys.exit(1)


@dataclass
class AudioFile:
    """Represents an audio file with metadata."""

    path: Path
    timestamp: Optional[datetime] = None
    duration: Optional[float] = None


class AudioFileHandler(FileSystemEventHandler):
    """File system event handler for audio files."""

    def __init__(self, watcher: "VoiceMemoWatcher"):
        self.watcher = watcher
        self.logger = logging.getLogger(__name__)
        self.audio_extensions = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus"}
        self.processed_files = set()  # Track recently processed files

    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            # Handle directory creation (e.g., USB drive mounted)
            self._handle_directory_event(Path(event.src_path))
        else:
            self._handle_file_event(Path(event.src_path), "created")

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            # Handle directory modification (e.g., new files added)
            self._handle_directory_event(Path(event.src_path))
        else:
            self._handle_file_event(Path(event.src_path), "modified")

    def on_moved(self, event):
        """Handle file move events."""
        if not event.is_directory:
            self._handle_file_event(Path(event.dest_path), "moved")

    def _handle_directory_event(self, dir_path: Path):
        """Handle directory events (e.g., USB drive mounted)."""
        self.logger.info(f"Directory event detected: {dir_path}")

        # Check if this is one of our watched directories or a parent of one
        relevant_dir = None
        for watched_dir in self.watcher.audio_in_dirs:
            # Check if the event is for the watched directory itself
            if dir_path == watched_dir:
                relevant_dir = watched_dir
                break
            # Check if the event is for a parent directory (e.g., /Volumes/VoiceBox when we watch /Volumes/VoiceBox/A)
            try:
                if watched_dir.is_relative_to(dir_path):
                    # The watched directory is a subdirectory of the event directory
                    # Check if the watched directory now exists
                    if watched_dir.exists():
                        relevant_dir = watched_dir
                        break
            except (ValueError, AttributeError):
                # is_relative_to might not be available in older Python versions
                if str(watched_dir).startswith(str(dir_path)):
                    if watched_dir.exists():
                        relevant_dir = watched_dir
                        break

        if not relevant_dir:
            self.logger.debug(
                f"Directory {dir_path} is not relevant to watched directories, ignoring"
            )
            return

        # Scan the relevant directory for audio files
        self.logger.info(f"Scanning directory for audio files: {relevant_dir}")
        try:
            audio_files = self.watcher.find_audio_files_in_directory(relevant_dir)

            # Filter out already processed files
            unprocessed_files = []
            for audio_file in audio_files:
                file_key = str(audio_file.path)
                if file_key not in self.processed_files:
                    unprocessed_files.append(audio_file)
                    # Mark as processed to avoid duplicate processing
                    self.processed_files.add(file_key)

            if unprocessed_files:
                self.logger.info(
                    f"Found {len(unprocessed_files)} unprocessed audio files in {relevant_dir}"
                )
                for audio_file in unprocessed_files:
                    # Process the file
                    success = self.watcher.process_file(audio_file)
                    if success:
                        self.logger.info(
                            f"Successfully processed: {audio_file.path.name}"
                        )
                    else:
                        self.logger.error(f"Failed to process: {audio_file.path.name}")
                        # Remove from processed set so it can be retried
                        self.processed_files.discard(str(audio_file.path))
            else:
                self.logger.info(f"No unprocessed audio files found in {relevant_dir}")

        except Exception as e:
            self.logger.error(f"Error scanning directory {relevant_dir}: {e}")

    def _handle_file_event(self, file_path: Path, event_type: str):
        """Process a file system event for a potential audio file."""
        # Check if it's an audio file
        if file_path.suffix.lower() not in self.audio_extensions:
            return

        # Avoid processing the same file multiple times in quick succession
        file_key = str(file_path)
        if file_key in self.processed_files:
            self.logger.debug(f"Skipping recently processed file: {file_path}")
            return

        self.logger.info(f"Audio file {event_type}: {file_path}")

        # Small delay to ensure file is completely written
        time.sleep(1)

        if file_path.exists() and file_path.is_file():
            try:
                # Create AudioFile object
                audio_file = AudioFile(path=file_path)

                # Extract metadata (same logic as find_audio_files)
                if file_path.suffix.lower() == ".m4a":
                    try:
                        audio = MP4(str(file_path))
                        if "©day" in audio:
                            date_str = audio["©day"][0]
                            try:
                                audio_file.timestamp = datetime.fromisoformat(
                                    date_str.replace("Z", "+00:00")
                                )
                            except:
                                self.logger.warning(
                                    f"Could not parse date from metadata: {date_str}"
                                )

                        if audio.info.length:
                            audio_file.duration = audio.info.length
                    except Exception as e:
                        self.logger.warning(
                            f"Could not read metadata from {file_path}: {e}"
                        )

                # Fallback to file modification time
                if not audio_file.timestamp:
                    stat = file_path.stat()
                    audio_file.timestamp = datetime.fromtimestamp(stat.st_mtime)

                # Mark as processed
                self.processed_files.add(file_key)

                # Process the file
                success = self.watcher.process_file(audio_file)

                if success:
                    self.logger.info(
                        f"Successfully processed {event_type} file: {file_path}"
                    )
                else:
                    self.logger.error(
                        f"Failed to process {event_type} file: {file_path}"
                    )
                    # Remove from processed set so it can be retried
                    self.processed_files.discard(file_key)

            except Exception as e:
                self.logger.error(f"Error handling {event_type} file {file_path}: {e}")
                self.processed_files.discard(file_key)

        # Clean up old entries from processed_files set to prevent memory growth
        if len(self.processed_files) > 1000:
            # Keep only the most recent 500 entries
            recent_files = list(self.processed_files)[-500:]
            self.processed_files = set(recent_files)


class VoiceMemoWatcher:
    """Watches for voice memos and processes them through the transcription pipeline.

    Supports multiple input directories for USB voice memo devices with A, B, C, D folders.
    Environment variable format: AUDIO_IN=/path/A:/path/B:/path/C:/path/D
    Command line format: --audio-in /path/A --audio-in /path/B
    """

    def __init__(
        self, audio_in_dirs: List[Path], audio_out: Path, transcript_out: Path
    ):
        self.audio_in_dirs = (
            audio_in_dirs if isinstance(audio_in_dirs, list) else [audio_in_dirs]
        )
        self.audio_out = audio_out
        self.transcript_out = transcript_out
        self.logger = logging.getLogger(__name__)
        self.running = True

        # Create output directories if they don't exist (but not input dirs - USB devices should exist when plugged in)
        for dir_path in [audio_out, transcript_out]:
            dir_path.mkdir(parents=True, exist_ok=True)

    # ========== File Discovery Methods ==========

    def _extract_audio_metadata(
        self, file_path: Path
    ) -> tuple[Optional[datetime], Optional[float]]:
        """Extract timestamp and duration metadata from audio files."""
        timestamp = None
        duration = None

        # Extract metadata for M4A files
        if file_path.suffix.lower() == ".m4a":
            try:
                audio = MP4(str(file_path))
                # Get creation date from metadata
                if "©day" in audio:
                    date_str = audio["©day"][0]
                    try:
                        timestamp = datetime.fromisoformat(
                            date_str.replace("Z", "+00:00")
                        )
                    except:
                        self.logger.warning(
                            f"Could not parse date from metadata: {date_str}"
                        )

                # Get duration
                if audio.info.length:
                    duration = audio.info.length
            except Exception as e:
                self.logger.warning(f"Could not read metadata from {file_path}: {e}")

        # Fallback to file modification time if no metadata timestamp
        if not timestamp:
            stat = file_path.stat()
            timestamp = datetime.fromtimestamp(stat.st_mtime)

        return timestamp, duration

    def _create_audio_file_with_metadata(self, file_path: Path) -> AudioFile:
        """Create an AudioFile object with extracted metadata."""
        timestamp, duration = self._extract_audio_metadata(file_path)
        return AudioFile(path=file_path, timestamp=timestamp, duration=duration)

    def find_audio_files_in_directory(self, directory: Path) -> List[AudioFile]:
        """Find all audio files in a specific directory."""
        audio_extensions = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus"}
        files = []

        if not directory.exists():
            return files

        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                audio_file = self._create_audio_file_with_metadata(file_path)
                files.append(audio_file)

        return sorted(files, key=lambda f: f.timestamp or datetime.min)

    def find_audio_files(self) -> List[AudioFile]:
        """Find all audio files in all input directories."""
        files = []

        for audio_in_dir in self.audio_in_dirs:
            if not audio_in_dir.exists():
                self.logger.warning(f"Input directory does not exist: {audio_in_dir}")
                continue

            dir_files = self.find_audio_files_in_directory(audio_in_dir)
            files.extend(dir_files)

        return sorted(files, key=lambda f: f.timestamp or datetime.min)

    # ========== Text Processing Methods ==========

    def extract_title_from_transcript(self, transcript_path: Path) -> Optional[str]:
        """Extract a title from the transcript content."""
        try:
            content = transcript_path.read_text(encoding="utf-8")

            # First, try to find a <title> tag
            title_match = re.search(r"<title>(.*?)</title>", content, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()
                if title:
                    return self.sanitize_filename(title)

            # Otherwise, try to find a suitable title from the first few lines
            lines = content.strip().split("\n")
            for line in lines[:5]:  # Check first 5 lines
                line = line.strip()
                # Skip lines that look like tags
                if line.startswith("<") and line.endswith(">"):
                    continue
                if len(line) > 10 and len(line) < 100:  # Reasonable title length
                    # Clean up the line to make it filename-safe
                    return self.sanitize_filename(line)

            # Fallback: use first 50 chars of content
            if content:
                # Skip past any initial tags
                content_start = content
                if content.startswith("<"):
                    tag_end = content.find(">")
                    if tag_end != -1 and tag_end < 50:
                        content_start = content[tag_end + 1 :].strip()
                return self.sanitize_filename(content_start[:50])
        except Exception as e:
            self.logger.warning(f"Could not extract title from {transcript_path}: {e}")

        return None

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize a string to be safe for use as a filename."""
        # Remove or replace unsafe characters (but keep spaces)
        unsafe_chars = r'[<>:"/\\|?*\x00-\x1f]'
        filename = re.sub(unsafe_chars, "_", filename)

        # Replace multiple spaces with single space
        filename = re.sub(r"\s+", " ", filename)

        # Remove leading/trailing whitespace and dots
        filename = filename.strip(" .")

        # Limit length (leave room for timestamp and extension)
        max_length = 100
        if len(filename) > max_length:
            filename = filename[:max_length].rsplit(" ", 1)[0]

        return filename or "untitled"

    # ========== Core Processing Methods ==========

    def _run_transcription_pipeline(self, audio_file: AudioFile) -> bool:
        """Run the transcription pipeline on an audio file."""
        cmd = [
            sys.executable,
            "transcribe_pipeline.py",
            str(audio_file.path),
            "-O",
            str(self.transcript_out),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.logger.error(f"Transcription failed: {result.stderr}")
            return False

        return True

    def _find_transcript_file(self, audio_file: AudioFile) -> Optional[Path]:
        """Find the generated transcript file for an audio file."""
        transcript_pattern = audio_file.path.stem + "*"
        transcript_files = list(self.transcript_out.glob(transcript_pattern + ".txt"))
        if not transcript_files:
            transcript_files = list(
                self.transcript_out.glob(transcript_pattern + ".md")
            )

        if not transcript_files:
            self.logger.error(f"No transcript file found matching {transcript_pattern}")
            return None

        return transcript_files[0]

    def _quarantine_file(self, audio_file: AudioFile, transcript_file: Path) -> None:
        """Move a problematic audio file to quarantine and clean up transcript."""
        # Create quarantine directory if it doesn't exist
        quarantine_dir = self.audio_out.parent / "quarantine"
        quarantine_dir.mkdir(exist_ok=True)

        # Move the problematic audio file to quarantine
        quarantine_audio_path = quarantine_dir / audio_file.path.name
        shutil.move(str(audio_file.path), str(quarantine_audio_path))

        # Clean up the empty transcript
        transcript_file.unlink()

        self.logger.warning(
            f"Moved problematic file to quarantine: {quarantine_audio_path}"
        )

    def _validate_and_process_transcript(
        self, transcript_file: Path, audio_file: AudioFile
    ) -> tuple[Optional[str], Optional[str]]:
        """Validate transcript content and extract title. Returns (content, title) or (None, None) if invalid."""
        # Read and validate transcript content
        content = transcript_file.read_text(encoding="utf-8").strip()

        # Check if transcript is empty or too short
        if not content or len(content) < 10:
            self.logger.error(
                f"Transcript is empty or too short for {audio_file.path.name}"
            )
            self.logger.debug(f"Transcript content length: {len(content)}")
            self._quarantine_file(audio_file, transcript_file)
            return None, None

        # Extract title from transcript
        title = self.extract_title_from_transcript(transcript_file)

        # Replace <title> tags with markdown header if found
        title_match = re.search(r"<title>(.*?)</title>", content, re.IGNORECASE)
        if title_match:
            # Replace the title tag with markdown header
            new_content = content.replace(
                title_match.group(0), f"# {title_match.group(1)}"
            )
            transcript_file.write_text(new_content, encoding="utf-8")

        return content, title

    def _generate_output_filenames(
        self, audio_file: AudioFile, title: Optional[str]
    ) -> tuple[str, str]:
        """Generate output filenames for audio and transcript files."""
        timestamp_str = (
            audio_file.timestamp.strftime("%Y%m%d_%H%M%S")
            if audio_file.timestamp
            else "unknown"
        )

        if title:
            new_audio_name = f"{timestamp_str}_{title}{audio_file.path.suffix}"
            new_transcript_name = f"{timestamp_str}_{title}.md"
        else:
            new_audio_name = (
                f"{timestamp_str}_{audio_file.path.stem}{audio_file.path.suffix}"
            )
            new_transcript_name = f"{timestamp_str}_{audio_file.path.stem}.md"

        return new_audio_name, new_transcript_name

    def _move_and_preserve_files(
        self,
        audio_file: AudioFile,
        transcript_file: Path,
        new_audio_name: str,
        new_transcript_name: str,
    ) -> tuple[Path, Path]:
        """Move files to output directories and preserve timestamps."""
        # Move audio file to output directory (handles cross-filesystem moves)
        new_audio_path = self.audio_out / new_audio_name
        shutil.move(str(audio_file.path), str(new_audio_path))

        # Move transcript file (handles cross-filesystem moves)
        new_transcript_path = self.transcript_out / new_transcript_name
        shutil.move(str(transcript_file), str(new_transcript_path))

        # Preserve timestamps
        if audio_file.timestamp:
            timestamp_epoch = audio_file.timestamp.timestamp()
            os.utime(new_audio_path, (timestamp_epoch, timestamp_epoch))
            os.utime(new_transcript_path, (timestamp_epoch, timestamp_epoch))

        return new_audio_path, new_transcript_path

    def process_file(self, audio_file: AudioFile) -> bool:
        """Process a single audio file through the transcription pipeline."""
        self.logger.info(f"Processing {audio_file.path.name}")

        try:
            # Run transcription pipeline
            if not self._run_transcription_pipeline(audio_file):
                return False

            # Find the generated transcript file
            transcript_file = self._find_transcript_file(audio_file)
            if not transcript_file:
                return False

            # Validate transcript and extract title
            content, title = self._validate_and_process_transcript(
                transcript_file, audio_file
            )
            if content is None:  # File was quarantined
                return False

            # Generate output filenames
            new_audio_name, new_transcript_name = self._generate_output_filenames(
                audio_file, title
            )

            # Move files and preserve timestamps
            new_audio_path, new_transcript_path = self._move_and_preserve_files(
                audio_file, transcript_file, new_audio_name, new_transcript_name
            )

            # Log success
            self.logger.info(
                f"Successfully processed: {audio_file.path.name} -> {new_audio_name}"
            )
            self.logger.info(f"Audio output: {new_audio_path.absolute()}")
            self.logger.info(f"Transcript output: {new_transcript_path.absolute()}")

            # Also print to stdout for immediate visibility
            print(f"\nCreated files:")
            print(f"  Audio: {new_audio_path.absolute()}")
            print(f"  Transcript: {new_transcript_path.absolute()}")

            return True

        except Exception as e:
            self.logger.error(f"Error processing {audio_file.path}: {e}")
            return False

    # ========== Run Mode Methods ==========

    def run(self) -> Dict[str, Any]:
        """Run the watcher and process all files."""
        files = self.find_audio_files()

        if not files:
            self.logger.info(
                f"No audio files found to process in directories: {[str(d) for d in self.audio_in_dirs]}"
            )
            return {"processed": 0, "failed": 0, "total": 0}

        self.logger.info(
            f"Found {len(files)} audio files to process across {len(self.audio_in_dirs)} directories"
        )

        processed = 0
        failed = 0

        for audio_file in files:
            if self.process_file(audio_file):
                processed += 1
            else:
                failed += 1

        return {"processed": processed, "failed": failed, "total": len(files)}

    def stop(self):
        """Stop the watcher."""
        self.running = False
        self.logger.info("Stopping voice memo watcher...")

    # ========== Monitoring Helper Methods ==========

    def run_watch_mode(self, interval: int = 60) -> None:
        """Run in watch mode, checking for new files every interval seconds."""
        self.logger.info(f"Starting watch mode - checking every {interval} seconds")
        self.logger.info(
            f"Watching directories: {[str(d.absolute()) for d in self.audio_in_dirs]}"
        )
        self.logger.info("Press Ctrl+C to stop")

        # Set up signal handler for graceful shutdown
        def signal_handler(signum, frame):
            self.logger.info("Received interrupt signal")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        total_processed = 0
        total_failed = 0
        iterations = 0

        try:
            while self.running:
                iterations += 1
                self.logger.debug(
                    f"Watch iteration {iterations} - checking for files..."
                )

                # Run one iteration
                results = self.run()

                # Update totals
                total_processed += results["processed"]
                total_failed += results["failed"]

                # Log results if files were found
                if results["total"] > 0:
                    self.logger.info(
                        f"Batch {iterations}: {results['processed']} processed, {results['failed']} failed"
                    )
                else:
                    self.logger.debug(f"No files found in iteration {iterations}")

                # Wait for next check (unless stopping)
                if self.running:
                    self.logger.debug(f"Waiting {interval} seconds until next check...")

                    # Sleep in smaller increments to allow for responsive shutdown
                    sleep_increment = min(5, interval)
                    elapsed = 0
                    while elapsed < interval and self.running:
                        time.sleep(sleep_increment)
                        elapsed += sleep_increment

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
            self.stop()
        except Exception as e:
            self.logger.error(f"Unexpected error in watch mode: {e}")
            self.stop()
        finally:
            # Print final summary
            self.logger.info("Watch mode stopped")
            print(f"\nFinal summary:")
            print(f"  Total iterations: {iterations}")
            print(f"  Total files processed: {total_processed}")
            print(f"  Total failures: {total_failed}")

            if total_failed > 0:
                sys.exit(1)

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            self.logger.info("Received interrupt signal")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _setup_file_watchers(self, event_handler, observer) -> list:
        """Set up file system watchers for all input directories."""
        watches = []
        for audio_in_dir in self.audio_in_dirs:
            if audio_in_dir.exists():
                try:
                    watch = observer.schedule(
                        event_handler, str(audio_in_dir), recursive=False
                    )
                    watches.append((watch, audio_in_dir))
                    self.logger.info(f"Watching directory: {audio_in_dir}")
                except Exception as e:
                    self.logger.error(f"Failed to watch directory {audio_in_dir}: {e}")
            else:
                self.logger.warning(
                    f"Directory does not exist (will retry if it appears): {audio_in_dir}"
                )
        return watches

    def _scan_for_new_usb_devices(self, observer, event_handler, watches) -> None:
        """Scan for newly available USB devices and add them to watches."""
        for audio_in_dir in self.audio_in_dirs:
            if audio_in_dir.exists():
                # Check if we're already watching this directory
                already_watching = any(
                    audio_in_dir == watched_dir for _, watched_dir in watches
                )
                if not already_watching:
                    try:
                        watch = observer.schedule(
                            event_handler, str(audio_in_dir), recursive=False
                        )
                        watches.append((watch, audio_in_dir))
                        self.logger.info(
                            f"Started watching newly available directory: {audio_in_dir}"
                        )

                        # Scan for existing files in the newly available directory
                        self.logger.info(
                            f"Scanning newly available directory for existing files: {audio_in_dir}"
                        )
                        existing_files = self.find_audio_files_in_directory(
                            audio_in_dir
                        )
                        if existing_files:
                            self.logger.info(
                                f"Found {len(existing_files)} existing files in {audio_in_dir}"
                            )
                            for audio_file in existing_files:
                                if self.process_file(audio_file):
                                    self.logger.info(
                                        f"Processed existing file: {audio_file.path.name}"
                                    )
                                else:
                                    self.logger.error(
                                        f"Failed to process existing file: {audio_file.path.name}"
                                    )
                        else:
                            self.logger.info(
                                f"No existing files found in {audio_in_dir}"
                            )
                    except Exception as e:
                        self.logger.error(
                            f"Failed to watch directory {audio_in_dir}: {e}"
                        )

    def _monitor_loop(self, observer, event_handler, watches, initial_results) -> None:
        """Main monitoring loop that handles USB device detection."""
        try:
            while self.running:
                time.sleep(1)

                # Periodically check for new directories (e.g., USB device mounted)
                # This happens every 30 seconds
                if int(time.time()) % 30 == 0:
                    self._scan_for_new_usb_devices(observer, event_handler, watches)

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
            self.stop()
        except Exception as e:
            self.logger.error(f"Unexpected error in monitor mode: {e}")
            self.stop()

    def _cleanup_monitor(self, observer, initial_results) -> None:
        """Clean up observer and print final summary."""
        # Stop the observer
        observer.stop()
        observer.join()

        # Print final summary
        self.logger.info("FSEvents monitoring stopped")
        print(f"\nFinal summary:")
        print(f"  Initial files processed: {initial_results['processed']}")
        print(f"  Initial failures: {initial_results['failed']}")
        print("  Real-time processing: enabled")
        print("  Note: Real-time processed files are logged individually above")

    def run_monitor_mode(self) -> None:
        """Run in FSEvents monitor mode, processing files as they appear."""
        if not WATCHDOG_AVAILABLE:
            self.logger.error(
                "FSEvents monitoring requires the 'watchdog' library. Install with: uv add watchdog"
            )
            self.logger.info("Falling back to watch mode...")
            self.run_watch_mode(60)
            return

        self.logger.info("Starting FSEvents monitor mode")
        self.logger.info(
            f"Monitoring directories: {[str(d.absolute()) for d in self.audio_in_dirs]}"
        )
        self.logger.info("Files will be processed immediately when detected")
        self.logger.info("Press Ctrl+C to stop")

        # Set up signal handlers
        self._setup_signal_handlers()

        # Create event handler and observer
        event_handler = AudioFileHandler(self)
        observer = Observer()

        # Set up file watchers
        watches = self._setup_file_watchers(event_handler, observer)
        if not watches:
            self.logger.error("No directories could be watched. Exiting.")
            return

        # Process any existing files first
        self.logger.info("Processing any existing files...")
        initial_results = self.run()
        if initial_results["total"] > 0:
            self.logger.info(
                f"Initial scan: {initial_results['processed']} processed, {initial_results['failed']} failed"
            )

        # Start the observer
        observer.start()
        self.logger.info("FSEvents monitoring started. Waiting for new files...")

        # Run the monitoring loop
        self._monitor_loop(observer, event_handler, watches, initial_results)

        # Clean up
        self._cleanup_monitor(observer, initial_results)


# ========== Main Function and Helpers ==========


def _parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Watch for voice memos and process them through transcription pipeline"
    )
    parser.add_argument(
        "--audio-in",
        type=Path,
        action="append",
        dest="audio_in_dirs",
        help="Input directory for audio files (can be specified multiple times, default: from .env)",
    )
    parser.add_argument(
        "--audio-out",
        type=Path,
        help="Output directory for processed audio files (default: from .env)",
    )
    parser.add_argument(
        "--transcript-out",
        type=Path,
        help="Output directory for transcripts (default: from .env)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--watch",
        "-W",
        action="store_true",
        help="Run in watch mode (continuously monitor for new files)",
    )
    parser.add_argument(
        "--monitor",
        "-M",
        action="store_true",
        help="Run in FSEvents monitor mode (real-time file detection, requires watchdog)",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=60,
        help="Check interval in seconds for watch mode (default: 60)",
    )
    return parser.parse_args()


def _setup_logging(verbose: bool):
    """Configure logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _get_directories_from_config(args):
    """Get directory paths from arguments or environment variables."""
    # Load environment variables
    load_dotenv()

    # Get input directories
    if args.audio_in_dirs:
        # Use directories from command line arguments
        audio_in_dirs = [
            Path(os.path.expanduser(str(path))) for path in args.audio_in_dirs
        ]
    else:
        # Parse from environment variable (supports colon-separated paths)
        env_audio_in = os.getenv("AUDIO_IN", "~/Dropbox/01-projects/voice_memo_inbox")
        if ":" in env_audio_in:
            # Split colon-separated paths
            audio_in_dirs = [
                Path(os.path.expanduser(path.strip()))
                for path in env_audio_in.split(":")
            ]
        else:
            # Single directory
            audio_in_dirs = [Path(os.path.expanduser(env_audio_in))]

    # Get output directories
    audio_out = args.audio_out or Path(
        os.path.expanduser(
            os.getenv("AUDIO_OUT", "~/Dropbox/01-projects/voice_memo_outbox")
        )
    )
    transcript_out = args.transcript_out or Path(
        os.path.expanduser(
            os.getenv("TRANSCRIPT_OUT", "~/Dropbox/01-projects/voice_memo_transcripts")
        )
    )

    return audio_in_dirs, audio_out, transcript_out


def _validate_arguments(args):
    """Validate command line arguments."""
    if args.watch and args.monitor:
        print("Error: --watch and --monitor are mutually exclusive. Choose one.")
        sys.exit(1)


def _run_watcher(args, watcher):
    """Run the watcher in the appropriate mode."""
    if args.monitor:
        # Run in FSEvents monitor mode
        watcher.run_monitor_mode()
    elif args.watch:
        # Run in watch mode
        watcher.run_watch_mode(args.interval)
    else:
        # Run once
        results = watcher.run()

        # Print summary
        print(f"\nProcessing complete:")
        print(f"  Total files: {results['total']}")
        print(f"  Processed: {results['processed']}")
        print(f"  Failed: {results['failed']}")

        # Exit with error if any files failed
        sys.exit(1 if results["failed"] > 0 else 0)


def main():
    """Main function that orchestrates the voice memo watching process."""
    # Parse arguments and setup
    args = _parse_arguments()
    _setup_logging(args.verbose)
    _validate_arguments(args)

    # Get directory configuration
    audio_in_dirs, audio_out, transcript_out = _get_directories_from_config(args)

    # Create and run watcher
    watcher = VoiceMemoWatcher(audio_in_dirs, audio_out, transcript_out)
    _run_watcher(args, watcher)


if __name__ == "__main__":
    main()
