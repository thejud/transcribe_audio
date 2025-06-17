#!/usr/bin/env python3

import os
import sys
import logging
import argparse
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from dotenv import load_dotenv

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


class VoiceMemoWatcher:
    """Watches for voice memos and processes them through the transcription pipeline."""

    def __init__(self, audio_in: Path, audio_out: Path, transcript_out: Path):
        self.audio_in = audio_in
        self.audio_out = audio_out
        self.transcript_out = transcript_out
        self.logger = logging.getLogger(__name__)

        # Create directories if they don't exist
        for dir_path in [audio_in, audio_out, transcript_out]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def find_audio_files(self) -> List[AudioFile]:
        """Find all audio files in the input directory."""
        audio_extensions = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus"}
        files = []

        for file_path in self.audio_in.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                audio_file = AudioFile(path=file_path)

                # Extract metadata for M4A files
                if file_path.suffix.lower() == ".m4a":
                    try:
                        audio = MP4(str(file_path))
                        # Get creation date from metadata
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

                        # Get duration
                        if audio.info.length:
                            audio_file.duration = audio.info.length
                    except Exception as e:
                        self.logger.warning(
                            f"Could not read metadata from {file_path}: {e}"
                        )

                # Fallback to file modification time if no metadata timestamp
                if not audio_file.timestamp:
                    stat = file_path.stat()
                    audio_file.timestamp = datetime.fromtimestamp(stat.st_mtime)

                files.append(audio_file)

        return sorted(files, key=lambda f: f.timestamp or datetime.min)

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

    def process_file(self, audio_file: AudioFile) -> bool:
        """Process a single audio file through the transcription pipeline."""
        self.logger.info(f"Processing {audio_file.path.name}")

        try:
            # Run transcription pipeline (default is memo mode)
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

            # Find the generated transcript file (could be .txt from pipeline)
            transcript_pattern = audio_file.path.stem + "*"
            transcript_files = list(
                self.transcript_out.glob(transcript_pattern + ".txt")
            )
            if not transcript_files:
                transcript_files = list(
                    self.transcript_out.glob(transcript_pattern + ".md")
                )

            if not transcript_files:
                self.logger.error(
                    f"No transcript file found matching {transcript_pattern}"
                )
                return False

            transcript_file = transcript_files[0]

            # Extract title from transcript and update content
            title = self.extract_title_from_transcript(transcript_file)

            # Replace <title> tags with markdown header if found
            content = transcript_file.read_text(encoding="utf-8")
            title_match = re.search(r"<title>(.*?)</title>", content, re.IGNORECASE)
            if title_match:
                # Replace the title tag with markdown header
                new_content = content.replace(
                    title_match.group(0), f"# {title_match.group(1)}"
                )
                transcript_file.write_text(new_content, encoding="utf-8")

            # Generate new filename with timestamp and title
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

            # Move audio file to output directory
            new_audio_path = self.audio_out / new_audio_name
            audio_file.path.rename(new_audio_path)

            # Rename transcript file
            new_transcript_path = self.transcript_out / new_transcript_name
            transcript_file.rename(new_transcript_path)

            # Preserve timestamps
            if audio_file.timestamp:
                timestamp_epoch = audio_file.timestamp.timestamp()
                os.utime(new_audio_path, (timestamp_epoch, timestamp_epoch))
                os.utime(new_transcript_path, (timestamp_epoch, timestamp_epoch))

            self.logger.info(
                f"Successfully processed: {audio_file.path.name} -> {new_audio_name}"
            )

            # Log full paths for easy access
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

    def run(self) -> Dict[str, Any]:
        """Run the watcher and process all files."""
        files = self.find_audio_files()

        if not files:
            self.logger.info("No audio files found to process")
            return {"processed": 0, "failed": 0, "total": 0}

        self.logger.info(f"Found {len(files)} audio files to process")

        processed = 0
        failed = 0

        for audio_file in files:
            if self.process_file(audio_file):
                processed += 1
            else:
                failed += 1

        return {"processed": processed, "failed": failed, "total": len(files)}


def main():
    parser = argparse.ArgumentParser(
        description="Watch for voice memos and process them through transcription pipeline"
    )
    parser.add_argument(
        "--audio-in",
        type=Path,
        help="Input directory for audio files (default: from .env)",
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

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load environment variables
    load_dotenv()

    # Get directories from args or environment
    audio_in = args.audio_in or Path(
        os.path.expanduser(
            os.getenv("AUDIO_IN", "~/Dropbox/01-projects/voice_memo_inbox")
        )
    )
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

    # Create watcher and run
    watcher = VoiceMemoWatcher(audio_in, audio_out, transcript_out)
    results = watcher.run()

    # Print summary
    print(f"\nProcessing complete:")
    print(f"  Total files: {results['total']}")
    print(f"  Processed: {results['processed']}")
    print(f"  Failed: {results['failed']}")

    # Exit with error if any files failed
    sys.exit(1 if results["failed"] > 0 else 0)


if __name__ == "__main__":
    main()
