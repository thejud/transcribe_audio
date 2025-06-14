#!/usr/bin/env python3
"""
Print audio/*.txt files in numerical order based on their final numeric segment.
"""

import re
from pathlib import Path
from typing import List


def extract_final_number(filepath: Path) -> int:
    """Extract the final numeric segment from a filename."""
    # Match the pattern: digits_number.txt
    match = re.search(r"_(\d+)\.txt$", filepath.name)
    if match:
        return int(match.group(1))
    return 0


def get_sorted_files() -> List[Path]:
    """Get all audio/*.txt files sorted by their final numeric segment."""
    audio_dir = Path("audio")
    files = list(audio_dir.glob("*.txt"))
    return sorted(files, key=extract_final_number)


def print_file_contents(filepath: Path) -> None:
    """Print a file with the requested format."""
    print(f"### {filepath.name}")
    print()
    content = filepath.read_text(encoding="utf-8").strip()
    print(content)
    print()


def main() -> None:
    """Main function to process and print all files."""
    files = get_sorted_files()

    for filepath in files:
        print_file_contents(filepath)


if __name__ == "__main__":
    main()
