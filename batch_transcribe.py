#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path

# Directory containing mp3 files
mp3_dir = Path("/path/to/audio/files")

# Read the prompt
with open("private/cs_prompt.txt") as f:
    prompt = f.read().strip()

# Find mp3 files without corresponding txt files
for mp3_file in mp3_dir.glob("*.mp3"):
    txt_file = mp3_file.with_suffix(".txt")

    if not txt_file.exists():
        print(f"Transcribing: {mp3_file.name}")

        # Run transcription
        cmd = ["python3", "transcribe.py", str(mp3_file), "--prompt", prompt]

        subprocess.run(cmd)
        print(f"Completed: {mp3_file.name}")
