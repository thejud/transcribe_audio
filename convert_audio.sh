#!/bin/bash

# Audio to MP3 Converter
# 
# Converts various audio formats to MP3 using ffmpeg. Supports AMR, WAV, M4A, 
# FLAC, OGG, and other formats supported by ffmpeg. Can process files from
# a source directory (defined in .env) or from command line arguments.
#
# Usage:
#   ./convert_audio.sh                         # Convert all audio files from SOURCE_AUDIO
#   ./convert_audio.sh file1.amr file2.wav     # Convert specific files
#   ./convert_audio.sh *.amr *.wav *.m4a       # Convert all audio files by extension
#
# Requirements:
#   - ffmpeg installed and available in PATH
#   - .env file with SOURCE_AUDIO (optional, for directory mode)
#
# Supported formats: AMR, WAV, M4A, FLAC, OGG, AAC, WMA, and others
#
# Author: Jud Dagnall (with Claude Code)

set -e

# Check if ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is required but not installed"
    echo "Install with: brew install ffmpeg"
    exit 1
fi

# Create output directory
mkdir -p audio

# Function to convert a single audio file to MP3
convert_file() {
    local input_file="$1"
    local filename=$(basename "$input_file")
    local name_only="${filename%.*}"
    local output_file="audio/${name_only}.mp3"
    
    echo "Converting: $(basename "$input_file") -> ${name_only}.mp3"
    
    if ffmpeg -i "$input_file" -acodec libmp3lame -ab 128k "$output_file" -y -loglevel error; then
        return 0
    else
        echo "Warning: Failed to convert $input_file"
        return 1
    fi
}

count=0

# Check if command line arguments are provided
if [ $# -gt 0 ]; then
    # Process files from command line arguments
    echo "Converting specified audio files to MP3..."
    
    for audio_file in "$@"; do
        if [ -f "$audio_file" ]; then
            if convert_file "$audio_file"; then
                count=$((count + 1))
            fi
        else
            echo "Warning: File not found: $audio_file"
        fi
    done
else
    # Process files from SOURCE_AUDIO directory (legacy mode)
    if [ ! -f .env ]; then
        echo "Error: No files specified and .env file not found"
        echo "Usage: $0 [audio_file1 audio_file2 ...]"
        exit 1
    fi

    source .env

    if [ -z "$SOURCE_AUDIO" ]; then
        echo "Error: No files specified and SOURCE_AUDIO not set in .env file"
        echo "Usage: $0 [audio_file1 audio_file2 ...]"
        exit 1
    fi

    if [ ! -d "$SOURCE_AUDIO" ]; then
        echo "Error: Source directory $SOURCE_AUDIO does not exist"
        exit 1
    fi

    echo "Converting audio files from $SOURCE_AUDIO to MP3..."

    # Process common audio formats
    for ext in amr wav m4a flac ogg aac wma mp3; do
        for audio_file in "$SOURCE_AUDIO"/*.$ext; do
            if [ -f "$audio_file" ]; then
                # Skip if it's already an MP3 with the same name in output
                local name_only=$(basename "$audio_file" .$ext)
                local output_file="audio/${name_only}.mp3"
                if [ "$ext" = "mp3" ] && [ -f "$output_file" ]; then
                    echo "Skipping: $(basename "$audio_file") (already exists as MP3)"
                    continue
                fi
                
                if convert_file "$audio_file"; then
                    count=$((count + 1))
                fi
            fi
        done
    done
fi

if [ $count -eq 0 ]; then
    echo "No audio files were converted"
else
    echo "Successfully converted $count audio files to MP3 format in ./audio/"
fi