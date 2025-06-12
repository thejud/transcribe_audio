#!/bin/bash

set -e

if [ ! -f .env ]; then
    echo "Error: .env file not found"
    exit 1
fi

source .env

if [ -z "$SOURCE_AUDIO" ]; then
    echo "Error: SOURCE_AUDIO not set in .env file"
    exit 1
fi

if [ ! -d "$SOURCE_AUDIO" ]; then
    echo "Error: Source directory $SOURCE_AUDIO does not exist"
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is required but not installed"
    echo "Install with: brew install ffmpeg"
    exit 1
fi

mkdir -p audio

echo "Converting AMR files from $SOURCE_AUDIO to MP3..."

count=0
for amr_file in "$SOURCE_AUDIO"/*.amr; do
    if [ -f "$amr_file" ]; then
        filename=$(basename "$amr_file" .amr)
        output_file="audio/${filename}.mp3"
        
        echo "Converting: $filename.amr -> $filename.mp3"
        
        if ffmpeg -i "$amr_file" -acodec libmp3lame -ab 128k "$output_file" -y -loglevel error; then
            count=$((count + 1))
        else
            echo "Warning: Failed to convert $amr_file"
        fi
    fi
done

if [ $count -eq 0 ]; then
    echo "No AMR files found in $SOURCE_AUDIO"
else
    echo "Successfully converted $count AMR files to MP3 format in ./audio/"
fi