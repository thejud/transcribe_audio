# Audio Transcription Tool

A simple Python tool for transcribing audio files using OpenAI's transcription models with intelligent chunking, context-aware prompts, and multiple output formats.

## Motivation

I created this tool to solve a particular problem I had: I had accumulated a
large collection of voicemails on my phone that I wanted to preserve and make
searchable. Using my Visual Voicemail app, I saved each voicemail to Dropbox
via the "share with" feature. However, these files were in AMR format - a
compressed audio format that's not widely supported.

The solution required two key steps:
1. **Format conversion**: Convert AMR files to the more universal MP3 format
2. **Transcription**: Generate searchable text from the audio content

Along the way, I generalized the tool to support multiple audio formats and use cases beyond voicemails. My ultimate goal is to use this system for capturing and organizing audio notes when exercising, driving, or in other situations where text input isn't practical.

There are various apps that provide transcription as part of their capture process.
However, I wanted to have a tool that could work on nearly any sort of audio
input in an app-independent way. 

The openAI APIs are very cost effective, and I wanted to avoid yet another monthly fee for a transcription app.

## Use Cases

- **Voicemail preservation**: Convert and transcribe saved voicemails for archival and search
- **Audio notes**: Transcribe voice memos recorded while exercising or driving  
- **Meeting recordings**: Process recorded conversations with context-aware prompts
- **Podcast/interview processing**: Batch transcribe audio content with timestamps
- **Family audio archives**: Preserve and transcribe personal audio recordings

## Features

- **Smart Audio Chunking**: Automatically splits audio at natural breaks (silence detection)
- **Multiple AI Models**: Support for Whisper-1, GPT-4o-transcribe, and GPT-4o-mini-transcribe
- **Flexible Output**: Save to files or output directly to stdout (text or JSON)
- **Context-Aware**: Use prompts to improve accuracy for names, places, and domain-specific terminology
- **Cost-Effective**: Choose from different models based on accuracy needs and budget
- **Batch Processing**: Process multiple audio files in a single command
- **Safe Operations**: Prevents accidental overwrites unless explicitly forced

## Installation

### Prerequisites

- Python 3.12+
- OpenAI API key
- ffmpeg (for audio processing)

### Setup

1. **Clone or download this repository**

2. **Install dependencies using uv**:
   ```bash
   uv sync
   ```

3. **Install ffmpeg** (if not already installed):
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

4. **Configure environment variables**:
   Copy the sample environment file and customize it:
   ```bash
   cp .env.sample .env
   ```
   
   Then edit `.env` with your specific settings:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   SOURCE_AUDIO=/path/to/your/source/audio/files
   DEFAULT_PROMPT=Names: Jud (not Judge), [other family names]. Places: [your locations]. Context: family voicemails, appointments, scheduling.
   ```

## Usage

### Basic Transcription

```bash
# Transcribe a single file (uses GPT-4o-transcribe by default)
uv run python transcribe.py audio/voicemail.mp3

# Transcribe multiple files
uv run python transcribe.py audio/*.mp3
```

### Model Selection

```bash
# Use GPT-4o-mini (most cost-effective)
uv run python transcribe.py audio/file.mp3 --mini

# Use Whisper-1 (detailed segments)
uv run python transcribe.py audio/file.mp3 --model whisper-1

# Use GPT-4o-transcribe (default, good accuracy)
uv run python transcribe.py audio/file.mp3 --4o
```

### Output Options

```bash
# Output text to stdout
uv run python transcribe.py audio/file.mp3 --txt

# Output JSON to stdout (Whisper-1 only)
uv run python transcribe.py audio/file.mp3 --model whisper-1 --json

# Force overwrite existing files
uv run python transcribe.py audio/file.mp3 --force
```

### Custom Context

```bash
# Use custom prompt for better accuracy
uv run python transcribe.py audio/file.mp3 --prompt "Names: Jud (not Judge). Technical terms: API, SDK"

# Default prompt from .env is used automatically if no --prompt specified
uv run python transcribe.py audio/file.mp3
```

## Audio Conversion

The repository includes a flexible shell script to convert various audio formats to MP3:

```bash
# Convert all audio files from SOURCE_AUDIO directory (from .env)
./convert_audio.sh

# Convert specific files (supports AMR, WAV, M4A, FLAC, OGG, AAC, WMA, etc.)
./convert_audio.sh file1.amr file2.wav file3.m4a

# Convert all audio files by extension in current directory
./convert_audio.sh *.amr *.wav *.m4a
```

**Supported formats**: AMR, WAV, M4A, FLAC, OGG, AAC, WMA, MP3, and other formats supported by ffmpeg.

## Models and Pricing

| Model | Cost/Minute | Features | Best For |
|-------|-------------|----------|----------|
| **whisper-1** | $0.006 | Detailed segments with timestamps, tokens, confidence scores | When you need detailed analysis |
| **gpt-4o-transcribe** | $0.006 | Better multilingual accuracy, simpler JSON format | General purpose, good accuracy |
| **gpt-4o-mini-transcribe** | $0.003 | Most economical, good accuracy | Cost-sensitive applications |

### Cost Estimation

For a collection of 38 files (35.91 minutes total):
- **whisper-1**: ~$0.22
- **gpt-4o-transcribe**: ~$0.22  
- **gpt-4o-mini-transcribe**: ~$0.11 (50% savings)

## Configuration

### Environment Variables

The `.env` file supports the following variables:

- **`OPENAI_API_KEY`** (required): Your OpenAI API key
- **`SOURCE_AUDIO`** (optional): Default source directory for audio files
- **`DEFAULT_PROMPT`** (optional): Default context prompt for improved transcription

### Default Prompt

The `DEFAULT_PROMPT` in your `.env` file provides context to improve transcription accuracy. It should include:

- **Names**: Include common misspellings and alternatives (e.g., "Jud (not Judge)")
- **Places**: Geographic locations frequently mentioned
- **Context**: Domain-specific information (e.g., "family voicemails, appointments")

Example:
```env
DEFAULT_PROMPT=Names: Jud (not Judge), [family names with alternatives]. Places: [your locations]. Context: family voicemails, appointments, scheduling.
```

## Output Formats

### File Output (Default)

Creates two files for each input:
- `filename.txt`: Plain text transcription
- `filename.json`: JSON with timestamps and metadata

### Stdout Output

- `--txt`: Outputs plain text to stdout (logs go to stderr)
- `--json`: Outputs JSON to stdout (Whisper-1 only, logs go to stderr)

## Advanced Usage

### Debug Mode

```bash
# Enable detailed logging
uv run python transcribe.py audio/file.mp3 --debug
```

### Batch Processing with Custom Settings

```bash
# Process all files with mini model and text output
uv run python transcribe.py audio/*.mp3 --mini --txt > all_transcriptions.txt
```

### Pipeline Integration

```bash
# Convert AMR files and transcribe in one go
./convert_audio.sh && uv run python transcribe.py audio/*.mp3 --mini
```

## Error Handling

The tool handles common errors gracefully:

- **Missing files**: Logs error and continues with other files
- **API failures**: Retries and logs detailed error information  
- **Invalid model combinations**: Validates before making API calls
- **File overwrites**: Prevents accidental overwrites unless `--force` is used

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found"**
   - Ensure your `.env` file exists and contains a valid API key

2. **"ffmpeg not found"**
   - Install ffmpeg using your system's package manager

3. **"JSON output not available for GPT-4o models"**
   - Use `--txt` output or switch to `--model whisper-1` for JSON

4. **Files already exist warning**
   - Use `--force` to overwrite or remove existing files

### Getting Help

```bash
# View all available options
uv run python transcribe.py --help
```

## Contributing

This tool was created by Jud Dagnall with Claude Code. Feel free to submit issues or improvements.

## License

This project is provided as-is for personal use.
