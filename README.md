# Audio Transcription Tool

A simple Python tool for transcribing audio files (MP3, M4A, WAV, FLAC, etc.) using OpenAI's transcription models with intelligent chunking, context-aware prompts, and multiple output formats.

## Quick Start

### Simple Voice Memo Pipeline (Recommended)

For most voice memo use cases, use the streamlined pipeline that handles transcription and post-processing in one command:

```bash
# Process voice memos (automatically transcribes and summarizes)
python transcribe_pipeline.py voicemail.mp3
python transcribe_pipeline.py *.mp3 *.m4a

# Specify output directory
python transcribe_pipeline.py audio/*.mp3 -O processed/

# Higher quality processing
python transcribe_pipeline.py meeting.mp3 --quality high
```

### Individual Tools (For Advanced Use)

For more control or general audio transcription:

```bash
# Just transcription (cost-effective)
python transcribe.py audio/*.mp3 --mini

# Just post-processing 
python post_process.py transcript.txt --memo

# Full control with advanced options
python transcribe.py audio.mp3 --advanced --model whisper-1 --complex-json
python post_process.py transcript.txt --advanced --verify --4o
```

### Three Usage Levels

1. **Simple Pipeline**: `transcribe_pipeline.py` - Voice memos → final text in one step
2. **Standard Tools**: `transcribe.py` and `post_process.py` - Simplified interfaces with sensible defaults  
3. **Advanced**: Add `--advanced` flag for full feature access

## Voice Memo Watcher

The `voice_memo_watcher.py` script provides automated processing of voice memos with smart file organization:

```bash
# Watch and process all voice memos in input directory
python voice_memo_watcher.py

# Use custom directories
python voice_memo_watcher.py --audio-in ~/voice_inbox --audio-out ~/voice_processed --transcript-out ~/transcripts

# Verbose logging
python voice_memo_watcher.py --verbose
```

### Features

- **Automated Processing**: Processes all audio files in the input directory through the transcription pipeline
- **Smart Naming**: Extracts titles from `<title>` tags and creates meaningful filenames
- **Obsidian-Ready**: Outputs markdown files (`.md`) with proper headers and space-friendly filenames
- **M4A Metadata**: Extracts creation timestamps from M4A voice memo files
- **File Organization**: Separates input, processed audio, and transcript files into different directories
- **Path Logging**: Prints full file paths for easy access and opening

### Configuration

Add voice memo directories to your `.env` file:

```env
# Voice memo watcher directories
AUDIO_IN=voice_memo_inbox
AUDIO_OUT=voice_memo_outbox  
TRANSCRIPT_OUT=voice_memo_transcripts
```

### Workflow

1. **Drop voice memos** into the `AUDIO_IN` directory
2. **Run the watcher** to process all files
3. **Find organized output**:
   - Original audio files → `AUDIO_OUT` (renamed with timestamps/titles)
   - Transcripts → `TRANSCRIPT_OUT` (markdown files with `# Title` headers)

### Output Example

Input: `Recording001.m4a` → Output:
- Audio: `20241215_143022_Meeting Notes for Project X.m4a`
- Transcript: `20241215_143022_Meeting Notes for Project X.md`

The transcript will contain a markdown header (`# Meeting Notes for Project X`) instead of XML tags, making it perfect for Obsidian and other markdown editors.

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
- **Intelligent Caching**: Caches audio chunks in `/tmp/` to avoid re-processing large files
- **Multiple AI Models**: Support for Whisper-1, GPT-4o-transcribe, and GPT-4o-mini-transcribe
- **Flexible Output**: Save to files (text only by default, optional JSON) or output directly to stdout
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

## Standard Usage

### Simple Transcription (Individual Tools)

```bash
# Basic transcription with cost-effective model
python transcribe.py audio/voicemail.mp3 --mini

# Multiple files  
python transcribe.py audio/*.mp3 --mini

# Custom context for better accuracy
python transcribe.py audio/file.mp3 --prompt "Names: Jud (not Judge)"

# Save to specific directory
python transcribe.py audio/*.mp3 -O transcriptions/
```

### Basic Post-Processing

```bash
# Voice memo summarization (recommended)
python post_process.py transcript.txt --memo

# Basic transcript formatting
python post_process.py transcript.txt

# Save to specific directory
python post_process.py transcript.txt -O processed/

# Process in-place (overwrites original)
python post_process.py transcript.txt --inplace
```

> **💡 Tip**: For voice memos, consider using `python transcribe_pipeline.py audio/*.mp3` which handles both steps automatically with optimized settings.


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
- **`AUDIO_IN`** (optional): Input directory for voice memo watcher
- **`AUDIO_OUT`** (optional): Output directory for processed audio files
- **`TRANSCRIPT_OUT`** (optional): Output directory for transcript markdown files

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

By default, creates one file for each input:
- `filename.txt`: Plain text transcription

With `--complex-json` flag, creates two files:
- `filename.txt`: Plain text transcription
- `filename.json`: JSON with timestamps and metadata

### Stdout Output

- `--txt`: Outputs plain text to stdout (logs go to stderr)
- `--json`: Outputs JSON to stdout (Whisper-1 only, logs go to stderr)

## Advanced Usage

For users who need more control or want to use this as a demonstration project, the individual tools provide extensive customization options when used with the `--advanced` flag.

### Advanced Transcription Options

```bash
# Advanced model and output control
python transcribe.py audio.mp3 --advanced --model whisper-1 --complex-json

# Stdout output for pipeline integration
python transcribe.py audio/*.mp3 --advanced --txt > all_transcriptions.txt

# JSON output with detailed segments (Whisper-1 only)
python transcribe.py audio.mp3 --advanced --model whisper-1 --json
```

### Advanced Post-Processing Options

```bash
# Advanced verification and chunking control
python post_process.py transcript.txt --advanced --verify --chunk-size 5000

# Custom models and processing options
python post_process.py transcript.txt --advanced --4o --verify-mode strict

# Testing and development
python post_process.py transcript.txt --advanced --test-mode --test-limit 1000
```

### Pipeline Integration Examples

```bash
# Convert and process in one pipeline
./convert_audio.sh *.amr && python transcribe_pipeline.py audio/*.mp3

# Custom processing pipeline with intermediate files
python transcribe.py audio/*.mp3 --advanced --mini -O temp/
python post_process.py temp/*.txt --advanced --memo --verify -O final/

# Batch processing with quality control
python transcribe.py audio/*.mp3 --advanced --model whisper-1 --complex-json
python post_process.py *.txt --advanced --verify --verify-mode word-only
```

## Audio Chunk Caching

The tool automatically caches processed audio chunks to dramatically speed up re-processing of large audio files. This is especially useful when:

- Re-transcribing files with different models
- Experimenting with different prompts
- Processing very large audio files that take significant time to chunk

### How Caching Works

- **Cache Location**: `/tmp/transcribe_cache/`
- **Cache Key**: Based on file path, modification time, and file size
- **Automatic Invalidation**: Cache is rebuilt when source file is updated
- **Diagnostic Output**: Clear logging shows when building vs. using cached chunks

### Cache Behavior

```bash
# First run - builds and caches chunks
uv run python transcribe.py large_audio.mp3 --debug
# Output: "Building chunks for large_audio.mp3 (not in cache)"
# Output: "Cached 15 chunks to /tmp/transcribe_cache/abc123.pkl"

# Subsequent runs - uses cached chunks
uv run python transcribe.py large_audio.mp3 --debug  
# Output: "Using cached chunks for large_audio.mp3"
# Output: "Loaded 15 chunks from cache (/tmp/transcribe_cache/abc123.pkl)"
```

### Cache Management

- Cache files are automatically cleaned up when corrupted
- Cache is invalidated when source files are modified
- Manual cache clearing: `rm -rf /tmp/transcribe_cache/`
- Cache files use minimal disk space (typically <1MB per audio file)

## Advanced Usage

### Debug Mode

```bash
# Enable detailed logging (shows cache hits/misses)
uv run python transcribe.py audio/file.mp3 --debug
```

### Batch Processing with Custom Settings

```bash
# Process all files with mini model and text output
uv run python transcribe.py audio/*.mp3 --mini --txt > all_transcriptions.txt

# Process all files and generate both text and JSON outputs
uv run python transcribe.py audio/*.mp3 --complex-json
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

## Testing

The project includes a comprehensive test suite that validates transcription accuracy:

```bash
# Run the main transcription test suite
cd test && python test_transcribe.py

# Regenerate test audio files and run tests
cd test && python test_transcribe.py --regenerate

# Test output directory and file options
cd test && python test_output_options.py

# Enable debug logging for detailed test information
cd test && python test_transcribe.py --debug

# View all available options
cd test && python test_transcribe.py --help
```

### Test Coverage

The test suite validates:
- **Basic transcription accuracy** with known content
- **Both MP3 and WAV format support** 
- **Text-only vs JSON output modes** (--complex-json flag)
- **Output directory functionality** (--out-dir parameter)
- **Post-processing options** (--inplace, --extension, --out-dir)
- **Name recognition** for challenging names like "Jud" vs "Judge"
- **Technical terminology** transcription
- **JSON structure validation** when using --complex-json
- **Error handling** and edge cases

Test audio files are generated using OpenAI's text-to-speech API and preserved as permanent fixtures in the `test/test_audio/` directory for consistent testing across runs.

## Contributing

This tool was created by Jud Dagnall with Claude Code. Feel free to submit issues or improvements.

## License

This project is provided as-is for personal use.
