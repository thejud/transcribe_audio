# Testing Guide

This document describes the automated test suite for the transcription tool.

## Test Directory Structure

All test-related files are organized in the `test/` directory:

```
test/
├── test_transcribe.py          # Main transcription accuracy tests
├── test_output_options.py      # Output directory and file option tests  
└── test_audio/                 # Test audio fixtures
    ├── test_basic.mp3/wav      # Basic transcription test files
    ├── test_names.mp3/wav      # Name recognition test files
    └── test_technical.mp3/wav  # Technical terminology test files
```

## Overview

The test suite (`test_transcribe.py`) validates the accuracy and functionality of the transcription system by:

1. **Generating known test audio** using OpenAI's text-to-speech API
2. **Running transcription** on the generated audio files  
3. **Validating output** against expected content and structure

## Test Files

The test suite creates the following permanent test fixtures in `test_audio/`:

### Audio Files (in test/test_audio/)
- `test_basic.mp3` / `test_basic.wav` - Basic transcription with names and scheduling
- `test_names.mp3` / `test_names.wav` - Challenging names (Jud, Catherine, Michael)  
- `test_technical.mp3` / `test_technical.wav` - Technical terminology (API, SDK, ML)

### Expected Output Files
- `*.txt` - Plain text transcriptions
- `*.json` - JSON with timestamps and segments (when using --complex-json)

## Running Tests

```bash
# Change to test directory first
cd test

# Run tests with existing audio files (recommended)
python test_transcribe.py

# Regenerate audio files and run tests (if needed)
python test_transcribe.py --regenerate

# Test output directory and file options
python test_output_options.py

# Enable debug logging for detailed information
python test_transcribe.py --debug

# Combine options for regeneration with debug output
python test_transcribe.py --regenerate --debug

# View all available options and help
python test_transcribe.py --help
```

### Command Line Options

- `--regenerate`: Force regeneration of test audio files using OpenAI's text-to-speech API. By default, existing audio files in `test_audio/` are reused for faster testing.
- `--debug`: Enable debug-level logging for detailed test execution information, including file operations and transcription details.
- `--help`: Display usage information and all available options.

## Test Coverage

### Formats Tested
- **MP3 format** - Standard compressed audio
- **WAV format** - Uncompressed audio 

### Output Modes Tested
- **Text only** (default behavior)
- **Text + JSON** (with --complex-json flag)

### Validation Criteria

#### Accuracy Metrics
- **Word accuracy**: Minimum 70% expected (typically 88-100%)
- **Length validation**: Ensures reasonable output length
- **Name recognition**: Validates challenging names like "Jud" vs "Judge"

#### JSON Structure Validation
When `--complex-json` is used, validates:
- Required fields: `text`, `segments`
- Segment structure: `start`, `end`, `text` timestamps
- Data types and format consistency

#### Error Handling
- Missing output files
- Invalid JSON structure  
- API failures and exceptions

## Test Results

The test suite provides:
- **Pass/fail status** for each test case
- **Word accuracy percentages** 
- **Warning messages** for low accuracy or missing expected content
- **Error details** for any failures
- **Summary statistics** across all tests

Example output:
```
=== Test Summary ===
Total tests: 12
Passed: 12  
Failed: 0

Average word accuracy: 94.4%
```

## Test Files Persistence

Test audio files are **preserved** between runs to:
- Ensure consistent test conditions
- Avoid unnecessary API calls for audio generation
- Provide stable test fixtures for CI/CD pipelines
- Enable faster test execution

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found"**
   - Ensure your `.env` file contains a valid OpenAI API key
   - The same key used for transcription is used for audio generation

2. **Low accuracy warnings**
   - Some variation in transcription is normal
   - Warnings appear for <70% word accuracy
   - Consider regenerating audio files if consistently low

3. **Missing test files**  
   - Run with `--regenerate` to create new audio files
   - Check that `test_audio/` directory is writable

### Debugging

Enable debug mode in the transcription calls by modifying the test script:
```python
cmd = ["python", "transcribe.py", str(audio_path), "--force", "--debug"]
```

This will show detailed logging from the transcription process.

## Extending Tests

To add new test cases:

1. **Add test content** to `get_test_content()` function
2. **Define expected results** including challenging words/names
3. **Run with --regenerate** to create new audio files  
4. **Validate** that new tests pass with reasonable accuracy

The test framework is designed to be easily extensible for additional scenarios.