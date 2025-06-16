# Batch Transcript Processing Guide

## Overview
The `batch_process_transcripts.py` script automates the postprocessing and validation of all transcript files in a directory. It performs the same steps we did manually but for all remaining transcripts.

## What the Script Does

For each `.txt` transcript file, it:
1. **Copies** the input file to the output directory
2. **Runs postprocessing** to create a formatted version with " - post" suffix  
3. **Generates validation report** with comprehensive analysis and " - report" suffix
4. **Skips existing files** by default to avoid reprocessing
5. **Tracks progress** and provides detailed summary statistics

## Usage

### Basic Usage
```bash
# Process all transcripts from source to tmp directory
python batch_process_transcripts.py "/path/to/transcripts/" "tmp/"
```

### Advanced Options
```bash
# Reprocess all files (don't skip existing)
python batch_process_transcripts.py "/path/to/transcripts/" "tmp/" --no-skip-existing

# Enable debug logging for troubleshooting
python batch_process_transcripts.py "/path/to/transcripts/" "tmp/" --debug
```

## Output Structure

For each transcript file like `"Speaker Name - Example Talk.txt"`, the script creates:

```
tmp/
├── Speaker Name - Example Talk.txt          # Original copy
├── Speaker Name - Example Talk - post.txt   # Postprocessed version
└── Speaker Name - Example Talk - report.txt # Validation report
```

## Expected Results

Based on our testing, you can expect:
- ✅ **Perfect semantic preservation** (1.0000 similarity scores)
- ✅ **Excellent formatting transformation** (0 line breaks → 100+ paragraphs)
- ✅ **Moderate text expansion** (~6% character increase for readability)
- ✅ **Fast processing** (~25 seconds per transcript with caching)

## Sample Output

```
2025-06-14 01:45:00,123 - INFO - Starting batch processing of 16 transcripts...
2025-06-14 01:45:00,124 - INFO - Output directory: tmp/

[1/16] Processing: Speaker Name - A Sacred Watchfulness.txt
✓ Already processed (skipped)

[2/16] Processing: Speaker Name - Catching Up With Soul.txt  
✓ Already processed (skipped)

[3/16] Processing: Speaker Name - Gleaning Spiritual Harvest.txt
✅ Successfully processed

...

============================================================
BATCH PROCESSING COMPLETE
============================================================
Total files: 16
Successfully processed: 14
Skipped (already processed): 2
Failed: 0
Total duration: 380.5 seconds
Average per file: 23.8 seconds
Output directory: /path/to/tmp
All files processed successfully! 🎉
```

## Error Handling

The script handles errors gracefully:
- **Continues processing** other files if one fails
- **Provides detailed error messages** for troubleshooting  
- **Tracks failed files** in the summary
- **Times out** long-running processes (5 minute limit per file)

## Performance Notes

- **Caching**: The postprocessing tool caches API results for efficiency
- **Concurrent processing**: Each transcript processed sequentially for stability
- **Memory efficient**: Processes one file at a time
- **Progress tracking**: Shows current file and overall progress

## Validation Reports

Each report includes:
- **File statistics** (character/word counts, line formatting)
- **Multiple validation modes** (strict, semantic, comprehensive)
- **Semantic analysis** using NLP techniques
- **Content integrity assessment**
- **Recommendations** for validation approach

This gives you confidence that all the talks are being processed with the same high quality and semantic preservation we demonstrated with the individual examples.