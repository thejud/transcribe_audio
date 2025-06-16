#!/usr/bin/env python3

"""
Preview Batch Processing Script

Shows what files would be processed without actually running the processing.
Useful for checking that the filtering works correctly.
"""

import sys
from pathlib import Path

# Import the filtering function from the main script
sys.path.append(".")
from batch_process_transcripts import (find_transcript_files,
                                       get_output_filenames)


def preview_processing(source_dir: str, output_dir: str):
    """Preview what would be processed."""
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    print(f"=== BATCH PROCESSING PREVIEW ===")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print()

    # Find transcript files (with filtering)
    transcript_files = find_transcript_files(source_path)

    print(f"Files that would be processed: {len(transcript_files)}")
    print()

    # Show what would be created for each file
    new_files = 0
    existing_files = 0

    for i, source_file in enumerate(transcript_files, 1):
        input_copy, postprocessed_file, report_file = get_output_filenames(
            source_file, output_path
        )

        # Check if already exists
        already_exists = postprocessed_file.exists() and report_file.exists()
        status = "EXISTS" if already_exists else "NEW"

        if already_exists:
            existing_files += 1
        else:
            new_files += 1

        print(f"[{i:2d}/{len(transcript_files)}] {status:6s} | {source_file.name}")

        if not already_exists:
            print(f"    Would create: {postprocessed_file.name}")
            print(f"    Would create: {report_file.name}")

        # Show just first few for brevity
        if i == 5 and len(transcript_files) > 10:
            remaining = len(transcript_files) - 5
            print(f"    ... and {remaining} more files")
            break

    print()
    print(f"Summary:")
    print(f"  Total transcripts found: {len(transcript_files)}")
    print(f"  Already processed: {existing_files}")
    print(f"  Would be processed: {new_files}")
    print()

    if new_files > 0:
        print(f"To run the actual processing:")
        print(f'  python batch_process_transcripts.py "{source_dir}" "{output_dir}"')
    else:
        print("All files are already processed!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preview_batch_processing.py SOURCE_DIR OUTPUT_DIR")
        print(
            'Example: python preview_batch_processing.py "mp3/" "tmp/"'
        )
        sys.exit(1)

    preview_processing(sys.argv[1], sys.argv[2])
