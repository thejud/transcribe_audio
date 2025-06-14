#!/usr/bin/env python3

"""
Text Integrity Verification Tool

This script compares two text files to verify they contain essentially the same
content, with options to ignore whitespace differences. Designed for verifying
that text processing operations preserve the original content.

Features:
- Multiple comparison modes (strict, normalized, word-only)
- Detailed diff output when differences are found
- Statistics about text similarity
- Pipeline-friendly exit codes
- Comprehensive logging

Usage:
    # Basic verification (ignore whitespace differences)
    python verify_text_integrity.py input.txt output.txt
    
    # Strict mode (exact match including whitespace)
    python verify_text_integrity.py input.txt output.txt --strict
    
    # Show detailed diff when differences found
    python verify_text_integrity.py input.txt output.txt --show-diff
    
    # Word-only comparison (ignore all whitespace and punctuation)
    python verify_text_integrity.py input.txt output.txt --mode word-only

Environment Variables:
    None required

Author: Jud Dagnall (with Claude Code)
Version: 1.0
"""

import argparse
import difflib
import logging
import re
import sys
import textwrap
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple


class ComparisonResult(NamedTuple):
    """Result of text comparison."""
    identical: bool
    similarity_score: float
    input_chars: int
    output_chars: int
    input_words: int
    output_words: int
    differences: Optional[List[str]] = None


def setup_logging(debug: bool = False) -> None:
    """
    Configure logging for the application.
    
    Args:
        debug: Enable debug-level logging
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )


def read_text_file(file_path: Path) -> str:
    """
    Read text content from a file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        str: Content of the text file
        
    Raises:
        SystemExit: If file cannot be read
    """
    if not file_path.exists():
        logging.error(f"File not found: {file_path}")
        sys.exit(1)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logging.debug(f"Read file: {file_path} ({len(content)} characters)")
        return content
        
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {str(e)}")
        sys.exit(1)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text for comparison.
    
    This function:
    - Converts all whitespace (spaces, tabs, newlines) to single spaces
    - Removes leading and trailing whitespace
    - Collapses multiple consecutive spaces into single spaces
    
    Args:
        text: Input text to normalize
        
    Returns:
        str: Text with normalized whitespace
    """
    # Replace all whitespace characters with single spaces
    normalized = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing whitespace
    return normalized.strip()


def extract_words_only(text: str) -> str:
    """
    Extract only words from text, removing all punctuation and whitespace.
    
    Args:
        text: Input text to process
        
    Returns:
        str: Words separated by single spaces
    """
    # Extract only word characters (letters, digits, underscores)
    words = re.findall(r'\w+', text.lower())
    return ' '.join(words)


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity ratio between two texts using difflib.
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        
    Returns:
        float: Similarity ratio between 0.0 and 1.0
    """
    return difflib.SequenceMatcher(None, text1, text2).ratio()


def wrap_text_for_diff(text: str, width: int = 80) -> List[str]:
    """
    Wrap text into lines for better diff display when original has no line breaks.
    
    Args:
        text: Text to wrap
        width: Maximum characters per line
        
    Returns:
        List[str]: Lines with proper wrapping
    """
    if '\n' in text and len(text.splitlines()) > 3:
        # Text already has reasonable line breaks
        return text.splitlines(keepends=True)
    
    # Text is one long line or has very few breaks - wrap by sentences
    import textwrap
    
    # Split by sentences first for better semantic breaks
    sentences = re.split(r'(?<=[.!?])\s+', text)
    wrapped_lines = []
    
    for sentence in sentences:
        if len(sentence) <= width:
            wrapped_lines.append(sentence + ' ')
        else:
            # Use textwrap for very long sentences
            wrapped_sentence_lines = textwrap.wrap(sentence, width=width)
            for i, line in enumerate(wrapped_sentence_lines):
                if i == len(wrapped_sentence_lines) - 1:
                    wrapped_lines.append(line + ' ')
                else:
                    wrapped_lines.append(line)
    
    return [line + '\n' for line in wrapped_lines]


def generate_diff(text1: str, text2: str, file1_name: str, file2_name: str) -> List[str]:
    """
    Generate unified diff between two texts with intelligent line wrapping.
    
    Args:
        text1: First text to compare
        text2: Second text to compare  
        file1_name: Name of first file for diff header
        file2_name: Name of second file for diff header
        
    Returns:
        List[str]: Unified diff lines
    """
    # Wrap texts for better diff display if they lack line breaks
    wrapped_text1 = wrap_text_for_diff(text1)
    wrapped_text2 = wrap_text_for_diff(text2)
    
    diff_lines = list(difflib.unified_diff(
        wrapped_text1,
        wrapped_text2,
        fromfile=file1_name,
        tofile=file2_name,
        lineterm=''
    ))
    return diff_lines


def compare_texts(
    input_text: str, 
    output_text: str, 
    mode: str = "normalized",
    generate_diff_output: bool = False,
    file1_name: str = "input",
    file2_name: str = "output"
) -> ComparisonResult:
    """
    Compare two texts according to the specified mode.
    
    Args:
        input_text: Original input text
        output_text: Processed output text
        mode: Comparison mode ('strict', 'normalized', 'word-only')
        generate_diff_output: Whether to generate diff output
        file1_name: Name of first file for reporting
        file2_name: Name of second file for reporting
        
    Returns:
        ComparisonResult: Result of the comparison
    """
    # Store original statistics
    input_chars = len(input_text)
    output_chars = len(output_text)
    input_words = len(input_text.split())
    output_words = len(output_text.split())
    
    # Prepare texts for comparison based on mode
    if mode == "strict":
        compare_text1 = input_text
        compare_text2 = output_text
    elif mode == "normalized":
        compare_text1 = normalize_whitespace(input_text)
        compare_text2 = normalize_whitespace(output_text)
    elif mode == "word-only":
        compare_text1 = extract_words_only(input_text)
        compare_text2 = extract_words_only(output_text)
    else:
        raise ValueError(f"Unknown comparison mode: {mode}")
    
    # Check if texts are identical
    identical = compare_text1 == compare_text2
    
    # Calculate similarity score
    similarity_score = calculate_similarity(compare_text1, compare_text2)
    
    # Generate diff if requested and texts are not identical
    differences = None
    if generate_diff_output and not identical:
        differences = generate_diff(compare_text1, compare_text2, file1_name, file2_name)
    
    return ComparisonResult(
        identical=identical,
        similarity_score=similarity_score,
        input_chars=input_chars,
        output_chars=output_chars,
        input_words=input_words,
        output_words=output_words,
        differences=differences
    )


def print_comparison_results(result: ComparisonResult, mode: str, show_diff: bool = False) -> None:
    """
    Print the results of text comparison.
    
    Args:
        result: ComparisonResult to display
        mode: Comparison mode used
        show_diff: Whether to show diff output
    """
    print(f"Comparison mode: {mode}")
    print(f"Files identical: {'✓ YES' if result.identical else '✗ NO'}")
    print(f"Similarity score: {result.similarity_score:.4f}")
    print(f"Input chars: {result.input_chars:,}, words: {result.input_words:,}")
    print(f"Output chars: {result.output_chars:,}, words: {result.output_words:,}")
    
    if not result.identical:
        char_diff = result.output_chars - result.input_chars
        word_diff = result.output_words - result.input_words
        print(f"Character difference: {char_diff:+,}")
        print(f"Word difference: {word_diff:+,}")
        
        if show_diff and result.differences:
            print("\nDetailed diff:")
            print("".join(result.differences))


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Verify text integrity by comparing two files",
        epilog="""
Examples:
  %(prog)s input.txt output.txt                    # Basic comparison (normalized whitespace)
  %(prog)s input.txt output.txt --strict           # Exact match including whitespace
  %(prog)s input.txt output.txt --mode word-only   # Compare words only
  %(prog)s input.txt output.txt --show-diff        # Show detailed differences
  %(prog)s input.txt output.txt --debug            # Enable debug logging
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Positional arguments
    parser.add_argument(
        "input_file",
        help="Path to the original input file"
    )
    parser.add_argument(
        "output_file", 
        help="Path to the processed output file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--mode",
        choices=["strict", "normalized", "word-only"],
        default="normalized",
        help="Comparison mode (default: normalized)"
    )
    parser.add_argument(
        "--strict",
        action="store_const",
        const="strict",
        dest="mode",
        help="Use strict comparison (exact match including whitespace)"
    )
    parser.add_argument(
        "--show-diff",
        action="store_true",
        help="Show detailed diff output when files differ"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


def main() -> None:
    """
    Main function to handle command line arguments and perform text verification.
    
    Exit codes:
        0: Files are identical (or meet similarity criteria)
        1: Files differ significantly
        2: Error occurred (file not found, read error, etc.)
    """
    try:
        args = parse_args()
        
        # Set up logging
        setup_logging(args.debug)
        
        # Read input files
        input_path = Path(args.input_file)
        output_path = Path(args.output_file)
        
        logging.info(f"Comparing {input_path} and {output_path}")
        
        input_text = read_text_file(input_path)
        output_text = read_text_file(output_path)
        
        # Perform comparison
        result = compare_texts(
            input_text, 
            output_text, 
            mode=args.mode,
            generate_diff_output=args.show_diff,
            file1_name=str(input_path),
            file2_name=str(output_path)
        )
        
        # Print results
        print_comparison_results(result, args.mode, args.show_diff)
        
        # Exit with appropriate code
        if result.identical:
            logging.info("Text integrity verification PASSED")
            sys.exit(0)
        else:
            logging.warning("Text integrity verification FAILED")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(2)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        sys.exit(2)


if __name__ == "__main__":
    main()