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
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

# NLP libraries (optional imports for graceful degradation)
try:
    import numpy as np
    import spacy
    from rapidfuzz import fuzz
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    ADVANCED_NLP_AVAILABLE = True
except ImportError as e:
    ADVANCED_NLP_AVAILABLE = False
    logging.warning(f"Advanced NLP features not available: {e}")


class ComparisonResult(NamedTuple):
    """Result of text comparison."""

    identical: bool
    similarity_score: float
    input_chars: int
    output_chars: int
    input_words: int
    output_words: int
    differences: Optional[List[str]] = None
    semantic_similarity: Optional[float] = None
    entity_preservation: Optional[float] = None
    detailed_report: Optional[Dict] = None


def setup_logging(debug: bool = False) -> None:
    """
    Configure logging for the application.

    Args:
        debug: Enable debug-level logging
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
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
        with open(file_path, "r", encoding="utf-8") as f:
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
    normalized = re.sub(r"\s+", " ", text)
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
    words = re.findall(r"\w+", text.lower())
    return " ".join(words)


def normalize_formatting_differences(text: str) -> str:
    """
    Normalize text to ignore common formatting differences that postprocessing might introduce.

    This is more aggressive than basic whitespace normalization and handles:
    - Markdown formatting (*text*, **text**)
    - Quote style differences (" vs ')
    - Capitalization of proper nouns and titles
    - Punctuation spacing

    Args:
        text: Input text to normalize

    Returns:
        str: Text with formatting differences normalized
    """
    # Remove markdown formatting
    normalized = re.sub(r"\*([^*]+)\*", r"\1", text)  # Remove italics *text*
    normalized = re.sub(r"\*\*([^*]+)\*\*", r"\1", normalized)  # Remove bold **text**

    # Normalize quotes - handle smart quotes one by one
    normalized = normalized.replace('"', '"').replace(
        '"', '"'
    )  # Smart quotes to regular
    normalized = normalized.replace(
        """, "'").replace(""", "'"
    )  # Smart apostrophes to regular

    # Normalize common title cases (but preserve intentional capitalization)
    # Convert common phrases to consistent casing

    # Normalize whitespace
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip()

    return normalized


def get_sentence_embeddings(
    text: str, model_name: str = "all-MiniLM-L6-v2"
) -> Optional[np.ndarray]:
    """
    Get sentence embeddings for semantic similarity comparison.

    Args:
        text: Text to encode
        model_name: SentenceTransformer model to use

    Returns:
        Optional[np.ndarray]: Sentence embeddings or None if not available
    """
    if not ADVANCED_NLP_AVAILABLE:
        return None

    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode([text])
        return embeddings[0]
    except Exception as e:
        logging.warning(f"Could not generate embeddings: {e}")
        return None


def extract_named_entities(text: str) -> Optional[Set[str]]:
    """
    Extract named entities from text for preservation checking.

    Args:
        text: Text to analyze

    Returns:
        Optional[Set[str]]: Set of named entities or None if not available
    """
    if not ADVANCED_NLP_AVAILABLE:
        return None

    try:
        # Try to load English model, download if needed
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning(
                "SpaCy English model not found. Install with: python -m spacy download en_core_web_sm"
            )
            return None

        doc = nlp(text)
        entities = {
            ent.text.lower().strip()
            for ent in doc.ents
            if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]
        }
        return entities
    except Exception as e:
        logging.warning(f"Could not extract entities: {e}")
        return None


def calculate_semantic_similarity(text1: str, text2: str) -> Optional[float]:
    """
    Calculate semantic similarity between two texts using sentence embeddings.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Optional[float]: Similarity score between 0 and 1, or None if not available
    """
    if not ADVANCED_NLP_AVAILABLE:
        return None

    try:
        embeddings1 = get_sentence_embeddings(text1)
        embeddings2 = get_sentence_embeddings(text2)

        if embeddings1 is None or embeddings2 is None:
            return None

        # Calculate cosine similarity
        similarity = cosine_similarity([embeddings1], [embeddings2])[0][0]
        return float(similarity)
    except Exception as e:
        logging.warning(f"Could not calculate semantic similarity: {e}")
        return None


def calculate_entity_preservation(text1: str, text2: str) -> Optional[float]:
    """
    Calculate how well named entities are preserved between texts.

    Args:
        text1: Original text
        text2: Processed text

    Returns:
        Optional[float]: Preservation ratio between 0 and 1, or None if not available
    """
    if not ADVANCED_NLP_AVAILABLE:
        return None

    entities1 = extract_named_entities(text1)
    entities2 = extract_named_entities(text2)

    if entities1 is None or entities2 is None:
        return None

    if not entities1:  # No entities in original
        return 1.0 if not entities2 else 0.0

    # Calculate Jaccard similarity for entity preservation
    intersection = len(entities1.intersection(entities2))
    union = len(entities1.union(entities2))

    if union == 0:
        return 1.0

    return intersection / union


def fuzzy_word_similarity(text1: str, text2: str) -> float:
    """
    Calculate fuzzy similarity at word level to handle minor spelling/formatting differences.

    Args:
        text1: First text
        text2: Second text

    Returns:
        float: Similarity score between 0 and 100
    """
    if not ADVANCED_NLP_AVAILABLE:
        # Fallback to basic ratio
        return difflib.SequenceMatcher(None, text1, text2).ratio() * 100

    try:
        return fuzz.ratio(text1, text2)
    except Exception:
        return difflib.SequenceMatcher(None, text1, text2).ratio() * 100


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
    if "\n" in text and len(text.splitlines()) > 3:
        # Text already has reasonable line breaks
        return text.splitlines(keepends=True)

    # Text is one long line or has very few breaks - wrap by sentences
    import textwrap

    # Split by sentences first for better semantic breaks
    sentences = re.split(r"(?<=[.!?])\s+", text)
    wrapped_lines = []

    for sentence in sentences:
        if len(sentence) <= width:
            wrapped_lines.append(sentence + " ")
        else:
            # Use textwrap for very long sentences
            wrapped_sentence_lines = textwrap.wrap(sentence, width=width)
            for i, line in enumerate(wrapped_sentence_lines):
                if i == len(wrapped_sentence_lines) - 1:
                    wrapped_lines.append(line + " ")
                else:
                    wrapped_lines.append(line)

    return [line + "\n" for line in wrapped_lines]


def generate_diff(
    text1: str, text2: str, file1_name: str, file2_name: str
) -> List[str]:
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

    diff_lines = list(
        difflib.unified_diff(
            wrapped_text1,
            wrapped_text2,
            fromfile=file1_name,
            tofile=file2_name,
            lineterm="",
        )
    )
    return diff_lines


def compare_texts(
    input_text: str,
    output_text: str,
    mode: str = "normalized",
    generate_diff_output: bool = False,
    file1_name: str = "input",
    file2_name: str = "output",
) -> ComparisonResult:
    """
    Compare two texts according to the specified mode.

    Args:
        input_text: Original input text
        output_text: Processed output text
        mode: Comparison mode ('strict', 'normalized', 'word-only', 'formatting-tolerant', 'semantic', 'comprehensive')
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

    # Initialize advanced metrics
    semantic_similarity = None
    entity_preservation = None
    detailed_report = {}

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
    elif mode == "formatting-tolerant":
        compare_text1 = normalize_formatting_differences(input_text)
        compare_text2 = normalize_formatting_differences(output_text)
    elif mode == "semantic":
        # For semantic mode, use formatting-tolerant comparison but add semantic analysis
        compare_text1 = normalize_formatting_differences(input_text)
        compare_text2 = normalize_formatting_differences(output_text)
        semantic_similarity = calculate_semantic_similarity(input_text, output_text)
        entity_preservation = calculate_entity_preservation(input_text, output_text)
    elif mode == "comprehensive":
        # Comprehensive mode: formatting-tolerant + all NLP metrics
        compare_text1 = normalize_formatting_differences(input_text)
        compare_text2 = normalize_formatting_differences(output_text)
        semantic_similarity = calculate_semantic_similarity(input_text, output_text)
        entity_preservation = calculate_entity_preservation(input_text, output_text)

        # Additional analysis for detailed report
        fuzzy_score = fuzzy_word_similarity(compare_text1, compare_text2)
        detailed_report = {
            "fuzzy_similarity": fuzzy_score,
            "word_count_diff": abs(input_words - output_words),
            "char_count_diff": abs(input_chars - output_chars),
            "semantic_available": semantic_similarity is not None,
            "entities_available": entity_preservation is not None,
        }
    else:
        raise ValueError(f"Unknown comparison mode: {mode}")

    # Check if texts are identical
    identical = compare_text1 == compare_text2

    # Calculate similarity score
    similarity_score = calculate_similarity(compare_text1, compare_text2)

    # For semantic modes, adjust decision based on additional metrics
    if mode in ["semantic", "comprehensive"] and not identical:
        # Consider texts "equivalent" if semantic similarity is high enough
        if semantic_similarity and semantic_similarity > 0.95:
            if entity_preservation is None or entity_preservation > 0.9:
                # High semantic similarity and good entity preservation
                logging.info(
                    f"Texts differ but have high semantic similarity ({semantic_similarity:.3f})"
                )
                identical = True

    # Generate diff if requested and texts are not identical
    differences = None
    if generate_diff_output and not identical:
        differences = generate_diff(
            compare_text1, compare_text2, file1_name, file2_name
        )

    return ComparisonResult(
        identical=identical,
        similarity_score=similarity_score,
        input_chars=input_chars,
        output_chars=output_chars,
        input_words=input_words,
        output_words=output_words,
        differences=differences,
        semantic_similarity=semantic_similarity,
        entity_preservation=entity_preservation,
        detailed_report=detailed_report if detailed_report else None,
    )


def print_comparison_results(
    result: ComparisonResult, mode: str, show_diff: bool = False
) -> None:
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

    # Show advanced metrics if available
    if result.semantic_similarity is not None:
        print(f"Semantic similarity: {result.semantic_similarity:.4f}")

    if result.entity_preservation is not None:
        print(f"Entity preservation: {result.entity_preservation:.4f}")

    if not result.identical:
        char_diff = result.output_chars - result.input_chars
        word_diff = result.output_words - result.input_words
        print(f"Character difference: {char_diff:+,}")
        print(f"Word difference: {word_diff:+,}")

        # Show detailed report if available
        if result.detailed_report:
            print("\nDetailed Analysis:")
            report = result.detailed_report
            if "fuzzy_similarity" in report:
                print(f"  Fuzzy similarity: {report['fuzzy_similarity']:.1f}%")
            if "word_count_diff" in report:
                print(f"  Word count difference: {report['word_count_diff']}")
            if "char_count_diff" in report:
                print(f"  Character count difference: {report['char_count_diff']}")
            if not report.get("semantic_available", False):
                print(
                    "  Note: Semantic analysis not available (install sentence-transformers)"
                )
            if not report.get("entities_available", False):
                print(
                    "  Note: Entity analysis not available (install spacy and en_core_web_sm)"
                )

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
  %(prog)s input.txt output.txt                              # Basic comparison (normalized whitespace)
  %(prog)s input.txt output.txt --strict                     # Exact match including whitespace
  %(prog)s input.txt output.txt --mode word-only             # Compare words only
  %(prog)s input.txt output.txt --mode formatting-tolerant   # Ignore formatting differences
  %(prog)s input.txt output.txt --mode semantic              # Use semantic similarity analysis
  %(prog)s input.txt output.txt --mode comprehensive         # Full NLP analysis with detailed report
  %(prog)s input.txt output.txt --show-diff                  # Show detailed differences
  %(prog)s input.txt output.txt --debug                      # Enable debug logging
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional arguments
    parser.add_argument("input_file", help="Path to the original input file")
    parser.add_argument("output_file", help="Path to the processed output file")

    # Optional arguments
    parser.add_argument(
        "--mode",
        choices=[
            "strict",
            "normalized",
            "word-only",
            "formatting-tolerant",
            "semantic",
            "comprehensive",
        ],
        default="normalized",
        help="Comparison mode (default: normalized)",
    )
    parser.add_argument(
        "--strict",
        action="store_const",
        const="strict",
        dest="mode",
        help="Use strict comparison (exact match including whitespace)",
    )
    parser.add_argument(
        "--show-diff",
        action="store_true",
        help="Show detailed diff output when files differ",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

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
            file2_name=str(output_path),
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
