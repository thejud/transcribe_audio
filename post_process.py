#!/usr/bin/env python3

"""
Post-Processing Tool for Transcript Formatting and Voice Memo Summarization

This script takes a text transcript file and uses OpenAI's chat models to
either reformat the text into natural paragraphs while preserving all original
text and wording exactly, or to summarize and clean up rambling voice memos.

Features:
- Support for gpt-4o, gpt-4o-mini, gpt-4.1-nano, gpt-4.1-mini, and gpt-4.1 models
- Two modes: transcript formatting and memo summarization
- Custom prompt support for specialized processing
- Preserves original text while improving readability (transcript mode)
- Summarizes and cleans up rambling voice memos (memo mode)
- Simple command-line interface
- Cost-effective model selection options

Usage:
    # Basic usage with default model (gpt-4o-mini)
    python post_process.py transcript.txt

    # Voice memo summarization mode
    python post_process.py memo.txt --memo

    # Custom prompt
    python post_process.py transcript.txt --prompt "Your custom instructions"

    # Use gpt-4o model
    python post_process.py transcript.txt --4o

    # Save to specific output directory
    python post_process.py transcript.txt --output-dir processed/

    # Add suffix to output filename
    python post_process.py transcript.txt --extension "_formatted"

    # Overwrite input file (use with caution)
    python post_process.py transcript.txt --inplace

    # Enable debug logging
    python post_process.py transcript.txt --debug

Environment Variables:
    OPENAI_API_KEY: Required OpenAI API key

Author: Jud Dagnall (with Claude Code)
Version: 2.0
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Default prompts
TRANSCRIPT_PROMPT = """
Reformat this transcript chunk into natural paragraphs while preserving 
all original text and wording exactly. Only add paragraph breaks where they 
improve readability. Do not change, add, or remove any words - just improve 
the paragraph structure.
"""

MEMO_PROMPT = """
Clean up and summarize this voice memo transcript. The original is likely 
rambling and stream-of-consciousness.
0. Add a 3-8 word summary title as the first line. Enclose it in an XML <title>tag.
1. Remove filler words, repetitions, and false starts
2. Organize the thoughts into coherent paragraphs
3. Preserve the main ideas and important details
4. Make it more concise while keeping the essential meaning
5. Use clear, natural language
The result should be a cleaned-up, more organized version of the original thoughts.
"""

import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI


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


def load_environment() -> Tuple[OpenAI, AsyncOpenAI]:
    """
    Load environment variables from .env file and return OpenAI clients.

    Returns:
        Tuple[OpenAI, AsyncOpenAI]: Configured OpenAI client instances

    Raises:
        SystemExit: If OPENAI_API_KEY is not found in environment
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    return OpenAI(api_key=api_key), AsyncOpenAI(api_key=api_key)


def read_transcript_file(file_path: Path) -> str:
    """
    Read the transcript text from a file.

    Args:
        file_path: Path to the transcript file

    Returns:
        str: Content of the transcript file

    Raises:
        SystemExit: If file cannot be read
    """
    if not file_path.exists():
        logging.error(f"Transcript file not found: {file_path}")
        raise FileNotFoundError(f"Transcript file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            logging.error(f"Transcript file is empty: {file_path}")
            raise ValueError(f"Transcript file is empty: {file_path}")

        logging.info(f"Read transcript file: {file_path} ({len(content)} characters)")
        return content

    except Exception as e:
        logging.error(f"Error reading transcript file {file_path}: {str(e)}")
        raise


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for
        model: Model name for encoding selection

    Returns:
        int: Number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to a common encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def get_cache_path(transcript_path: Path) -> Path:
    """
    Get cache directory path for a transcript file.

    Args:
        transcript_path: Path to the transcript file

    Returns:
        Path: Cache directory path
    """
    cache_dir = transcript_path.parent / ".post_process_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def get_chunk_cache_key(
    chunk_text: str, model: str, context: str = "", prompt: str = ""
) -> str:
    """
    Generate cache key for a chunk.

    Args:
        chunk_text: The chunk text
        model: Model name
        context: Context text
        prompt: Custom prompt

    Returns:
        str: MD5 hash of the chunk + model + context + prompt
    """
    content = f"{chunk_text}|{model}|{context}|{prompt}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def load_chunk_from_cache(cache_path: Path, cache_key: str) -> Optional[str]:
    """
    Load processed chunk from cache.

    Args:
        cache_path: Cache directory path
        cache_key: Cache key for the chunk

    Returns:
        Optional[str]: Cached chunk content or None if not found
    """
    cache_file = cache_path / f"{cache_key}.txt"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logging.warning(f"Failed to load cache file {cache_file}: {e}")
    return None


def save_chunk_to_cache(cache_path: Path, cache_key: str, content: str) -> None:
    """
    Save processed chunk to cache.

    Args:
        cache_path: Cache directory path
        cache_key: Cache key for the chunk
        content: Processed chunk content
    """
    cache_file = cache_path / f"{cache_key}.txt"
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        logging.warning(f"Failed to save cache file {cache_file}: {e}")


def split_into_chunks(
    text: str, chunk_size: int = 4000, overlap: int = 300, model: str = "gpt-4o"
) -> List[Tuple[str, int, int]]:
    """
    Split text into overlapping chunks based on token count.

    Args:
        text: Text to split
        chunk_size: Maximum tokens per chunk
        overlap: Overlap tokens between chunks
        model: Model name for token counting

    Returns:
        List[Tuple[str, int, int]]: List of (chunk_text, start_pos, end_pos)
    """
    # Split by sentences for better chunk boundaries
    sentences = [s.strip() + ". " for s in text.split(". ") if s.strip()]
    if not sentences:
        return [(text, 0, len(text))]

    chunks = []
    current_chunk = ""
    current_tokens = 0
    sentence_start = 0

    for i, sentence in enumerate(sentences):
        sentence_tokens = count_tokens(sentence, model)

        # If adding this sentence would exceed chunk size, start a new chunk
        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            # Find the overlap starting point
            overlap_text = ""
            overlap_tokens = 0
            overlap_sentences = []

            # Add sentences from the end of current chunk for overlap
            for j in range(len(sentences) - 1, -1, -1):
                if j < sentence_start:
                    break
                overlap_candidate = sentences[j] + overlap_text
                overlap_candidate_tokens = count_tokens(overlap_candidate, model)
                if overlap_candidate_tokens <= overlap:
                    overlap_text = overlap_candidate
                    overlap_tokens = overlap_candidate_tokens
                    overlap_sentences.insert(0, j)
                else:
                    break

            # Store the current chunk
            chunk_start = sum(len(sentences[k]) for k in range(sentence_start))
            chunk_end = chunk_start + len(current_chunk)
            chunks.append((current_chunk.strip(), chunk_start, chunk_end))

            # Start new chunk with overlap
            if overlap_sentences:
                current_chunk = overlap_text
                current_tokens = overlap_tokens
                sentence_start = overlap_sentences[0]
            else:
                current_chunk = sentence
                current_tokens = sentence_tokens
                sentence_start = i
        else:
            # Add sentence to current chunk
            current_chunk += sentence
            current_tokens += sentence_tokens

    # Add the final chunk
    if current_chunk:
        chunk_start = sum(len(sentences[k]) for k in range(sentence_start))
        chunk_end = chunk_start + len(current_chunk)
        chunks.append((current_chunk.strip(), chunk_start, chunk_end))

    logging.info(
        f"Split text into {len(chunks)} chunks (avg {current_tokens//len(chunks) if chunks else 0} tokens/chunk)"
    )
    return chunks


def get_default_prompt(mode: str = "transcript") -> str:
    """
    Get the default prompt for a given processing mode.

    Args:
        mode: Processing mode ('transcript' or 'memo')

    Returns:
        str: Default system prompt for the mode
    """
    if mode == "memo":
        return MEMO_PROMPT.strip()
    else:  # transcript mode
        return TRANSCRIPT_PROMPT.strip()


async def reformat_chunk_async(
    chunk_text: str,
    client: AsyncOpenAI,
    model: str = "gpt-4.1-nano",
    context: str = "",
    chunk_idx: int = 0,
    custom_prompt: str = "",
    mode: str = "transcript",
) -> str:
    """
    Use OpenAI to reformat a text chunk into natural paragraphs (async version).

    Args:
        chunk_text: The chunk text to reformat
        client: AsyncOpenAI client instance
        model: The chat model to use for reformatting
        context: Previous context for continuity
        chunk_idx: Index of the chunk for logging
        custom_prompt: Custom system prompt (overrides default)
        mode: Processing mode ('transcript' or 'memo')

    Returns:
        str: Reformatted chunk with improved paragraph structure

    Raises:
        Exception: If the API call fails
    """
    system_prompt = custom_prompt if custom_prompt else get_default_prompt(mode)

    user_content = chunk_text
    if context:
        user_content = f"Previous context (for continuity only, do not reformat): {context}\n\nText to reformat: {chunk_text}"

    try:
        logging.debug(f"Processing chunk {chunk_idx + 1} with model: {model}")
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,  # Low temperature for consistent formatting
        )

        reformatted_text = response.choices[0].message.content.strip()
        logging.debug(f"Successfully reformatted chunk {chunk_idx + 1}")
        return reformatted_text

    except Exception as e:
        logging.error(f"Error calling OpenAI API for chunk {chunk_idx + 1}: {str(e)}")
        raise


async def reformat_transcript_chunked(
    transcript: str,
    client: AsyncOpenAI,
    model: str = "gpt-4.1-nano",
    chunk_size: int = 4000,
    overlap: int = 300,
    max_concurrent: int = 3,
    transcript_path: Optional[Path] = None,
    use_cache: bool = True,
    custom_prompt: str = "",
    mode: str = "transcript",
) -> str:
    """
    Use OpenAI to reformat transcript into natural paragraphs using chunked processing.

    Args:
        transcript: The original transcript text
        client: AsyncOpenAI client instance
        model: The chat model to use for reformatting
        chunk_size: Maximum tokens per chunk
        overlap: Overlap tokens between chunks
        max_concurrent: Maximum concurrent API calls
        transcript_path: Path to transcript file for caching
        use_cache: Whether to use caching
        custom_prompt: Custom system prompt (overrides default)
        mode: Processing mode ('transcript' or 'memo')

    Returns:
        str: Reformatted transcript with improved paragraph structure

    Raises:
        SystemExit: If the processing fails
    """
    try:
        # Check if the transcript is small enough to process as a single chunk
        total_tokens = count_tokens(transcript, model)
        logging.info(f"Total transcript tokens: {total_tokens}")

        if total_tokens <= chunk_size:
            logging.info("Transcript small enough for single-chunk processing")
            response = await reformat_chunk_async(
                transcript, client, model, "", 0, custom_prompt, mode
            )
            return response

        # Split into chunks
        chunks = split_into_chunks(transcript, chunk_size, overlap, model)
        logging.info(
            f"Processing {len(chunks)} chunks with max {max_concurrent} concurrent requests"
        )

        # Set up caching
        cache_path = None
        if use_cache and transcript_path:
            cache_path = get_cache_path(transcript_path)
            logging.info(f"Using cache directory: {cache_path}")

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_chunk_with_semaphore(
            chunk_info: Tuple[str, int, int], idx: int
        ) -> Tuple[int, str]:
            chunk_text, start_pos, end_pos = chunk_info

            # Extract context from previous chunk if available
            context = ""
            if idx > 0:
                prev_chunk_text = chunks[idx - 1][0]
                # Get last 2 sentences as context
                context_sentences = prev_chunk_text.split(". ")[-2:]
                context = ". ".join(context_sentences).strip()
                if context and not context.endswith("."):
                    context += "."

            # Check cache first
            cached_result = None
            cache_key = None
            if cache_path:
                cache_key = get_chunk_cache_key(
                    chunk_text, model, context, custom_prompt
                )
                cached_result = load_chunk_from_cache(cache_path, cache_key)
                if cached_result:
                    logging.debug(f"Using cached result for chunk {idx + 1}")
                    return idx, cached_result

            async with semaphore:
                # Add small delay between requests to respect rate limits
                if idx > 0:
                    await asyncio.sleep(0.1)
                reformatted = await reformat_chunk_async(
                    chunk_text, client, model, context, idx, custom_prompt, mode
                )

                # Save to cache
                if cache_path and cache_key:
                    save_chunk_to_cache(cache_path, cache_key, reformatted)

                return idx, reformatted

        # Process chunks concurrently
        tasks = [
            process_chunk_with_semaphore(chunk_info, idx)
            for idx, chunk_info in enumerate(chunks)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions
        failed_chunks = []
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Chunk {i + 1} failed: {str(result)}")
                failed_chunks.append(i)
            else:
                successful_results.append(result)

        if failed_chunks:
            logging.error(f"Failed to process {len(failed_chunks)} chunks")
            raise RuntimeError(f"Failed to process {len(failed_chunks)} chunks")

        # Sort results by chunk index and combine
        successful_results.sort(key=lambda x: x[0])
        reformatted_chunks = [result[1] for result in successful_results]

        # Combine chunks, removing duplicate content from overlaps
        final_text = reformatted_chunks[0] if reformatted_chunks else ""

        for i in range(1, len(reformatted_chunks)):
            # Simple heuristic: try to find overlap and remove it
            current_chunk = reformatted_chunks[i]

            # Look for sentences that might be duplicated
            prev_sentences = final_text.split(". ")[
                -3:
            ]  # Last 3 sentences from previous
            current_sentences = current_chunk.split(". ")[
                :3
            ]  # First 3 sentences from current

            # Find overlap and remove it
            overlap_found = False
            for j in range(len(prev_sentences)):
                for k in range(len(current_sentences)):
                    if (
                        prev_sentences[j].strip()
                        and prev_sentences[j] in current_sentences[k]
                    ):
                        # Remove overlapping content
                        remaining_sentences = current_chunk.split(". ")[k + 1 :]
                        current_chunk = ". ".join(remaining_sentences)
                        if current_chunk and not current_chunk.startswith("."):
                            current_chunk = ". " + current_chunk
                        overlap_found = True
                        break
                if overlap_found:
                    break

            final_text += current_chunk

        logging.info("Successfully combined all reformatted chunks")
        return final_text.strip()

    except Exception as e:
        logging.error(f"Error in chunked processing: {str(e)}")
        raise


def normalize_whitespace_for_verification(text: str) -> str:
    """
    Normalize whitespace in text for integrity verification.

    Args:
        text: Input text to normalize

    Returns:
        str: Text with normalized whitespace
    """
    # Replace all whitespace characters with single spaces
    normalized = re.sub(r"\s+", " ", text)
    # Strip leading and trailing whitespace
    return normalized.strip()


def verify_text_integrity(
    original_text: str, processed_text: str, mode: str = "normalized"
) -> bool:
    """
    Verify that processed text preserves the original content.

    Args:
        original_text: Original input text
        processed_text: Processed output text
        mode: Comparison mode ('strict', 'normalized', 'word-only')

    Returns:
        bool: True if texts are considered identical according to mode
    """
    if mode == "strict":
        return original_text == processed_text
    elif mode == "normalized":
        return normalize_whitespace_for_verification(
            original_text
        ) == normalize_whitespace_for_verification(processed_text)
    elif mode == "word-only":
        # Extract only words for comparison
        original_words = re.findall(r"\w+", original_text.lower())
        processed_words = re.findall(r"\w+", processed_text.lower())
        return original_words == processed_words
    else:
        raise ValueError(f"Unknown verification mode: {mode}")


def reformat_transcript(
    transcript: str,
    client: OpenAI,
    model: str = "gpt-4.1-nano",
    custom_prompt: str = "",
    mode: str = "transcript",
) -> str:
    """
    Use OpenAI to reformat transcript into natural paragraphs (legacy single-chunk version).

    Args:
        transcript: The original transcript text
        client: OpenAI client instance
        model: The chat model to use for reformatting
        custom_prompt: Custom system prompt (overrides default)
        mode: Processing mode ('transcript' or 'memo')

    Returns:
        str: Reformatted transcript with improved paragraph structure

    Raises:
        Exception: If the API call fails
    """
    prompt = custom_prompt if custom_prompt else get_default_prompt(mode)

    try:
        logging.info(f"Sending transcript to OpenAI using model: {model}")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": transcript},
            ],
            temperature=0.1,  # Low temperature for consistent formatting
        )

        reformatted_text = response.choices[0].message.content.strip()
        logging.info("Successfully reformatted transcript")
        return reformatted_text

    except Exception as e:
        logging.error(f"Error calling OpenAI API: {str(e)}")
        raise


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Reformat transcript text into natural paragraphs using OpenAI",
        epilog="""
Simple Examples:
  %(prog)s transcript.txt                          # Basic transcript formatting
  %(prog)s memo.txt --memo                         # Voice memo summarization (recommended)
  %(prog)s transcript.txt --prompt "Custom prompt" # Use custom prompt
  %(prog)s transcript.txt -O processed/            # Save to specific directory
  %(prog)s transcript.txt --inplace               # Overwrite input file

For streamlined voice memo processing, consider using: python pipeline.py audio/*.mp3

Advanced Examples (use --advanced for full options):
  %(prog)s transcript.txt --advanced --4o --verify --chunk-size 5000
  %(prog)s transcript.txt --advanced --no-chunking --verify-mode strict
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional arguments
    parser.add_argument(
        "transcript_file", help="Path to the transcript file to reformat"
    )

    # Core options (always visible)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Processing mode and prompt options
    parser.add_argument(
        "--memo",
        action="store_true",
        help="Voice memo mode: summarize and clean up rambling transcripts",
    )
    parser.add_argument(
        "--prompt",
        help="Custom system prompt to use instead of default",
    )

    # Output options
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-o",
        "--output",
        help="Output file path",
    )
    output_group.add_argument(
        "-O",
        "--output-dir",
        help="Output directory for result files (default: same directory as input)",
    )
    output_group.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file with processed result",
    )

    # Extension option
    parser.add_argument(
        "-E",
        "--extension",
        help="Suffix to add to output filename before the file extension (e.g., '_processed')",
    )

    # Advanced mode toggle
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Enable advanced options (models, chunking, verification, testing)",
    )

    # Advanced options (only shown with --advanced)
    if "--advanced" in sys.argv or "-h" in sys.argv or "--help" in sys.argv:
        # Model selection options (advanced)
        model_group = parser.add_mutually_exclusive_group()
        model_group.add_argument(
            "--model",
            default="gpt-4.1-nano",
            choices=[
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4.1-nano",
                "gpt-4.1-mini",
                "gpt-4.1",
            ],
            help="OpenAI chat model to use (default: gpt-4.1-nano)",
        )
        model_group.add_argument(
            "--4o",
            action="store_const",
            const="gpt-4o",
            dest="model",
            help="Use gpt-4o model",
        )
        model_group.add_argument(
            "--mini",
            action="store_const",
            const="gpt-4o-mini",
            dest="model",
            help="Use gpt-4o-mini model",
        )
        model_group.add_argument(
            "--nano",
            action="store_const",
            const="gpt-4.1-nano",
            dest="model",
            help="Use gpt-4.1-nano model (fastest, most cost-effective)",
        )
        model_group.add_argument(
            "--4.1-mini",
            action="store_const",
            const="gpt-4.1-mini",
            dest="model",
            help="Use gpt-4.1-mini model",
        )
        model_group.add_argument(
            "--4.1",
            action="store_const",
            const="gpt-4.1",
            dest="model",
            help="Use gpt-4.1 model",
        )

        # Chunking options (advanced)
        parser.add_argument(
            "--no-chunking",
            action="store_true",
            help="Disable chunking and process entire transcript at once (legacy mode)",
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=4000,
            help="Maximum tokens per chunk (default: 4000)",
        )
        parser.add_argument(
            "--overlap",
            type=int,
            default=300,
            help="Overlap tokens between chunks (default: 300)",
        )
        parser.add_argument(
            "--max-concurrent",
            type=int,
            default=3,
            help="Maximum concurrent API requests (default: 3)",
        )
        parser.add_argument(
            "--no-cache",
            action="store_true",
            help="Disable caching of processed chunks",
        )

        # Verification options (advanced)
        parser.add_argument(
            "--verify",
            action="store_true",
            help="Verify text integrity before outputting results",
        )
        parser.add_argument(
            "--verify-mode",
            choices=["strict", "normalized", "word-only"],
            default="normalized",
            help="Verification mode (default: normalized)",
        )

        # Test mode options (advanced)
        parser.add_argument(
            "--test-mode",
            action="store_true",
            help="Enable test mode with smaller chunks and limited input",
        )
        parser.add_argument(
            "--test-limit",
            type=int,
            default=2000,
            help="Maximum characters to process in test mode (default: 2000)",
        )

    # Parse arguments
    args = parser.parse_args()

    # Set default values for advanced options if not in advanced mode
    if not args.advanced:
        args.model = "gpt-4.1-nano"
        args.no_chunking = False
        args.chunk_size = 4000
        args.overlap = 300
        args.max_concurrent = 3
        args.no_cache = False
        args.verify = False
        args.verify_mode = "normalized"
        args.test_mode = False
        args.test_limit = 2000
    elif not hasattr(args, "model") or args.model is None:
        args.model = "gpt-4.1-nano"

    return args


async def main_async() -> None:
    """
    Async main function to handle command line arguments and process transcript.

    This function sets up argument parsing, loads environment configuration,
    reads the transcript file, sends it to OpenAI for reformatting, and
    outputs the result to stdout.
    """
    args = parse_args()

    # Set up logging and load environment
    setup_logging(args.debug)
    sync_client, async_client = load_environment()

    if not sync_client:
        raise RuntimeError("OPENAI_API_KEY not found in environment variables")

    # Read transcript file
    transcript_path = Path(args.transcript_file)
    transcript_text = read_transcript_file(transcript_path)

    # Apply test mode limits if enabled
    if args.test_mode:
        original_length = len(transcript_text)
        transcript_text = transcript_text[: args.test_limit]
        logging.info(
            f"Test mode: Processing {len(transcript_text)} characters (reduced from {original_length})"
        )

    # Determine processing mode and prompt
    mode = "memo" if args.memo else "transcript"
    custom_prompt = args.prompt or ""

    logging.info(f"Processing mode: {mode}")
    if custom_prompt:
        logging.info("Using custom prompt")

    # Reformat transcript using OpenAI
    # Adjust chunk sizes for test mode
    chunk_size = args.chunk_size
    if args.test_mode:
        chunk_size = min(args.chunk_size, 1000)  # Smaller chunks in test mode
        logging.info(f"Test mode: Using chunk size of {chunk_size}")

    if args.no_chunking:
        logging.info("Using legacy single-chunk processing")
        reformatted_text = reformat_transcript(
            transcript_text, sync_client, args.model, custom_prompt, mode
        )
    else:
        logging.info("Using optimized chunked processing")
        reformatted_text = await reformat_transcript_chunked(
            transcript_text,
            async_client,
            args.model,
            chunk_size,
            args.overlap,
            args.max_concurrent,
            transcript_path,
            not args.no_cache,
            custom_prompt,
            mode,
        )

    # Verify text integrity if requested
    if args.verify:
        logging.info(f"Verifying text integrity using mode: {args.verify_mode}")
        integrity_check = verify_text_integrity(
            transcript_text, reformatted_text, args.verify_mode
        )

        if integrity_check:
            logging.info("✓ Text integrity verification PASSED")
        else:
            # Print error details to stderr
            print("\n✗ TEXT INTEGRITY VERIFICATION FAILED", file=sys.stderr)
            print(
                "The processed text does not preserve the original content according to the verification criteria.",
                file=sys.stderr,
            )
            print(f"Verification mode: {args.verify_mode}", file=sys.stderr)
            print("\nProcessed output (for debugging):", file=sys.stderr)
            print(reformatted_text, file=sys.stderr)
            raise ValueError("Text integrity verification failed")

    # Determine output handling
    if args.output:
        # Explicit output file specified
        try:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(reformatted_text)
            logging.info(f"Output written to {output_path}")
        except Exception as e:
            logging.error(f"Failed to write output file {args.output}: {str(e)}")
            raise
    elif args.inplace:
        # Overwrite input file (only on successful processing)
        try:
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(reformatted_text)
            logging.info(f"Input file overwritten: {transcript_path}")
        except Exception as e:
            logging.error(f"Failed to overwrite input file {transcript_path}: {str(e)}")
            raise
    elif args.output_dir or args.extension:
        # Generate output file with directory and/or extension
        try:
            # Determine output directory
            if args.output_dir:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = transcript_path.parent

            # Generate filename with optional extension suffix
            base_name = transcript_path.stem
            file_ext = transcript_path.suffix

            if args.extension:
                output_filename = f"{base_name}{args.extension}{file_ext}"
            else:
                output_filename = f"{base_name}{file_ext}"

            output_path = output_dir / output_filename

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(reformatted_text)
            logging.info(f"Output written to {output_path}")
        except Exception as e:
            logging.error(f"Failed to write output file: {str(e)}")
            raise
    else:
        # Output to stdout
        print(reformatted_text)


def main() -> None:
    """
    Main function wrapper to run the async main function.
    """
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
