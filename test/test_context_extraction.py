#!/usr/bin/env python3

"""
Test Context Extraction from Voice Memo Transcripts

This script tests the context extraction functionality of the post-processing
system with various sample transcripts to evaluate the effectiveness of the
current prompt and potential improvements.

Usage:
    python test_context_extraction.py
    python test_context_extraction.py --prompt-variation 1
    python test_context_extraction.py --all-variations
"""

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path to import post_process
sys.path.insert(0, str(Path(__file__).parent.parent))

from post_process import (
    AsyncOpenAI,
    load_environment,
    reformat_chunk_async,
    setup_logging,
)


# Define prompt variations to test
PROMPT_VARIATIONS = {
    "current": """Clean up and summarize this voice memo transcript. The original is likely 
rambling and stream-of-consciousness.
1. Add a 3-8 word summary title as the first line. Enclose it in an XML <title>tag.
2. There may be a "context" hint near the beginning of the transcript that describes a context for the entry.
   If present, extract this context and insert it inside an XML <context> tag.
3. Remove filler words, repetitions, and false starts
4. Organize the thoughts into coherent paragraphs
5. Preserve the main ideas and important details
6. Make it more concise while keeping the essential meaning
7. Use clear, natural language
The result should be a cleaned-up, more organized version of the original thoughts.""",
    
    "improved_v1": """Clean up and summarize this voice memo transcript. The original is likely 
rambling and stream-of-consciousness.
1. Add a 3-8 word summary title as the first line. Enclose it in an XML <title>tag.
2. Check if there's context information at the very beginning of the transcript:
   - Look for "context" followed by a description
   - Look for "begin context"..."end context" markers
   - Only extract if clearly marked at the start
   - If no clear context marker is found, do not add a <context> tag
   - Extract the exact context phrase, not a paraphrase
3. Remove filler words, repetitions, and false starts
4. Organize the thoughts into coherent paragraphs
5. Preserve the main ideas and important details
6. Make it more concise while keeping the essential meaning
7. Use clear, natural language
The result should be a cleaned-up, more organized version of the original thoughts.""",
    
    "improved_v2": """Clean up and summarize this voice memo transcript. The original is likely 
rambling and stream-of-consciousness.
1. Add a 3-8 word summary title as the first line. Enclose it in an XML <title>tag.
2. Context extraction rules:
   - ONLY add a <context> tag if you find explicit context markers in the first ~50 words
   - Look for: "context" followed by description, or "begin context"/"end context" markers
   - Extract the exact words after "context", do not paraphrase or summarize
   - If the transcript starts with "this is about" extract that as context
   - If no explicit context marker exists, do NOT create a <context> tag
3. Remove filler words, repetitions, and false starts
4. Organize the thoughts into coherent paragraphs
5. Preserve the main ideas and important details
6. Make it more concise while keeping the essential meaning
7. Use clear, natural language
The result should be a cleaned-up, more organized version of the original thoughts.""",
    
    "improved_v3": """Clean up and summarize this voice memo transcript. The original is likely 
rambling and stream-of-consciousness.
1. Add a 3-8 word summary title as the first line. Enclose it in an XML <title>tag.
2. Context extraction (BE STRICT):
   - Search ONLY the first 100 characters for context indicators
   - If you find "context" followed by words, extract those exact words
   - If you find "begin context", extract everything until "end context"  
   - If the very first words are "this is about", extract what follows
   - Otherwise, DO NOT add any <context> tag - many memos have no context
   - Never paraphrase - use the exact words from the transcript
3. Remove filler words, repetitions, and false starts
4. Organize the thoughts into coherent paragraphs
5. Preserve the main ideas and important details
6. Make it more concise while keeping the essential meaning
7. Use clear, natural language
The result should be a cleaned-up, more organized version of the original thoughts.""",
    
    "improved_v4": """Clean up and summarize this voice memo transcript. The original is likely 
rambling and stream-of-consciousness.
1. Add a 3-8 word summary title as the first line. Enclose it in an XML <title>tag.
2. Context extraction rules (FOLLOW EXACTLY):
   - Look at the VERY FIRST WORDS of the transcript only
   - If it starts with "context" followed by words → extract those exact words in <context> tag
   - If it starts with "begin context" → extract until "end context" in <context> tag
   - If it starts with "this is about" → extract what follows in <context> tag
   - If it starts with anything else → DO NOT add a <context> tag at all
   - NEVER create context from the middle or end of the transcript
   - NEVER paraphrase - use exact words only
3. Remove filler words, repetitions, and false starts
4. Organize the thoughts into coherent paragraphs
5. Preserve the main ideas and important details
6. Make it more concise while keeping the essential meaning
7. Use clear, natural language
The result should be a cleaned-up, more organized version of the original thoughts.""",
    
    "improved_v5": """Clean up and summarize this voice memo transcript. The original is likely 
rambling and stream-of-consciousness.
1. Add a 3-8 word summary title as the first line. Enclose it in an XML <title>tag.
2. Context extraction (CRITICAL RULES):
   - ONLY if the transcript literally starts with these exact phrases:
     • "context [description]" → extract the description part
     • "begin context [text] end context" → extract the text between markers  
     • "this is about [topic]" → extract the topic part
   - If the transcript starts with ANY OTHER WORDS (like "so", "I", "yesterday", etc.), absolutely DO NOT add a <context> tag
   - Most voice memos have NO context - this is normal and expected
   - If in doubt, skip the context tag entirely
3. Remove filler words, repetitions, and false starts
4. Organize the thoughts into coherent paragraphs
5. Preserve the main ideas and important details
6. Make it more concise while keeping the essential meaning
7. Use clear, natural language
The result should be a cleaned-up, more organized version of the original thoughts.""",
    "enhanced": """Clean up and summarize this voice memo transcript. The original is likely 
rambling and stream-of-consciousness.
1. Add a 3-8 word summary title as the first line. Enclose it in an XML <title>tag.
2. Look for context information at the beginning of the transcript. This might be:
   - Explicitly labeled with "context" or similar
   - An opening statement that sets the topic/project/subject
   - Information between markers like "begin context" and "end context"
   Extract this and insert it inside an XML <context> tag.
3. Remove filler words, repetitions, and false starts
4. Organize the thoughts into coherent paragraphs
5. Preserve the main ideas and important details
6. Make it more concise while keeping the essential meaning
7. Use clear, natural language
The result should be a cleaned-up, more organized version of the original thoughts.""",
    "flexible": """Clean up and summarize this voice memo transcript. The original is likely 
rambling and stream-of-consciousness.
1. Add a 3-8 word summary title as the first line. Enclose it in an XML <title>tag.
2. Extract contextual information from the beginning of the transcript:
   - Check the first 1-3 sentences for topic/project/subject information
   - Look for patterns like "context", "this is about", "regarding", etc.
   - If found, extract and insert inside an XML <context> tag
   - Ignore later mentions of the word "context" in the body
3. Remove filler words, repetitions, and false starts
4. Organize the thoughts into coherent paragraphs
5. Preserve the main ideas and important details
6. Make it more concise while keeping the essential meaning
7. Use clear, natural language
The result should be a cleaned-up, more organized version of the original thoughts.""",
    "structured": """Clean up and summarize this voice memo transcript. The original is likely 
rambling and stream-of-consciousness.
1. Add a 3-8 word summary title as the first line. Enclose it in an XML <title>tag.
2. Context extraction:
   - Primary: Look for "context" followed by description in the first ~100 words
   - Secondary: Check for "begin context" ... "end context" markers
   - Fallback: Extract topic from first sentence if it clearly sets the subject
   Insert any found context inside an XML <context> tag.
3. Remove filler words, repetitions, and false starts
4. Organize the thoughts into coherent paragraphs
5. Preserve the main ideas and important details
6. Make it more concise while keeping the essential meaning
7. Use clear, natural language
The result should be a cleaned-up, more organized version of the original thoughts.""",
}


def extract_context_from_output(output: str) -> Optional[str]:
    """Extract context from the processed output."""
    match = re.search(r"<context>(.*?)</context>", output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_title_from_output(output: str) -> Optional[str]:
    """Extract title from the processed output."""
    match = re.search(r"<title>(.*?)</title>", output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


async def test_sample(
    sample_path: Path, client: AsyncOpenAI, prompt: str, model: str = "gpt-4.1-nano"
) -> Dict[str, str]:
    """Test a single sample with a given prompt."""
    # Read the sample
    with open(sample_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Process with the given prompt
    try:
        result = await reformat_chunk_async(
            chunk_text=content,
            client=client,
            model=model,
            context="",
            chunk_idx=0,
            custom_prompt=prompt,
            mode="memo",
        )

        return {
            "sample": sample_path.name,
            "output": result,
            "context": extract_context_from_output(result),
            "title": extract_title_from_output(result),
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "sample": sample_path.name,
            "output": None,
            "context": None,
            "title": None,
            "success": False,
            "error": str(e),
        }


async def test_all_samples(
    prompt_name: str, prompt: str, client: AsyncOpenAI, model: str = "gpt-4.1-nano"
) -> List[Dict[str, str]]:
    """Test all samples with a given prompt."""
    samples_dir = Path(__file__).parent / "samples"
    sample_files = sorted(samples_dir.glob("context_test_*.txt"))

    if not sample_files:
        logging.error(f"No sample files found in {samples_dir}")
        return []

    logging.info(f"Testing {len(sample_files)} samples with prompt: {prompt_name}")

    # Run tests concurrently
    tasks = [
        test_sample(sample_path, client, prompt, model) for sample_path in sample_files
    ]

    results = await asyncio.gather(*tasks)

    # Add prompt name to results
    for result in results:
        result["prompt_name"] = prompt_name

    return results


def analyze_results(results: List[Dict[str, str]]) -> None:
    """Analyze and display test results."""
    print("\n" + "=" * 80)
    print("CONTEXT EXTRACTION TEST RESULTS")
    print("=" * 80)

    # Group by prompt
    by_prompt = {}
    for result in results:
        prompt_name = result["prompt_name"]
        if prompt_name not in by_prompt:
            by_prompt[prompt_name] = []
        by_prompt[prompt_name].append(result)

    # Expected contexts for each test case
    expected_contexts = {
        "context_test_explicit.txt": "AI programming Claude agents maintaining proper context in sessions",
        "context_test_implicit.txt": "new deployment pipeline for the mobile app",
        "context_test_markers.txt": "project alpha Q4 planning meeting followup budget constraints and timeline adjustments",
        "context_test_multiple.txt": "machine learning model optimization for edge devices",
        "context_test_no_context.txt": None,
    }

    # Analyze each prompt's performance
    for prompt_name, prompt_results in by_prompt.items():
        print(f"\n### Prompt: {prompt_name}")
        print("-" * 40)

        success_count = 0
        correct_extractions = 0

        for result in prompt_results:
            sample_name = result["sample"]
            expected = expected_contexts.get(sample_name)
            extracted = result["context"]

            print(f"\nSample: {sample_name}")

            if result["success"]:
                print(f"Title: {result['title']}")
                print(f"Context extracted: {extracted if extracted else 'None'}")

                # Check if extraction matches expectation
                if expected is None and extracted is None:
                    correct_extractions += 1
                    print("✓ Correctly identified no context")
                elif expected and extracted:
                    # Fuzzy matching - check if key terms are present
                    expected_lower = expected.lower()
                    extracted_lower = extracted.lower()
                    key_terms_match = all(
                        term in extracted_lower
                        for term in expected_lower.split()
                        if len(term) > 3  # Skip short words
                    )
                    if key_terms_match:
                        correct_extractions += 1
                        print("✓ Context correctly extracted")
                    else:
                        print(
                            f"✗ Context mismatch - Expected key terms from: {expected}"
                        )
                else:
                    print(f"✗ Context extraction failed - Expected: {expected}")

                success_count += 1
            else:
                print(f"✗ Processing failed: {result['error']}")

        # Summary statistics
        total = len(prompt_results)
        print(f"\n{'-'*40}")
        print(f"Success rate: {success_count}/{total} ({success_count/total*100:.1f}%)")
        print(
            f"Correct extractions: {correct_extractions}/{total} ({correct_extractions/total*100:.1f}%)"
        )


async def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="Test context extraction from voice memo transcripts"
    )
    parser.add_argument(
        "--prompt-variation",
        choices=list(PROMPT_VARIATIONS.keys()),
        help="Test a specific prompt variation",
    )
    parser.add_argument(
        "--all-variations", action="store_true", help="Test all prompt variations"
    )
    parser.add_argument(
        "--model", default="gpt-4.1-nano", help="Model to use for testing"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Setup
    setup_logging(args.debug)
    _, async_client = load_environment()

    # Determine which prompts to test
    if args.all_variations:
        prompts_to_test = list(PROMPT_VARIATIONS.items())
    elif args.prompt_variation:
        prompts_to_test = [
            (args.prompt_variation, PROMPT_VARIATIONS[args.prompt_variation])
        ]
    else:
        prompts_to_test = [("current", PROMPT_VARIATIONS["current"])]

    # Run tests
    all_results = []
    for prompt_name, prompt in prompts_to_test:
        results = await test_all_samples(prompt_name, prompt, async_client, args.model)
        all_results.extend(results)

    # Analyze results
    analyze_results(all_results)

    # Save detailed results if testing all variations
    if args.all_variations:
        output_path = Path(__file__).parent / "context_extraction_results.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("DETAILED CONTEXT EXTRACTION TEST RESULTS\n")
            f.write("=" * 80 + "\n\n")

            for result in all_results:
                f.write(f"Prompt: {result['prompt_name']}\n")
                f.write(f"Sample: {result['sample']}\n")
                f.write(f"Success: {result['success']}\n")
                if result["success"]:
                    f.write(f"Title: {result['title']}\n")
                    f.write(f"Context: {result['context']}\n")
                    f.write(f"Full output:\n{result['output']}\n")
                else:
                    f.write(f"Error: {result['error']}\n")
                f.write("\n" + "-" * 80 + "\n\n")

        print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
