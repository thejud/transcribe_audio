# Context Prompt Improvement Plan

## Executive Summary

This document outlines a plan to test and improve the context extraction functionality in the voice memo processing system. The goal is to enhance the system's ability to identify and extract contextual information from the beginning of voice memos, distinguishing it from later mentions of the word "context" or similar terms.

## Current Implementation Analysis

### Existing Prompt (from `post_process.py` lines 71-82)

```
Clean up and summarize this voice memo transcript. The original is likely 
rambling and stream-of-consciousness.
1. Add a 3-8 word summary title as the first line. Enclose it in an XML <title>tag.
2. There may be a "context" hint near the beginning of the transcript that describes a context for the entry.
   If present, extract this context and insert it inside an XML <context> tag.
3. Remove filler words, repetitions, and false starts
4. Organize the thoughts into coherent paragraphs
5. Preserve the main ideas and important details
6. Make it more concise while keeping the essential meaning
7. Use clear, natural language
The result should be a cleaned-up, more organized version of the original thoughts.
```

### Strengths
- Clear instructions for context extraction
- Uses XML tags for structured output
- Specifies that context should be near the beginning
- Comprehensive cleanup instructions

### Potential Weaknesses
- Relies on the word "context" being explicitly mentioned
- May miss contextual information not labeled as "context"
- No fallback for implicit context statements

## Test Scenarios

### Scenario 1: Explicit Context Label
**Input Pattern**: "Context: [topic details]. [Main content discussing context in another sense]"

**Example**:
```
Context: AI programming, Claude agents, session management. 

So I was thinking about how we maintain proper context in agent sessions. The importance of context cannot be overstated when building these systems. Each context window has limitations...
```

**Expected Output**: Should extract "AI programming, Claude agents, session management" as context.

### Scenario 2: Implicit Context Statement
**Input Pattern**: Opening statement that sets context without using the word "context"

**Example**:
```
This is about the new deployment pipeline we're building for the mobile app.

I've been considering different approaches to handle the build process. The context of our discussion yesterday made me realize...
```

**Expected Output**: Should ideally extract "new deployment pipeline for the mobile app" as context.

### Scenario 3: Marker-Based Context
**Input Pattern**: Using explicit markers like "Begin context:" and "End context"

**Example**:
```
Begin context: Project Alpha, Q4 planning, budget constraints. End context.

Alright, so based on what we discussed, I think we need to reconsider our timeline...
```

**Expected Output**: Should extract "Project Alpha, Q4 planning, budget constraints" as context.

### Scenario 4: Multiple Context References
**Input Pattern**: Context at beginning, with later unrelated uses of the word "context"

**Example**:
```
Context: Machine learning model optimization for edge devices.

I was reviewing the performance metrics and in the context of our current hardware limitations, we need to consider quantization. The broader context here is that...
```

**Expected Output**: Should only extract "Machine learning model optimization for edge devices" as the context tag.

## Proposed Prompt Variations

### Variation 1: Enhanced Pattern Recognition
```
2. Look for context information at the beginning of the transcript. This might be:
   - Explicitly labeled with "Context:" or similar
   - An opening statement that sets the topic/project/subject
   - Information between markers like "Begin context" and "End context"
   Extract this and insert it inside an XML <context> tag.
```

### Variation 2: Flexible Context Detection
```
2. Extract contextual information from the beginning of the transcript:
   - Check the first 1-3 sentences for topic/project/subject information
   - Look for patterns like "Context:", "This is about:", "Regarding:", etc.
   - If found, extract and insert inside an XML <context> tag
   - Ignore later mentions of the word "context" in the body
```

### Variation 3: Structured Markers
```
2. Context extraction:
   - Primary: Look for "Context:" followed by description
   - Secondary: Check for "Begin context" ... "End context" markers
   - Fallback: Extract topic from first sentence if it sets the subject
   Insert any found context inside an XML <context> tag.
```

## Testing Methodology

### 1. Sample Creation
- Create 10-15 sample transcripts covering all scenarios
- Include realistic filler words and rambling speech patterns
- Vary context placement and formatting

### 2. Automated Testing
- Process each sample with current prompt
- Process with each variation
- Compare outputs programmatically

### 3. Evaluation Criteria
- Accuracy of context extraction
- False positive rate (extracting non-context as context)
- Handling of edge cases
- Consistency across similar inputs

## Implementation Recommendations

### Phase 1: Testing Current Implementation
1. Create comprehensive test suite
2. Establish baseline performance metrics
3. Identify failure patterns

### Phase 2: Prompt Refinement
1. Test variations on sample set
2. Select best performing approach
3. Fine-tune based on results

### Phase 3: User Guidance
1. Document best practices for users
2. Consider adding examples to help output
3. Potentially add pre-processing to detect context patterns

## Success Metrics

1. **Precision**: 95%+ accuracy in extracting explicitly marked context
2. **Recall**: 80%+ success rate for implicit context statements
3. **Robustness**: <5% false positive rate for body "context" mentions
4. **Flexibility**: Successfully handle 3+ different context marking styles

## Next Steps

1. Create test sample files
2. Implement testing script
3. Run baseline tests with current prompt
4. Test variations and analyze results
5. Make recommendations based on findings