"""Prompt construction helpers for answer generation."""

# ============================================================================
# Imports
# ============================================================================

import re


# ============================================================================
# Anthropic Prompt Construction
# ============================================================================

def _normalize_query_space(text):
    return re.sub(r"\s+", " ", (text or "")).strip()











def _build_anthropic_prompt_parts(query, context, focus_instruction=None, answer_style_instruction=None, constraint_instruction=None):



    """Build (system_blocks, user_content) for Anthropic with prompt caching. System + context use cache_control."""



    system_instruction = """You are an assistant for question-answering over provided notes. Use ONLY the numbered notes below as your factual source (you may add brief connecting logic, but no outside facts).



If the notes contain at least some relevant information, give the **best partial answer you can** based only on these notes and then briefly mention what is missing.



Only if the notes are essentially unrelated to the question, say exactly: "The provided notes do not contain enough information to answer this."







Rules:



- Base every claim strictly on these notes. One sentence or bullet per idea is fine.



- Write in a clear, exam-oriented style: use bullet points (\u2022) for key points; use 2-space indented bullets for sub-points. Use **double asterisks** around important terms (diagnoses, drugs, criteria). Do not use ## for headings\u2014use a single bold line with \u25cf\x8f then bullets underneath.



- NUMBERED LISTS: When the question or notes refer to a numbered list (e.g. six childhood exanthems, 1st\xe2\u20ac\u201c6th disease, steps 1\xe2\u20ac\u201c6 of ventricular partitioning), **always** present in **strict numerical order**: 1st, then 2nd, then 3rd, then 4th, then 5th, then 6th. **Include every step or number that any note describes**\u2014if Note X explicitly mentions "3rd step" or "third step", you MUST include it in your answer and cite it [X]. Do not skip steps; do not reorder by relevance.



- INLINE CITATIONS: Cite the supporting note(s) using [N] or [N,M] where N is between 1 and the number of notes below only. Do not use citation numbers outside that range.



- At the end, on one line, list all note numbers you cited. Format: RELEVANT_NOTES: 1,3,5"""



    num_notes = context.count("Note ")  # approximate; we pass explicit N in caller if needed



    context_block = f"""Context information is below. There are notes numbered Note 1, Note 2, ... (cite only using numbers 1 to the number of notes below).



---------------------



{context}



---------------------"""



    system_blocks = [



        {"type": "text", "text": system_instruction},



        {"type": "text", "text": context_block, "cache_control": {"type": "ephemeral"}},



    ]



    user_content = f"""Given the context information and not prior knowledge, answer the question.







Question: {query}"""



    extra_instructions = [focus_instruction, answer_style_instruction, constraint_instruction]
    for instruction in extra_instructions:
        if instruction:
            user_content += "\n" + instruction



    return system_blocks, user_content
