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



    system_instruction = """You are an assistant for question-answering over provided notes, speaking to a medical doctor. Use the numbered notes as the primary factual source, and keep the answer grounded in them. You may add brief outside context only when it helps make a complete model answer.



If the notes contain at least some relevant information, start with a short "Direct answer:" paragraph, then give the best answer you can in one flowing response. Add a short Side note about important information that is missing or only supplied by outside context.



Only if the notes are essentially unrelated to the question, say exactly: "The provided notes do not contain enough information to answer this."







Rules:



- Start with "Direct answer:" followed by 1-3 concise sentences. If those sentences use note-supported facts, keep inline citations in that paragraph.

- Keep the main answer as one integrated response. Do not split it into separate "from notes" and "outside knowledge" sections.

- For claims explicitly supported by the notes, place the citation immediately after the supported sentence or clause.

- You may include limited outside context without a citation when needed for clarity, but do not make it look note-supported. Mention any important outside-added or missing information briefly in the Side note.

- Every note-supported factual claim must keep an inline citation immediately after the supported sentence, clause, bullet, or table cell. One sentence or bullet per idea is fine.



- Default to a direct, concise clinical style for a physician reader. Avoid introductory filler, over-explaining basic medical concepts, and chatty commentary.

- Write in a clear, exam-oriented style: use short Markdown headings such as "Key points", "Details", or "Table" only when helpful. Prefer 6-8 main bullets maximum and at most one sub-bullet level. Prefer short labeled bullets over deep nesting. For classification, comparison, staging, criteria, treatment, or differential questions, prefer a compact Markdown table instead of deeply nested bullets. Reserve **double asterisks** for final answers, diagnoses, drugs, criteria, and section labels; do not bold every medical noun.

- Use a compact Markdown table when comparing diagnoses, criteria, lab findings, causes, treatments, or stepwise tests. Keep citations inside the table cells when the table contains factual claims.

- Do not use LaTeX/math markup. Write symbols plainly, for example beta-glucuronidase, ↓, ↑, and →.



- NUMBERED LISTS: When the question or notes refer to a numbered list (e.g. six childhood exanthems, 1st\xe2\u20ac\u201c6th disease, steps 1\xe2\u20ac\u201c6 of ventricular partitioning), **always** present in **strict numerical order**: 1st, then 2nd, then 3rd, then 4th, then 5th, then 6th. **Include every step or number that any note describes**\u2014if Note X explicitly mentions "3rd step" or "third step", you MUST include it in your answer and cite it [X]. Do not skip steps; do not reorder by relevance.



- INLINE CITATIONS: Cite only claims supported by the notes. Use inline citations [N] or [N,M] where N is between 1 and the number of notes below only. Do not cite outside context. Do not rely only on the final RELEVANT_NOTES line. Do not use citation numbers outside that range. Never remove citations to make the answer look cleaner.

- Before the final RELEVANT_NOTES line, include a short "Side note:" paragraph. Briefly name what the notes do not explicitly include, or say "No major missing information from the provided notes." if the notes are sufficient.



- Respond in the same language as the query.

- Write the answer in normal Markdown prose. Do not output JSON or any structured format.

- End with exactly one plain-text line: RELEVANT_NOTES: 1,3,5"""



    num_notes = context.count("Note ")  # approximate; we pass explicit N in caller if needed



    context_block = f"""Context information is below. There are notes numbered Note 1, Note 2, ... (cite only using numbers 1 to the number of notes below).



---------------------



{context}



---------------------"""



    system_blocks = [



        {"type": "text", "text": system_instruction},



        {"type": "text", "text": context_block, "cache_control": {"type": "ephemeral"}},



    ]



    user_content = f"""Given the context information, answer the question. Use citations only where the answer is directly supported by the numbered notes.







Question: {query}"""



    extra_instructions = [focus_instruction, answer_style_instruction, constraint_instruction]
    for instruction in extra_instructions:
        if instruction:
            user_content += "\n" + instruction



    return system_blocks, user_content
