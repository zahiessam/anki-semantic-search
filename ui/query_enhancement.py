"""Query expansion, HyDE generation, and relevance-mode helpers."""

# ============================================================================
# Imports
# ============================================================================

import json
import re
import urllib.request

from ..utils import load_config, log_debug
from ..utils.anthropic_response import extract_anthropic_text


# ============================================================================
# Query Enhancement And Relevance Helpers
# ============================================================================

def _expand_query(self, query, config):
    """Expand query with synonyms using AI and small deterministic fallbacks."""
    search_config = load_config().get('search_config', {})

    try:
        q_lower = (query or "").lower()
        extra_terms = []

        synonym_groups = [
            ["adrenaline", "epinephrine"],
            ["noradrenaline", "norepinephrine"],
            ["acetaminophen", "paracetamol"],
            ["pth", "parathyroid hormone"],
            ["vitamin d", "cholecalciferol", "ergocalciferol"],
        ]

        for group in synonym_groups:
            if any(term in q_lower for term in group):
                extra_terms.extend(term for term in group if term not in q_lower)

        spelling_variants = {
            "pnuemonia": "pneumonia",
            "pneumonias": "pneumonia",
        }

        for variant, canonical in spelling_variants.items():
            if (
                re.search(rf"\b{re.escape(variant)}\b", q_lower)
                and not re.search(rf"\b{re.escape(canonical)}\b", q_lower)
            ):
                extra_terms.append(canonical)

        overrides = search_config.get("synonym_overrides") or []
        for item in overrides:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                group = [str(t).strip().lower() for t in item if t and str(t).strip()]
                if len(group) < 2:
                    continue
                if any(term in q_lower for term in group):
                    extra_terms.extend(term for term in group if term not in q_lower)

        if extra_terms:
            original_query = query
            extra_terms = list(dict.fromkeys(extra_terms))
            query = f"{query} " + " ".join(extra_terms)
            log_debug(f"Built-in query alias expansion: {original_query} -> {query}")
    except Exception:
        pass

    if not search_config.get('enable_query_expansion', False):
        return query

    try:
        prompt = (
            "You are improving a medical Anki search query for retrieval.\n"
            "First infer the overall medical/search intent of the full query. "
            "Then inspect each meaningful term, acronym, abbreviation, or typo and add only expansions "
            "that preserve that whole-query intent.\n"
            "- expand medical acronyms and abbreviations\n"
            "- correct obvious spelling mistakes\n"
            "- add full forms, alternate clinical wording, and very close synonyms\n"
            "- prefer wording that could appear in matching study notes or deck cards\n"
            "- do not expand generic task words like overview, explain, types, causes, or management "
            "unless the word is medically meaningful in this specific query\n"
            "- do not add broad unrelated diagnoses, neighboring topics, or facts outside the query intent\n\n"
            "Return only a comma-separated list of 3 to 10 short terms/phrases. "
            "No explanations, labels, bullets, numbering, or quotes.\n\n"
            f"Query: {query}\n\n"
            "Expanded terms:"
        )

        provider = (config.get('provider', 'openai') or 'openai').lower()
        expanded = ""

        if provider == 'ollama':
            sc = config.get('search_config') or search_config
            base_url = (sc.get('ollama_base_url') or 'http://localhost:11434').strip()
            model = (
                sc.get('ollama_query_expansion_model')
                or sc.get('ollama_chat_model')
                or 'llama3.2'
            )
            url = base_url.rstrip("/") + "/api/generate"
            data = {
                "model": str(model).strip(),
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 120, "temperature": 0.1},
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=8)
            result = json.loads(resp.read())
            expanded = (result.get("response") or "").strip()

        elif provider in ('local_openai', 'local_server', 'lm studio', 'local openai'):
            sc = config.get('search_config') or search_config
            base_url = (
                sc.get('local_llm_url')
                or config.get('local_llm_url')
                or 'http://localhost:1234/v1'
            )
            model = (
                sc.get('query_expansion_model')
                or sc.get('answer_local_model')
                or sc.get('local_llm_model')
                or config.get('local_llm_model')
                or 'model-identifier'
            )
            url = base_url.rstrip("/") + "/chat/completions"
            data = {
                "model": str(model).strip(),
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You expand search queries for medical retrieval. "
                            "Return only comma-separated search terms."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 120,
                "temperature": 0.1,
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=8)
            result = json.loads(resp.read())
            expanded = result["choices"][0]["message"]["content"].strip()

        else:
            api_key = config.get('api_key', '')
            if not api_key:
                return query

            model = self.get_best_model(provider)

            if provider == "openai":
                url = "https://api.openai.com/v1/chat/completions"
                data = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You expand search queries for medical retrieval. "
                                "Return only comma-separated search terms."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 120,
                    "temperature": 0.1,
                }
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
            elif provider == "anthropic":
                url = "https://api.anthropic.com/v1/messages"
                data = {
                    "model": model,
                    "max_tokens": 120,
                    "messages": [{"role": "user", "content": prompt}],
                }
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                }
            elif provider == "google":
                url = (
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{model}:generateContent?key={api_key}"
                )
                data = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": 120, "temperature": 0.1},
                }
                headers = {"Content-Type": "application/json"}
            else:
                return query

            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode(),
                headers=headers,
            )
            resp = urllib.request.urlopen(req, timeout=8)
            result = json.loads(resp.read())

            if provider == "openai":
                expanded = result["choices"][0]["message"]["content"].strip()
            elif provider == "anthropic":
                expanded = extract_anthropic_text(result, source="Anthropic query expansion").strip()
            elif provider == "google":
                expanded = result["candidates"][0]["content"]["parts"][0]["text"].strip()

        terms = _parse_query_expansion_terms(expanded, query)
        if not terms:
            return query

        expanded_query = f"{query} " + " ".join(terms)
        log_debug(f"AI query expansion: {query} -> {expanded_query}")
        return expanded_query

    except Exception as e:
        log_debug(f"Query expansion failed: {e}")

    return query


def _parse_query_expansion_terms(expanded, query):
    expanded = (expanded or "").strip()
    expanded = re.sub(r"(?i)^(synonyms|expanded terms|terms)\s*:\s*", "", expanded)
    if not expanded:
        return []

    if expanded.startswith("[") and expanded.endswith("]"):
        try:
            parsed = json.loads(expanded)
            raw_terms = [str(t) for t in parsed] if isinstance(parsed, list) else [expanded]
        except Exception:
            raw_terms = re.split(r",|\n|;", expanded)
    else:
        raw_terms = re.split(r",|\n|;", expanded)

    q_words = set(re.findall(r"\b\w+\b", (query or "").lower()))
    terms = []
    for term in raw_terms:
        term = re.sub(r"^\s*[-*]?\s*\d*[\).\s-]*", "", term or "").strip()
        term = term.strip(" \t\r\n\"'`[]{}")
        term = re.sub(r"\s+", " ", term)
        if not term or len(term) > 80:
            continue
        if term.lower() in q_words:
            continue
        terms.append(term)

    return list(dict.fromkeys(terms))[:10]



def _get_ai_excluded_terms(self, query, config):



    """One short LLM call to detect generic query terms to exclude. Returns set of lowercased terms, or empty set on failure."""



    prompt = (



        "List only words that are too generic to help find specific study notes "



        "(e.g. 'difference', 'between', 'what', 'the', 'mechanism', 'treatment'). "



        "Include question filler and words that appear in most notes. "



        "Return comma-separated words only, or exactly 'none' if all words are useful. "



        f"Query: {query}"



    )



    try:



        import urllib.request



        import json



        provider = config.get('provider', 'openai')



        search_config = config.get('search_config') or {}



        response_text = ""



        if provider == 'ollama':



            sc = config.get('search_config') or search_config



            base_url = (sc.get('ollama_base_url') or 'http://localhost:11434').strip()



            model = (sc.get('ollama_chat_model') or 'llama3.2').strip()



            url = base_url.rstrip("/") + "/api/generate"



            data = {"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": 50}}



            req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"}, method="POST")



            resp = urllib.request.urlopen(req, timeout=8)



            result = json.loads(resp.read())



            response_text = (result.get("response") or "").strip()



        else:



            api_key = config.get('api_key', '')



            if not api_key:



                return set()



            model = self.get_best_model(provider)



            if provider == "openai":



                url = "https://api.openai.com/v1/chat/completions"



                data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 50, "temperature": 0.1}



                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}



                req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers, method="POST")



                resp = urllib.request.urlopen(req, timeout=8)



                result = json.loads(resp.read())



                response_text = (result.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()



            elif provider == "anthropic":



                url = "https://api.anthropic.com/v1/messages"



                data = {"model": model, "max_tokens": 50, "messages": [{"role": "user", "content": prompt}]}



                headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}



                req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers, method="POST")



                resp = urllib.request.urlopen(req, timeout=8)



                result = json.loads(resp.read())



                response_text = extract_anthropic_text(result, source="Anthropic generic-term detection").strip()



            elif provider == "google":



                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"



                data = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": 50, "temperature": 0.1}}



                req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"}, method="POST")



                resp = urllib.request.urlopen(req, timeout=8)



                result = json.loads(resp.read())



                response_text = (result.get("candidates") or [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()



            else:



                return set()



        raw = (response_text or "").strip().lower()



        if not raw:



            return set()



        if raw in ("none", "n/a", "no", "nil"):



            return set()



        terms = []



        for part in raw.replace(";", ",").split(","):



            t = part.strip()



            if t and t not in ("none", "n/a", "no", "nil"):



                terms.append(t)



        result = set(terms)



        if search_config.get('verbose_search_debug') and result:



            log_debug(f"AI generic term detection excluded for this query: {result}")



        return result



    except Exception as e:



        log_debug(f"AI generic term detection failed: {e}")



        return set()



def _hyde_prompt_type(query):
    text = str(query or "")
    return "clinical_scenario" if len(text) > 200 or text.count("\n") >= 2 else "standard"


def _build_hyde_prompt(query):
    prompt_type = _hyde_prompt_type(query)
    if prompt_type == "clinical_scenario":
        prompt = (
            "Write a concise clinical retrieval hypothesis for the pasted scenario. "
            "Plain text only, no markdown. Include likely diagnosis, discriminating findings, "
            "related pathogens or drugs when relevant, and close clinical synonyms/search terms. "
            "Keep it to 2-3 compact sentences.\n\nScenario: " + str(query or "")
        )
    else:
        prompt = (
            "Write a brief 1-2 sentence hypothetical answer, as if from your study notes. "
            "Plain text only, no markdown.\n\nQuestion: " + str(query or "")
        )
    return prompt_type, prompt


def _hyde_has_clinical_anchor(text):
    value = str(text or "").lower()
    anchors = (
        "diagnosis", "diagnostic", "differential", "pathogen", "organism",
        "bacteria", "virus", "fungal", "drug", "treatment", "therapy",
        "clinical", "syndrome", "disease", "synonym",
    )
    return any(anchor in value for anchor in anchors)


def _finalize_hyde_document(self, text, prompt_type):
    text = (text or "").strip()
    try:
        self._last_hyde_prompt_type = prompt_type
        self._last_hyde_length = len(text)
        self._last_hyde_has_clinical_anchor = _hyde_has_clinical_anchor(text)
    except Exception:
        pass
    return text


def _generate_hyde_document(self, query, config):



    """Generate a brief hypothetical answer (HyDE) for retrieval: AI 'hallucinates' an answer, then we search on it."""



    HYDE_MAX_TOKENS = 90
    prompt_type, prompt = _build_hyde_prompt(query)
    try:
        self._last_hyde_prompt_type = prompt_type
        self._last_hyde_length = 0
        self._last_hyde_has_clinical_anchor = False
    except Exception:
        pass



    try:



        provider = config.get('provider', 'openai')



        api_key = config.get('api_key', '')



        if provider == 'ollama':



            sc = config.get('search_config') or {}



            base_url = (sc.get('ollama_base_url') or 'http://localhost:11434').strip()



            model = (sc.get('ollama_chat_model') or 'llama3.2').strip()



            url = base_url.rstrip("/") + "/api/generate"



            data = {



                "model": model,



                "prompt": prompt,



                "stream": False,



                "options": {"num_predict": HYDE_MAX_TOKENS}



            }



            req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"}, method="POST")



            resp = urllib.request.urlopen(req, timeout=15)



            result = json.loads(resp.read())



            return _finalize_hyde_document(self, result.get("response"), prompt_type)



        model = self.get_best_model(provider)



        if provider == "openai":



            url = "https://api.openai.com/v1/chat/completions"



            data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": HYDE_MAX_TOKENS, "temperature": 0.3}



            req = urllib.request.Request(url, data=json.dumps(data).encode(),



                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")



            resp = urllib.request.urlopen(req, timeout=60)



            result = json.loads(resp.read())



            return _finalize_hyde_document(self, result['choices'][0]['message']['content'], prompt_type)



        elif provider == "anthropic":



            url = "https://api.anthropic.com/v1/messages"



            data = {"model": model, "max_tokens": HYDE_MAX_TOKENS, "messages": [{"role": "user", "content": prompt}]}



            req = urllib.request.Request(url, data=json.dumps(data).encode(),



                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}, method="POST")



            resp = urllib.request.urlopen(req, timeout=60)



            result = json.loads(resp.read())



            return _finalize_hyde_document(self, extract_anthropic_text(result, source="Anthropic HyDE"), prompt_type)



        elif provider == "google":



            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"



            data = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": HYDE_MAX_TOKENS, "temperature": 0.3}}



            req = urllib.request.Request(url, data=json.dumps(data).encode(),



                headers={"Content-Type": "application/json"}, method="POST")



            resp = urllib.request.urlopen(req, timeout=60)



            result = json.loads(resp.read())



            return _finalize_hyde_document(self, result['candidates'][0]['content']['parts'][0]['text'], prompt_type)



        elif provider == "openrouter":



            url = "https://openrouter.ai/api/v1/chat/completions"



            data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": HYDE_MAX_TOKENS, "temperature": 0.3}



            req = urllib.request.Request(url, data=json.dumps(data).encode(),



                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")



            resp = urllib.request.urlopen(req, timeout=60)



            result = json.loads(resp.read())



            return _finalize_hyde_document(self, result['choices'][0]['message']['content'], prompt_type)



        else:



            # Custom endpoint (OpenAI-compatible shape)



            api_url = (config.get('api_url') or '').strip()



            if not api_url:



                return None



            data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": HYDE_MAX_TOKENS, "temperature": 0.3}



            req = urllib.request.Request(api_url, data=json.dumps(data).encode(),



                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")



            resp = urllib.request.urlopen(req, timeout=60)



            result = json.loads(resp.read())



            if 'choices' in result:



                return _finalize_hyde_document(self, result['choices'][0]['message']['content'], prompt_type)



            if 'content' in result:



                return _finalize_hyde_document(self, extract_anthropic_text(result, source="Anthropic HyDE compatible"), prompt_type)



            return None



    except Exception as e:



        log_debug(f"HyDE generation failed: {e}")



    return None



def _passes_focused_balanced_broad(



    self,



    matched_keywords,



    final_score,



    emb_score,



    max_emb_score,



    keywords,



    search_method,



    embeddings_available,



    min_emb_frac=0.25,



    very_high_emb_frac=0.9,



):



    """Compute whether a note would pass Focused, Balanced, or Broad inclusion. Returns (passes_focused, passes_balanced, passes_broad)."""



    n_kw = len(keywords) if keywords else 0



    # Focused



    min_kw_focused = max(2, int(n_kw * 0.4)) if n_kw else 1



    if n_kw <= 2:



        min_kw_focused = 1



    min_score_focused = 18



    # Balanced



    min_kw_balanced = max(1, int(n_kw * 0.25)) if n_kw else 1



    min_score_balanced = 10



    # Broad



    min_kw_broad = max(1, int(n_kw * 0.2)) if n_kw else 1



    min_score_broad = 8







    if search_method == "embedding" and embeddings_available:



        if emb_score > 0:



            return (True, True, True)



        return (False, False, False)







    if search_method == "hybrid" and embeddings_available and max_emb_score > 0:



        very_high = emb_score >= very_high_emb_frac * max_emb_score



        decent = emb_score >= min_emb_frac * max_emb_score



        pf = (



            (decent and (matched_keywords >= min_kw_focused or final_score > min_score_focused)) or very_high



        ) and (matched_keywords > 0 or very_high)



        pb_al = (decent and (matched_keywords >= min_kw_balanced or final_score > min_score_balanced)) or very_high



        pb_br = (decent and (matched_keywords >= min_kw_broad or final_score > min_score_broad)) or very_high



        return (pf, pb_al, pb_br)







    # Keyword-only or fallback



    pf = matched_keywords >= min_kw_focused



    if n_kw <= 2:



        pb_al = matched_keywords >= 1



        pb_br = matched_keywords >= 1



    else:



        pb_al = matched_keywords >= min_kw_balanced or final_score > min_score_balanced



        pb_br = matched_keywords >= min_kw_broad or final_score > min_score_broad



    return (pf, pb_al, pb_br)


class SearchQueryEnhancementMixin:
    """Owns query expansion, HyDE generation, and relevance-mode filtering."""

    _expand_query = _expand_query
    _get_ai_excluded_terms = _get_ai_excluded_terms
    _generate_hyde_document = _generate_hyde_document
    _passes_focused_balanced_broad = _passes_focused_balanced_broad
