"""Query expansion, HyDE generation, and relevance-mode helpers."""

# ============================================================================
# Imports
# ============================================================================

import json
import urllib.request

from ..utils import load_config, log_debug


# ============================================================================
# Query Enhancement And Relevance Helpers
# ============================================================================

def _expand_query(self, query, config):



    """Expand query with synonyms using AI (optional). Supports Ollama and cloud providers."""



    search_config = load_config().get('search_config', {})







    # Always apply a small built\xe2\u20ac\u2018in synonym map for very common medical



    # variants so you don't have to remember which spelling your deck



    # used (e.g. adrenaline vs epinephrine). This runs even when the



    # AI-based expansion setting is disabled.



    try:



        q_lower = (query or "").lower()



        extra_terms = []



        # Pairs and small groups of common aliases / spelling variants



        synonym_groups = [



            # Catecholamines



            ["adrenaline", "epinephrine"],



            ["noradrenaline", "norepinephrine"],



            # Analgesics



            ["acetaminophen", "paracetamol"],



            # Hormones / vitamins (common exam phrasing variants)



            ["pth", "parathyroid hormone"],



            ["vitamin d", "cholecalciferol", "ergocalciferol"],



        ]



        for group in synonym_groups:



            present = [term for term in group if term in q_lower]



            if present:



                for term in group:



                    if term not in q_lower:



                        extra_terms.append(term)



        # Config-driven synonym overrides (same logic; no UI, edit config.json if needed)



        overrides = search_config.get("synonym_overrides") or []



        for item in overrides:



            if isinstance(item, (list, tuple)) and len(item) >= 2:



                group = [str(t).strip().lower() for t in item if t and str(t).strip()]



                if len(group) < 2:



                    continue



                present = [term for term in group if term in q_lower]



                if present:



                    for term in group:



                        if term not in q_lower:



                            extra_terms.append(term)



        if extra_terms:



            query = f"{query} " + " ".join(extra_terms)



    except Exception:



        # If anything goes wrong here, just fall back to the original query



        pass







    # Optional AI-based expansion (controlled from settings)



    if not search_config.get('enable_query_expansion', False):



        return query







    try:



        # Use a simple prompt to get synonyms / closely related terms



        prompt = (



            "Given this search query, list 2\xe2\u20ac\u201c4 key synonyms or closely related medical terms "



            "that would help find the same content. Return only the terms, comma-separated, "



            "no explanations or labels.\n\n"



            f"Query: {query}\n\n"



            "Synonyms:"



        )







        provider = config.get('provider', 'openai')



        import urllib.request



        import json







        # Ollama: fully local HTTP API



        if provider == 'ollama':



            sc = config.get('search_config') or search_config



            base_url = (sc.get('ollama_base_url') or 'http://localhost:11434').strip()



            # Allow a dedicated expansion model, falling back to chat model



            model = (



                sc.get('ollama_query_expansion_model')



                or sc.get('ollama_chat_model')



                or 'llama3.2'



            )



            model = str(model).strip()



            url = base_url.rstrip("/") + "/api/generate"



            data = {



                "model": model,



                "prompt": prompt,



                "stream": False,



                "options": {"num_predict": 64},



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







        else:



            api_key = config.get('api_key', '')



            if not api_key:



                return query







            model = self.get_best_model(provider)







            # Quick API call for expansion (cloud providers)



            if provider == "openai":



                url = "https://api.openai.com/v1/chat/completions"



                data = {



                    "model": model,



                    "messages": [{"role": "user", "content": prompt}],



                    "max_tokens": 50,



                    "temperature": 0.3,



                }



                headers = {



                    "Authorization": f"Bearer {api_key}",



                    "Content-Type": "application/json",



                }



            elif provider == "anthropic":



                url = "https://api.anthropic.com/v1/messages"



                data = {



                    "model": model,



                    "max_tokens": 50,



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



                    "generationConfig": {"maxOutputTokens": 50, "temperature": 0.3},



                }



                headers = {"Content-Type": "application/json"}



            else:



                # Skip expansion for unsupported providers



                return query







            req = urllib.request.Request(



                url,



                data=json.dumps(data).encode(),



                headers=headers,



            )



            resp = urllib.request.urlopen(req, timeout=5)



            result = json.loads(resp.read())







            if provider == "openai":



                expanded = result["choices"][0]["message"]["content"].strip()



            elif provider == "anthropic":



                expanded = result["content"][0]["text"].strip()



            elif provider == "google":



                expanded = result["candidates"][0]["content"]["parts"][0]["text"].strip()



            else:



                return query







        # Clean up and combine



        expanded = (expanded or "").replace("Synonyms:", "").strip()



        if not expanded:



            return query







        # Parse comma-separated terms, trim, and drop empties



        terms = [t.strip() for t in expanded.split(",") if t.strip()]



        if not terms:



            return query







        # Append terms to the original query so keyword extraction sees them



        return f"{query} " + " ".join(terms)



    except Exception as e:



        log_debug(f"Query expansion failed: {e}")







    return query



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



                response_text = (result.get("content") or [{}])[0].get("text", "").strip()



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



def _generate_hyde_document(self, query, config):



    """Generate a brief hypothetical answer (HyDE) for retrieval: AI 'hallucinates' an answer, then we search on it."""



    HYDE_MAX_TOKENS = 60



    prompt = (



        "Write a brief 1\xe2\u20ac\u201c2 sentence hypothetical answer, as if from your study notes. "



        "Plain text only, no markdown.\n\nQuestion: " + query



    )



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



            return (result.get("response") or "").strip()



        model = self.get_best_model(provider)



        if provider == "openai":



            url = "https://api.openai.com/v1/chat/completions"



            data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": HYDE_MAX_TOKENS, "temperature": 0.3}



            req = urllib.request.Request(url, data=json.dumps(data).encode(),



                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")



            resp = urllib.request.urlopen(req, timeout=60)



            result = json.loads(resp.read())



            return result['choices'][0]['message']['content'].strip()



        elif provider == "anthropic":



            url = "https://api.anthropic.com/v1/messages"



            data = {"model": model, "max_tokens": HYDE_MAX_TOKENS, "messages": [{"role": "user", "content": prompt}]}



            req = urllib.request.Request(url, data=json.dumps(data).encode(),



                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}, method="POST")



            resp = urllib.request.urlopen(req, timeout=60)



            result = json.loads(resp.read())



            return result['content'][0]['text'].strip()



        elif provider == "google":



            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"



            data = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": HYDE_MAX_TOKENS, "temperature": 0.3}}



            req = urllib.request.Request(url, data=json.dumps(data).encode(),



                headers={"Content-Type": "application/json"}, method="POST")



            resp = urllib.request.urlopen(req, timeout=60)



            result = json.loads(resp.read())



            return result['candidates'][0]['content']['parts'][0]['text'].strip()



        elif provider == "openrouter":



            url = "https://openrouter.ai/api/v1/chat/completions"



            data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": HYDE_MAX_TOKENS, "temperature": 0.3}



            req = urllib.request.Request(url, data=json.dumps(data).encode(),



                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")



            resp = urllib.request.urlopen(req, timeout=60)



            result = json.loads(resp.read())



            return result['choices'][0]['message']['content'].strip()



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



                return result['choices'][0]['message']['content'].strip()



            if 'content' in result and result['content']:



                return result['content'][0].get('text', '').strip()



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
