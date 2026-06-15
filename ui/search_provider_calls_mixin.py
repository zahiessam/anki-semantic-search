import json
import re
import urllib.error
import urllib.request

from ..utils import log_debug
from ..utils.anthropic_response import extract_anthropic_text


PROVIDER_DIGIT_RE = re.compile(r"\d+")


class SearchProviderCallsMixin:
    def _emit_stream_text(self, chunk_callback, text):
        if chunk_callback and text:
            chunk_callback(text)

    def _openai_stream_response(self, url, headers, data, timeout_seconds=300, chunk_callback=None):
        payload = dict(data)
        payload["stream"] = True
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        full_response = ""
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content") or ""
                if text:
                    full_response += text
                    self._emit_stream_text(chunk_callback, text)
        return full_response

    def _ollama_generate_stream_response(self, prompt, base_url, model, timeout_seconds=300, chunk_callback=None, num_predict=4096):
        url = base_url.rstrip("/") + "/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": num_predict},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        full_response = ""
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                event = json.loads(line)
                text = event.get("response") or ""
                if text:
                    full_response += text
                    self._emit_stream_text(chunk_callback, text)
                if event.get("done"):
                    break
        return full_response

    def call_ollama_stream(self, prompt, base_url, model, notes, chunk_callback=None):
        log_debug(f"Calling Ollama streaming API: {base_url}, model={model}")
        try:
            full_response = self._ollama_generate_stream_response(
                prompt, base_url, model, timeout_seconds=300, chunk_callback=chunk_callback
            )
            return self.parse_response((full_response or "").strip(), notes)
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = str(e)
            log_debug(f"Ollama streaming HTTP error: {e.code} {err_body}")
            raise Exception(f"Ollama error ({e.code}): {err_body[:200]}")
        except urllib.error.URLError as e:
            msg = str(getattr(e, "reason", e))
            if "timed out" in msg.lower():
                raise Exception("Ollama request timed out. Try a smaller model or more notes.")
            raise Exception(f"Cannot reach Ollama: {msg}. Is Ollama running (ollama serve)?")

    def call_openai_stream(self, prompt, api_key, model, notes, chunk_callback=None):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 4096}
        full_response = self._openai_stream_response(url, headers, data, chunk_callback=chunk_callback)
        return self.parse_response(full_response, notes)

    def call_openrouter_stream(self, prompt, api_key, model, notes, chunk_callback=None):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 4096}
        full_response = self._openai_stream_response(url, headers, data, chunk_callback=chunk_callback)
        return self.parse_response(full_response, notes)

    def call_custom_stream(self, prompt, api_key, model, api_url, notes, timeout_seconds=300, max_tokens=4096, chunk_callback=None):
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
        full_response = self._openai_stream_response(
            api_url, headers, data, timeout_seconds=timeout_seconds, chunk_callback=chunk_callback
        )
        return self.parse_response(full_response, notes)

    def call_ollama(self, prompt, base_url, model, notes):



        """Call Ollama /api/generate for AI answers (no API key)."""



        import json



        import urllib.request



        import urllib.error



        log_debug(f"Calling Ollama API: {base_url}, model={model}")



        url = base_url.rstrip("/") + "/api/generate"



        data = {



            "model": model,



            "prompt": prompt,



            "stream": False,



            "options": {"num_predict": 4096}



        }



        req = urllib.request.Request(



            url,



            data=json.dumps(data).encode("utf-8"),



            headers={"Content-Type": "application/json"},



            method="POST"



        )



        try:



            # Reasoning models (e.g. deepseek-r1) can take several minutes; use 5 min timeout



            with urllib.request.urlopen(req, timeout=300) as resp:



                result = json.loads(resp.read().decode("utf-8"))



            # /api/generate returns "response"; /api/chat returns message.content; some models use "thinking"



            full_response = (



                result.get("response")



                or (result.get("message") or {}).get("content")



                or result.get("thinking")



                or ""



            )



            if isinstance(full_response, list):



                # Some models return content as list of parts



                full_response = "".join(



                    p.get("text", p) if isinstance(p, dict) else str(p)



                    for p in full_response



                )



            full_response = (full_response or "").strip()



            return self.parse_response(full_response, notes)



        except urllib.error.HTTPError as e:



            try:



                err_body = e.read().decode("utf-8")



            except Exception:



                err_body = str(e)



            log_debug(f"Ollama HTTP error: {e.code} {err_body}")



            raise Exception(f"Ollama error ({e.code}): {err_body[:200]}")



        except urllib.error.URLError as e:



            msg = str(getattr(e, "reason", e))



            if "timed out" in msg.lower():



                raise Exception("Ollama request timed out. Try a smaller model or more notes.")



            raise Exception(f"Cannot reach Ollama: {msg}. Is Ollama running (ollama serve)?")



        except Exception as e:



            log_debug(f"Ollama error: {e}")



            raise Exception(f"Ollama error: {e}")







    def call_anthropic(self, prompt=None, api_key=None, model=None, notes=None, system_blocks=None, user_content=None):



        """Call Anthropic API. Use system_blocks+user_content for prompt caching (recommended); else single prompt."""



        log_debug(f"Calling Anthropic API with model: {model}")



        url = "https://api.anthropic.com/v1/messages"



        headers = {



            "Content-Type": "application/json",



            "x-api-key": api_key,



            "anthropic-version": "2023-06-01"



        }



        if system_blocks is not None and user_content is not None:



            data = {



                "model": model,



                "max_tokens": 4096,



                "system": system_blocks,



                "messages": [{"role": "user", "content": user_content}]



            }



        else:



            data = {



                "model": model,



                "max_tokens": 4096,



                "messages": [{"role": "user", "content": prompt}]



            }



        response_text = self.make_request(url, headers, data)



        result = json.loads(response_text)



        full_response = extract_anthropic_text(result, source="Anthropic answer")



        return self.parse_response(full_response, notes)







    def call_openai(self, prompt, api_key, model, notes):



        url = "https://api.openai.com/v1/chat/completions"



        headers = {



            "Content-Type": "application/json"



        }


        if api_key:



            headers["Authorization"] = f"Bearer {api_key}"



        data = {



            "model": model,



            "messages": [{"role": "user", "content": prompt}],



            "max_tokens": 4096



        }







        response_text = self.make_request(url, headers, data)



        result = json.loads(response_text)



        full_response = result['choices'][0]['message']['content']



        return self.parse_response(full_response, notes)







    def call_google(self, prompt, api_key, model, notes):



        url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"



        headers = {"Content-Type": "application/json"}



        data = {



            "contents": [{"parts": [{"text": prompt}]}],



            "generationConfig": {



                "maxOutputTokens": 4096



            }



        }







        response_text = self.make_request(url, headers, data)



        result = json.loads(response_text)



        full_response = result['candidates'][0]['content']['parts'][0]['text']



        return self.parse_response(full_response, notes)







    def call_openrouter(self, prompt, api_key, model, notes):



        url = "https://openrouter.ai/api/v1/chat/completions"



        headers = {



            "Content-Type": "application/json"



        }



        if api_key:



            headers["Authorization"] = f"Bearer {api_key}"



        data = {



            "model": model,



            "messages": [{"role": "user", "content": prompt}],



            "max_tokens": 4096



        }







        response_text = self.make_request(url, headers, data)



        result = json.loads(response_text)



        full_response = result['choices'][0]['message']['content']



        return self.parse_response(full_response, notes)







    def call_custom(self, prompt, api_key, model, api_url, notes, timeout_seconds=30, max_tokens=4096):



        headers = {



            "Content-Type": "application/json",



            "Authorization": f"Bearer {api_key}"



        }



        data = {



            "model": model,



            "messages": [{"role": "user", "content": prompt}],



            "max_tokens": max_tokens



        }







        response_text = self.make_request(api_url, headers, data, timeout_seconds=timeout_seconds)



        result = json.loads(response_text)







        if 'choices' in result:



            full_response = result['choices'][0]['message']['content']



        elif 'content' in result:



            full_response = extract_anthropic_text(result, source="Anthropic-compatible answer")



        else:



            full_response = str(result)







        return self.parse_response(full_response, notes)







    def _openai_compatible_chat_url(self, base_url):



        """Return a chat-completions endpoint for OpenAI-compatible servers."""



        url = (base_url or '').strip()



        if not url:



            raise ValueError("Local server URL is empty. Set it in Settings > API Settings.")



        if "://" not in url:



            url = "http://" + url



        url = url.rstrip("/")



        lower_url = url.lower()



        if lower_url.endswith("/chat/completions"):



            return url


        if lower_url.endswith("/v1"):



            return url + "/chat/completions"


        for suffix in ("/api/chat", "/api/generate", "/api/tags"):



            if lower_url.endswith(suffix):



                return url[: -len(suffix)] + "/v1/chat/completions"


        if lower_url.endswith("/v1/models"):



            return url[:-7] + "/chat/completions"


        if lower_url.endswith("/models"):



            base = url[:-7]



            if ":11434" in lower_url:



                return base + "/v1/chat/completions"



            return base + "/chat/completions"


        if ":11434" in lower_url:



            return url + "/v1/chat/completions"


        if lower_url.endswith("/api"):



            return url[:-4] + "/v1/chat/completions"


        return url + "/chat/completions"



    def make_request(self, url, headers, data, timeout_seconds=30):



        """Make HTTP request with proper timeout and error handling"""



        log_debug(f"Making API request to: {url}")



        log_debug(f"Request data keys: {list(data.keys())}")







        req = urllib.request.Request(



            url,



            data=json.dumps(data).encode('utf-8'),



            headers=headers,



            method='POST'



        )







        try:



            timeout_seconds = max(1, int(timeout_seconds or 30))



            log_debug(f"Opening URL connection (timeout: {timeout_seconds} seconds)...")



            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:



                log_debug(f"Received response, status: {response.status}")



                response_data = response.read().decode('utf-8')



                log_debug(f"Response length: {len(response_data)} characters")



                return response_data







        except urllib.error.HTTPError as e:



            try:



                error_msg = e.read().decode('utf-8')



            except:



                error_msg = str(e)



            log_debug(f"HTTP Error {e.code}: {error_msg}")



            raise Exception(f"API Error ({e.code}): {error_msg}")







        except urllib.error.URLError as e:



            # Handle common "no internet / host unreachable" cases more clearly



            reason = getattr(e, "reason", e)



            msg = str(reason)



            log_debug(f"URL Error: {msg}")



            lower = msg.lower()







            # Windows / general "no internet or cannot resolve host" patterns



            if (



                "getaddrinfo failed" in lower



                or "name or service not known" in lower



                or "temporary failure in name resolution" in lower



                or "nodename nor servname provided" in lower



                or "winerror 11001" in lower  # host not found



                or "winerror 10051" in lower  # network unreachable



                or "winerror 10065" in lower  # no route to host



            ):



                raise Exception("No internet connection or the API host cannot be reached.")







            if "timed out" in lower:



                raise Exception(f"Request timed out after {timeout_seconds} seconds. The API may be slow or overloaded.")







            raise Exception(f"Network error: {msg}")







        except Exception as e:



            log_debug(f"Unexpected error: {type(e).__name__}: {str(e)}")


            lower = str(e).lower()



            if "winerror 10054" in lower or "forcibly closed" in lower or "connection reset" in lower:



                raise Exception(
                    "Local AI server closed the connection. This usually means the server URL endpoint is wrong, "
                    "the selected model crashed/unloaded, or the request exceeded what the provider can handle. "
                    "For OpenAI-compatible local servers use a /v1 base URL such as http://localhost:11434/v1 "
                    "for Ollama, http://localhost:1234/v1 for LM Studio, or http://localhost:1337/v1 for Jan."
                )



            raise Exception(f"Request error: {str(e)}")







    def parse_response(self, full_response, notes):



        import re



        answer_part = ""



        relevant_notes = []



        if "RELEVANT_NOTES:" in full_response:

            parts = full_response.split("RELEVANT_NOTES:")

            answer_part = parts[0].strip()



            if len(parts) > 1:

                notes_str = parts[1].strip()

                numbers = PROVIDER_DIGIT_RE.findall(notes_str)

                relevant_notes = [int(n) - 1 for n in numbers if n.isdigit() and 0 <= int(n) - 1 < len(notes)]

        else:

            answer_part = full_response

            relevant_notes = list(range(min(3, len(notes))))







        log_debug(f"Parsed {len(relevant_notes)} relevant notes from AI response")



        return answer_part, relevant_notes
