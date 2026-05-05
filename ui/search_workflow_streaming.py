"""Anthropic streaming worker for the search dialog."""

from aqt.qt import QThread, pyqtSignal

from ..utils import log_debug


class AnthropicStreamWorker(QThread):



    """Worker thread for Anthropic streaming. Emits text chunks for real-time UI updates."""



    chunk_signal = pyqtSignal(str)



    done_signal = pyqtSignal(str)



    error_signal = pyqtSignal(str)







    # --- Anthropic Streaming Thread Lifecycle ---

    def __init__(self, api_key, model, system_blocks, user_content, notes):



        super().__init__()



        self.api_key = api_key



        self.model = model



        self.system_blocks = system_blocks



        self.user_content = user_content



        self.notes = notes







    def run(self):



        try:



            import anthropic



            client = anthropic.Anthropic(api_key=self.api_key)



            full_text = ""



            with client.messages.stream(



                model=self.model,



                max_tokens=4096,



                system=self.system_blocks,



                messages=[{"role": "user", "content": self.user_content}],



            ) as stream:



                for text in stream.text_stream:



                    full_text += text



                    self.chunk_signal.emit(text)



            self.done_signal.emit(full_text)



        except Exception as e:



            log_debug(f"AnthropicStreamWorker error: {e}")



            self.error_signal.emit(str(e))











# END OF PART 2 - PART 3: Methods below are indented under EmbeddingSearchWorker
