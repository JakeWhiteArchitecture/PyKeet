"""Optional small-language-model pass over a dictation: LiquidAI LFM2.5-350M by default.

The model is asked to remove false starts / repeated phrases and tidy punctuation. Its answer is
never trusted: main.merge_cleanup() accepts only deleted repeats and punctuation/capital changes,
and keeps your original words for everything else. Needs `pip install transformers` (recent
enough for LFM2, >= 4.55) and the PyTorch you already have; the model (~700 MB) downloads on
first use.
"""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger("pykeet")

SYSTEM = ("You tidy up dictated text. Remove false starts and repeated phrases (when the speaker "
          "starts a sentence, then starts it again). Fix punctuation and capital letters. Do not "
          "add, change, reorder or translate any other words. Reply with the cleaned text only.")
EXAMPLES = [
    ("I might start a sentence and actually I might start a sentence and then finish it",
     "I might start a sentence and then finish it."),
    ("we need the drawings by friday can you confirm",
     "We need the drawings by Friday. Can you confirm?"),
]


class LLMCleaner:
    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("loading cleanup model %s (first use downloads it)", model_name)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        self.model.eval()
        self._lock = threading.Lock()

    def generate(self, text: str) -> str:
        messages = [{"role": "system", "content": SYSTEM}]
        for src, dst in EXAMPLES:
            messages += [{"role": "user", "content": src}, {"role": "assistant", "content": dst}]
        messages.append({"role": "user", "content": text})
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True)
        n_in = ids["input_ids"].shape[1]
        budget = int(len(text.split()) * 2) + 24
        with self._lock, self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=budget, do_sample=False)
        return self.tok.decode(out[0][n_in:], skip_special_tokens=True).strip()


_CLEANER = None
_FAILED = False


def get_cleaner(model_name: str):
    """The loaded cleaner, or None if it could not be loaded (logged once)."""
    global _CLEANER, _FAILED
    if _CLEANER is None and not _FAILED:
        try:
            t = time.monotonic()
            _CLEANER = LLMCleaner(model_name)
            log.info("cleanup model ready in %.1fs", time.monotonic() - t)
        except ImportError:
            _FAILED = True
            log.warning("cleanup_backend = 'llm' needs: pip install transformers")
        except Exception:
            _FAILED = True
            log.exception("could not load the cleanup model; continuing without it")
    return _CLEANER
