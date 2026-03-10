"""
Qwen 3.5 tokenizer wrapper for Forge Neo.

Wraps HuggingFace's AutoTokenizer for the Qwen 3.5 model (vocab_size=248320).
This is NOT the same as the Qwen3 tokenizer (vocab=151936).

The tokenizer is used by Qwen35AnimaTextProcessingEngine to tokenize prompts
before passing them to the Qwen 3.5 4B hybrid text encoder.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Extension directory paths
EXTENSION_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
QWEN35_TOKENIZER_DIR = os.path.join(EXTENSION_DIR, "qwen35_tokenizer")


class Qwen35TokenizerWrapper:
    """
    Tokenizer wrapper for the Qwen 3.5 text encoder (vocab_size=248320).

    CRITICAL: The Qwen3.5 4B encoder IS the official Qwen3.5-4B text backbone.
    It requires the Qwen3.5 tokenizer, NOT the Qwen3 tokenizer.

    The Qwen3 tokenizer has vocab=151936: using it means:
    - Different BPE merge rules produce different token boundaries
    - Every token ID maps to the wrong embedding row
    - 96,384 trained embedding rows (151936-248319) are never accessed
    - The model receives garbled input it was never trained on

    Tokenizer files (vocab.json, merges.txt, tokenizer.json) should be placed in
    the 'qwen35_tokenizer/' subdirectory of the extension, OR will be auto-downloaded
    from Qwen/Qwen3.5-4B on HuggingFace.
    """

    def __init__(self):
        self._tokenizer = None
        self.pad_token_id = 151643  # <|endoftext|>
        self.eos_token_id = 248044  # Qwen3.5 eos

    def _ensure_loaded(self):
        """Lazy-load the tokenizer on first use."""
        if self._tokenizer is not None:
            return

        tokenizer_path = self._find_tokenizer()
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=False
        )
        logger.info(f"[Qwen3.5-Anima] Loaded Qwen3.5 tokenizer (vocab_size={self._tokenizer.vocab_size})")

    def _find_tokenizer(self):
        """
        Find Qwen3.5 tokenizer files. Priority:
        1. Bundled qwen35_tokenizer/ directory in the extension folder
        2. HuggingFace auto-download from Qwen/Qwen3.5-4B
        """
        # Option 1: Bundled with extension
        if os.path.isdir(QWEN35_TOKENIZER_DIR):
            has_vocab = os.path.exists(os.path.join(QWEN35_TOKENIZER_DIR, "vocab.json"))
            has_tokenizer_json = os.path.exists(os.path.join(QWEN35_TOKENIZER_DIR, "tokenizer.json"))
            if has_vocab or has_tokenizer_json:
                logger.info(f"[Qwen3.5-Anima] Using bundled Qwen3.5 tokenizer from: {QWEN35_TOKENIZER_DIR}")
                return QWEN35_TOKENIZER_DIR

        # Option 2: HuggingFace auto-download (~10MB on first use)
        logger.info("[Qwen3.5-Anima] Qwen3.5 tokenizer not found locally, will load from Qwen/Qwen3.5-4B...")
        return "Qwen/Qwen3.5-4B"

    def __call__(self, texts, **kwargs):
        """Tokenize texts. Returns dict with 'input_ids' key, matching HuggingFace interface."""
        self._ensure_loaded()
        return self._tokenizer(texts, **kwargs)

    def encode(self, text, **kwargs):
        """Encode a single text string to token IDs."""
        self._ensure_loaded()
        return self._tokenizer.encode(text, **kwargs)

    def decode(self, token_ids, **kwargs):
        """Decode token IDs back to text."""
        self._ensure_loaded()
        return self._tokenizer.decode(token_ids, **kwargs)

    @property
    def vocab_size(self):
        self._ensure_loaded()
        return self._tokenizer.vocab_size
