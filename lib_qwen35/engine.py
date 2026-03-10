"""
Qwen 3.5 4B Text Processing Engine for Forge Neo.

This follows the exact same pattern as Forge Neo's AnimaTextProcessingEngine
(backend/text_processing/anima_engine.py) but uses our custom Qwen 3.5 4B
hybrid model and tokenizer instead of the default Qwen3-0.6B.

Key design: We still use the Anima model's original LLM adapter
(preprocess_text_embeds) for the T5 pathway — our extension only replaces
the Qwen text encoding, not the adapter.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch
import logging

from backend import memory_management
from backend.text_processing import emphasis, parsing
from modules.shared import opts

logger = logging.getLogger(__name__)


class PromptChunk:
    """Holds tokenized prompt chunk with both Qwen3.5 and T5 tokens."""
    def __init__(self):
        self.qwen_tokens = []
        self.qwen_multipliers = []
        self.t5_tokens = []
        self.t5_multipliers = []


class Qwen35AnimaTextProcessingEngine:
    """
    Text processing engine that replaces Qwen3-0.6B with Qwen3.5-4B
    in the Anima pipeline.

    This engine:
    1. Tokenizes prompts with the Qwen3.5 tokenizer (vocab=248320)
    2. Runs embeddings through the Qwen3.5 4B hybrid model (SSM+Attention)
    3. Uses the ORIGINAL Anima model's LLM adapter (preprocess_text_embeds)
       for the T5 pathway — we only replace the Qwen text encoding
    4. Supports prompt emphasis/weighting via Forge's emphasis system

    Args:
        text_encoder: The Qwen35HybridModel instance
        qwen_tokenizer: Qwen35TokenizerWrapper instance (vocab=248320)
        t5_tokenizer: The original T5 tokenizer from the Anima model
        original_text_encoder: The original Qwen3-0.6B model (for its LLM adapter)
    """

    def __init__(self, text_encoder, qwen_tokenizer, t5_tokenizer, original_text_encoder=None):
        super().__init__()

        self.text_encoder = text_encoder  # Our Qwen3.5 4B hybrid model
        self.qwen_tokenizer = qwen_tokenizer  # Qwen3.5 tokenizer (vocab=248320)
        self.t5_tokenizer = t5_tokenizer  # T5 tokenizer from original Anima
        self.original_text_encoder = original_text_encoder  # For LLM adapter

        # Qwen3.5 pad token (same as Qwen3)
        self.id_pad = 151643
        # T5 end token
        self.id_end = 1

    def tokenize(self, texts):
        """Tokenize texts with both Qwen3.5 and T5 tokenizers.

        Returns:
            (qwen_token_ids, t5_token_ids) tuple
        """
        return (
            self.qwen_tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"],
            self.t5_tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"],
        )

    def tokenize_line(self, line: str):
        """Parse a prompt line with emphasis and tokenize both streams.

        Follows the exact same pattern as AnimaTextProcessingEngine.tokenize_line().
        """
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)
        qwen_tokenized, t5_tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()

        def next_chunk():
            nonlocal chunk

            if not chunk.qwen_tokens:
                chunk.qwen_tokens.append(self.id_pad)
                chunk.qwen_multipliers.append(1.0)

            chunk.t5_tokens.append(self.id_end)
            chunk.t5_multipliers.append(1.0)

            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens in qwen_tokenized:
            position = 0
            while position < len(tokens):
                token = tokens[position]
                chunk.qwen_tokens.append(token)
                chunk.qwen_multipliers.append(1.0)
                position += 1

        for tokens, (text, weight) in zip(t5_tokenized, parsed):
            position = 0
            while position < len(tokens):
                token = tokens[position]
                chunk.t5_tokens.append(token)
                chunk.t5_multipliers.append(weight)
                position += 1

        if not chunks:
            next_chunk()

        return chunks

    def __call__(self, texts: "SdConditioning"):
        """Main entry point — matches AnimaTextProcessingEngine.__call__() exactly."""
        zs = []
        cache = {}

        self.emphasis = emphasis.get_current_option(opts.emphasis)()

        for line in texts:
            if line in cache:
                z = cache[line]
            else:
                chunks: list[PromptChunk] = self.tokenize_line(line)
                assert len(chunks) == 1

                for chunk in chunks:
                    tokens = chunk.qwen_tokens
                    multipliers = chunk.qwen_multipliers

                    z: torch.Tensor = self.process_tokens([tokens], [multipliers])[0]

                cache[line] = z

            zs.append(
                self.anima_preprocess(
                    z,
                    torch.tensor(chunk.t5_tokens, dtype=torch.int),
                    torch.tensor(chunk.t5_multipliers),
                )
            )

        return zs

    def anima_preprocess(self, cross_attn: torch.Tensor, t5xxl_ids: torch.Tensor, t5xxl_weights: torch.Tensor) -> torch.Tensor:
        """Run the Anima LLM adapter on the embeddings.

        Uses the ORIGINAL Anima model's preprocess_text_embeds (LLM adapter)
        since that's part of the diffusion model, not the text encoder.
        """
        device = memory_management.text_encoder_device()

        cross_attn = cross_attn.unsqueeze(0).to(device=device)
        t5xxl_ids = t5xxl_ids.unsqueeze(0).to(device=device)

        # Use the original Anima model's LLM adapter
        if self.original_text_encoder is not None:
            # Cast to match the LLM adapter's dtype (e.g. bfloat16) — our 4B model
            # may output float32, but the adapter's projections are in the 0.6B dtype.
            adapter_dtype = next(self.original_text_encoder.llm_adapter.parameters()).dtype
            cross_attn = cross_attn.to(dtype=adapter_dtype)
            cross_attn = self.original_text_encoder.preprocess_text_embeds(cross_attn, t5xxl_ids)
        else:
            logger.warning("[Qwen3.5-Anima] No original text encoder available for LLM adapter, skipping preprocess_text_embeds")

        if t5xxl_weights is not None:
            cross_attn *= t5xxl_weights.unsqueeze(0).unsqueeze(-1).to(cross_attn)

        if cross_attn.shape[1] < 512:
            cross_attn = torch.nn.functional.pad(cross_attn, (0, 0, 0, 512 - cross_attn.shape[1]))

        return cross_attn

    def process_embeds(self, batch_tokens):
        """Convert token IDs to embeddings via our Qwen3.5 model's embedding layer.

        Follows the exact same pattern as AnimaTextProcessingEngine.process_embeds().
        """
        device = memory_management.text_encoder_device()

        embeds_out = []
        attention_masks = []
        num_tokens = []

        for tokens in batch_tokens:
            attention_mask = []
            tokens_temp = []
            eos = False
            index = 0

            for t in tokens:
                try:
                    token = int(t)
                    attention_mask.append(0 if eos else 1)
                    tokens_temp += [token]
                    if not eos and token == self.id_pad:
                        eos = True
                except TypeError:
                    pass
                index += 1

            tokens_embed = torch.tensor([tokens_temp], device=device, dtype=torch.long)
            tokens_embed = self.text_encoder.get_input_embeddings()(tokens_embed)

            embeds_out.append(tokens_embed)
            attention_masks.append(attention_mask)
            num_tokens.append(sum(attention_mask))

        return torch.cat(embeds_out), torch.tensor(attention_masks, device=device, dtype=torch.long), num_tokens

    def process_tokens(self, batch_tokens, batch_multipliers):
        """Run the Qwen3.5 text encoder forward pass.

        Unlike AnimaTextProcessingEngine which calls the Qwen3-0.6B model,
        this calls our custom Qwen35HybridModel. The interface is the same:
        the model's forward() returns (output, intermediate).

        Note: Qwen-side multipliers are always 1.0 in Anima (emphasis only
        applies to T5 weights in anima_preprocess), matching the original engine.
        """
        embeds, mask, count = self.process_embeds(batch_tokens)
        z, _ = self.text_encoder(
            input_ids=None,
            embeds=embeds,
            attention_mask=mask,
            num_tokens=count,
        )
        return z
