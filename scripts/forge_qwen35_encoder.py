"""
Qwen 3.5 4B Text Encoder Extension for Forge Neo.

Replaces the default Qwen3-0.6B text encoder in the Anima 2B pipeline with
the Qwen 3.5 4B hybrid (Mamba2 + Attention) text encoder for improved prompt
understanding and text-image alignment.

Architecture: 32-layer hybrid (24 Mamba2 SSM + 8 Self-Attention), vocab=248320,
hidden=2560, output=1024. Roughly 7x larger than the stock 0.6B encoder.

Usage:
1. Place qwen35_4b.safetensors in models/text_encoder/
2. Load an Anima 2B model WITH the stock qwen_3_06b_base in the VAE/Text Encoder
   dropdown (the LLM adapter lives on the 0.6B model — it's always needed)
3. Enable "Qwen3.5 Text Encoder (Anima)" in the UI
4. Select qwen35_4b.safetensors in the extension's Model File dropdown
5. Generate as normal — the extension intercepts text encoding automatically

IMPORTANT: Do NOT put qwen35_4b.safetensors in the top VAE/Text Encoder dropdown.
That dropdown is for the stock 0.6B encoder. The 4B file goes in THIS extension only.
"""

import os
import logging

import gradio as gr
import torch

from modules import scripts
from modules.infotext_utils import PasteField
from modules.processing import StableDiffusionProcessing
from modules.ui_components import InputAccordion
from modules_forge.main_entry import module_list

logger = logging.getLogger(__name__)


def log(msg):
    """Print to console directly — Forge's logging may not show extension loggers."""
    print(msg)


def _grid_reference():
    """Find the xyz_grid script module."""
    for data in scripts.scripts_data:
        if data.script_class.__module__ in ("scripts.xyz_grid", "xyz_grid.py") and hasattr(data, "module"):
            return data.module
    return None


def _xyz_support(cache: dict):
    """Register Qwen3.5 axis options in X/Y/Z Plot."""
    xyz_grid = _grid_reference()
    if xyz_grid is None:
        log("[Qwen3.5-Anima] Could not find X/Y/Z Plot script — XYZ grid support disabled.")
        return

    def apply_field(field):
        def _(p, x, xs):
            cache[field] = x
        return _

    extra_axis_options = [
        xyz_grid.AxisOption("Q35 Enable", str, apply_field("enable"), choices=xyz_grid.boolean_choice()),
        xyz_grid.AxisOption("Q35 Use Alignment", str, apply_field("use_alignment"), choices=xyz_grid.boolean_choice()),
        xyz_grid.AxisOption("Q35 Alignment Strength", float, apply_field("alignment_strength")),
    ]
    xyz_grid.axis_options.extend(extra_axis_options)


class Qwen35EncoderForForge(scripts.Script):
    sorting_priority = 260209300  # After mod_guidance (260209268)
    XYZ_CACHE: dict = {}

    def __init__(self):
        super().__init__()
        self._cached_model = None
        self._cached_model_name = None
        self._cached_tokenizer = None
        self._qwen35_clip = None  # CLIP patcher for memory management
        _xyz_support(self.XYZ_CACHE)

    def title(self):
        return "Qwen3.5 Text Encoder (Anima)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        modules = list(module_list.keys())

        with InputAccordion(False, label=self.title()) as enable:
            notice = gr.Markdown(
                "**Important:** Keep `qwen_3_06b_base.safetensors` in the top VAE/Text Encoder "
                "dropdown (its LLM adapter is still needed). Select `qwen35_4b.safetensors` "
                "in the Model File dropdown below — do **not** put it in the top dropdown."
            )
            notice.do_not_save_to_config = True
            with gr.Row():
                model_file = gr.Dropdown(
                    label="Model File",
                    choices=modules,
                    value=next((m for m in modules if "qwen35" in m.lower()), next(iter(modules), None)),
                    info="Select qwen35_4b.safetensors from models/text_encoder/",
                )
            with gr.Row():
                use_calibration = gr.Checkbox(
                    label="Use Calibration",
                    value=False,
                    info="Per-dimension affine calibration to align 4B output with 0.6B distribution. Requires calibration_params.safetensors.",
                )
                use_alignment = gr.Checkbox(
                    label="Use Alignment",
                    value=False,
                    info="Procrustes rotation to align 4B concept directions with 0.6B. Requires rotation_matrix.safetensors.",
                )
            with gr.Row():
                alignment_strength = gr.Slider(
                    label="Alignment Strength",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.05,
                    info="Bias shift: 0=keep 4B magnitude, 1=shift to 0.6B magnitude. Rotation is always applied when alignment is on.",
                )
                output_scale = gr.Slider(
                    label="Output Scale",
                    minimum=0.0,
                    maximum=1000.0,
                    value=1.0,
                    step=0.1,
                    info="Uniform scale factor applied after calibration. Usually leave at 1.0.",
                )

        self.infotext_fields = [
            PasteField(model_file, "q35_model"),
            PasteField(use_calibration, "q35_calibration"),
            PasteField(use_alignment, "q35_alignment"),
            PasteField(alignment_strength, "q35_align_str"),
            PasteField(output_scale, "q35_scale"),
        ]

        return [enable, model_file, use_calibration, use_alignment, alignment_strength, output_scale]

    def _load_model(self, model_name: str, use_calibration: bool, use_alignment: bool,
                    alignment_strength: float, output_scale: float):
        """Load or re-use the Qwen3.5 4B model.

        Returns:
            (model, tokenizer) tuple
        """
        from backend.utils import load_torch_file

        # Re-use cached model if same file
        if self._cached_model is not None and self._cached_model_name == model_name:
            model = self._cached_model
            # Update settings even for cached model
            model._output_scale = output_scale
            model._use_calibration = use_calibration
            model._use_alignment = use_alignment
            model._alignment_strength = alignment_strength
            if use_calibration:
                model._load_calibration()
            if use_alignment:
                model._load_alignment()
            return model, self._cached_tokenizer

        # Clean up old model
        if self._cached_model is not None:
            del self._cached_model
            self._cached_model = None
            self._cached_model_name = None
            self._cached_tokenizer = None
            self._qwen35_clip = None

        # Find model file path
        if model_name in module_list:
            model_path = module_list[model_name]
        else:
            raise FileNotFoundError(f"Model file not found: {model_name}. Place it in models/text_encoder/.")

        log(f"[Qwen3.5-Anima] Loading text encoder from: {model_path}")

        # Load state dict
        sd = load_torch_file(model_path)

        # Detect dtype from checkpoint
        dtype = None
        for norm_key in ["model.norm.weight", "model.layers.0.input_layernorm.weight",
                         "norm.1.weight", "layers.0.input_layernorm.weight"]:
            if norm_key in sd:
                dtype = sd[norm_key].dtype
                break

        if dtype is None:
            # Default to bfloat16 if can't detect
            dtype = torch.bfloat16

        log(f"[Qwen3.5-Anima] Detected dtype: {dtype}")

        # Create model
        from lib_qwen35.model import Qwen35HybridModel

        model = Qwen35HybridModel(device="cpu", dtype=dtype)

        # Load weights
        missing, unexpected = model.load_state_dict(sd, strict=False)
        del sd

        if missing:
            log(f"[Qwen3.5-Anima] Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            log(f"[Qwen3.5-Anima] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

        # Apply settings
        model._output_scale = output_scale
        model._use_calibration = use_calibration
        model._use_alignment = use_alignment
        model._alignment_strength = alignment_strength

        if use_calibration:
            model._load_calibration()
        if use_alignment:
            model._load_alignment()

        model.eval()

        # Create tokenizer
        from lib_qwen35.tokenizer import Qwen35TokenizerWrapper
        tokenizer = Qwen35TokenizerWrapper()

        # Cache
        self._cached_model = model
        self._cached_model_name = model_name
        self._cached_tokenizer = tokenizer

        param_count = sum(p.numel() for p in model.parameters())
        log(f"[Qwen3.5-Anima] Text encoder loaded ({param_count:,} parameters, dtype={dtype})")

        return model, tokenizer

    def process_batch(self, p: StableDiffusionProcessing, enable: bool,
                      model_file: str, use_calibration: bool, use_alignment: bool,
                      alignment_strength: float, output_scale: float, **kwargs):
        """Called BEFORE setup_conds() — monkeypatch get_learned_conditioning here."""
        # Apply XYZ overrides
        cache = self.XYZ_CACHE
        if cache:
            if "enable" in cache:
                enable = str(cache["enable"]).lower() in ("true", "yes", "y", "1")
            if "use_alignment" in cache:
                use_alignment = str(cache["use_alignment"]).lower() in ("true", "yes", "y", "1")
            if "alignment_strength" in cache:
                alignment_strength = float(cache["alignment_strength"])
            cache.clear()

        if not enable:
            return

        log("[Qwen3.5-Anima] process_batch fired, setting up text encoder replacement...")

        # CRITICAL: Invalidate the conditioning cache so setup_conds() actually
        # calls get_learned_conditioning instead of reusing stale cached results
        # from a previous generation (with the stock 0.6B encoder).
        p.cached_c = [None, None, None]
        p.cached_uc = [None, None, None]
        from modules.processing import StableDiffusionProcessing as SDP
        SDP.cached_c = [None, None, None]
        SDP.cached_uc = [None, None, None]

        # Check if the current model is Anima
        if not hasattr(p.sd_model, 'text_processing_engine_anima'):
            log("[Qwen3.5-Anima] Current model does not appear to be Anima. Skipping.")
            return

        if not hasattr(p.sd_model, 'is_wan') or not p.sd_model.is_wan:
            log("[Qwen3.5-Anima] Current model is not a WAN-type model. Skipping.")
            return

        from backend import memory_management
        from backend.patcher.clip import CLIP

        try:
            qwen35_model, qwen35_tokenizer = self._load_model(
                model_file, use_calibration, use_alignment,
                alignment_strength, output_scale
            )
        except Exception as e:
            log(f"[Qwen3.5-Anima] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return

        # Get the original Anima clip objects
        original_clip = p.sd_model.forge_objects.clip
        original_text_encoder = original_clip.cond_stage_model.qwen3_06b
        original_t5_tokenizer = original_clip.tokenizer.t5xxl

        # Create a CLIP patcher wrapper for our Qwen3.5 model so it can be
        # loaded/offloaded by memory_management
        if self._qwen35_clip is None:
            self._qwen35_clip = CLIP(
                model_dict={"qwen35_4b": qwen35_model},
                tokenizer_dict={},
            )

        qwen35_clip = self._qwen35_clip

        # Create the engine
        from lib_qwen35.engine import Qwen35AnimaTextProcessingEngine
        qwen35_engine = Qwen35AnimaTextProcessingEngine(
            text_encoder=qwen35_model,
            qwen_tokenizer=qwen35_tokenizer,
            t5_tokenizer=original_t5_tokenizer,
            original_text_encoder=original_text_encoder,
        )

        sd_model = p.sd_model

        # Save original get_learned_conditioning for restoration in postprocess
        if not hasattr(sd_model, '_qwen35_original_get_learned_conditioning'):
            sd_model._qwen35_original_get_learned_conditioning = sd_model.get_learned_conditioning

        @torch.inference_mode()
        def patched_get_learned_conditioning(prompt: list[str]):
            """Replacement for Anima.get_learned_conditioning that uses Qwen3.5 4B."""
            log(f"[Qwen3.5-Anima] patched_get_learned_conditioning called with {len(prompt)} prompt(s)")
            # Load original text encoder first (for LLM adapter weights)
            memory_management.load_model_gpu(original_clip.patcher)
            # Load our Qwen3.5 model
            memory_management.load_model_gpu(qwen35_clip.patcher)
            result = qwen35_engine(prompt)
            log(f"[Qwen3.5-Anima] Conditioning computed: {len(result)} result(s), shape={result[0].shape if result else 'empty'}")
            return result

        # Monkeypatch — fires BEFORE setup_conds() computes conditioning
        sd_model.get_learned_conditioning = patched_get_learned_conditioning

        # Write generation params
        p.extra_generation_params.update({
            "qwen35_encoder": True,
            "q35_model": model_file,
            "q35_calibration": use_calibration,
            "q35_alignment": use_alignment,
            "q35_align_str": alignment_strength,
            "q35_scale": output_scale,
        })

        log(f"[Qwen3.5-Anima] Text encoder replacement active "
            f"(model={model_file}, cal={use_calibration}, align={use_alignment}, "
            f"strength={alignment_strength}, scale={output_scale})")

    def process_before_every_sampling(self, p: StableDiffusionProcessing, enable: bool,
                                       model_file: str, use_calibration: bool, use_alignment: bool,
                                       alignment_strength: float, output_scale: float, **kwargs):
        """Reserve extra GPU memory for the 4B model during sampling."""
        if not enable:
            return
        if self._qwen35_clip is None:
            return

        unet = p.sd_model.forge_objects.unet
        extra_memory = self._qwen35_clip.patcher.model_size()
        unet.model_options.setdefault("extra_preserved_memory_during_sampling", 0)
        unet.model_options["extra_preserved_memory_during_sampling"] = max(
            unet.model_options["extra_preserved_memory_during_sampling"],
            extra_memory
        )

    def postprocess(self, p, processed, *args):
        """Restore original get_learned_conditioning after generation."""
        if hasattr(p.sd_model, '_qwen35_original_get_learned_conditioning'):
            p.sd_model.get_learned_conditioning = p.sd_model._qwen35_original_get_learned_conditioning
            del p.sd_model._qwen35_original_get_learned_conditioning
            # Also invalidate conditioning cache so next gen with extension off
            # doesn't reuse our 4B conditioning
            p.cached_c = [None, None, None]
            p.cached_uc = [None, None, None]
            from modules.processing import StableDiffusionProcessing as SDP
            SDP.cached_c = [None, None, None]
            SDP.cached_uc = [None, None, None]
            log("[Qwen3.5-Anima] Restored original text encoder, cleared conditioning cache")
