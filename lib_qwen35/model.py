"""
Qwen 3.5 4B Hybrid Text Encoder Model for Forge Neo.

Ported from ComfyUI custom node: comfyui-qwen35-anima/__init__.py
All model architecture logic is preserved exactly.

Key adaptations from ComfyUI → Forge Neo:
- comfy.ops.disable_weight_init → torch.nn (Forge manages weight init via ForgeOperations context)
- comfy.ldm.common_dit.rms_norm → torch.nn.functional.rms_norm
- comfy.model_management → backend.memory_management
- safetensors_torch → safetensors.torch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os

import safetensors.torch as safetensors_torch

logger = logging.getLogger(__name__)


# ============================================================================
# RMSNorm helper — replaces comfy.ldm.common_dit.rms_norm
# ============================================================================

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm: x * weight / rms(x). Compatible replacement for comfy.ldm.common_dit.rms_norm."""
    return torch.nn.functional.rms_norm(x, weight.shape, weight, eps)


# ============================================================================
# Model Architecture Components
# ============================================================================

class RMSNorm(nn.Module):
    """RMSNorm with learnable scale."""
    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


class ExpRMSNorm(nn.Module):
    """
    RMSNorm with exp(weight) parameterization.

    Used for the late normalization layer where learned weights are near-zero
    (mean~-0.003). Standard RMSNorm would interpret these as "scale to ~0",
    collapsing all information. With exp(weight), near-zero means exp(0)~1,
    i.e. "nearly identity normalization" with tiny learned perturbations.

    Evidence:
    - All internal RMSNorms have weights centered 0.04-1.11 (normal scaling)
    - ONLY the late norm has weights at -0.003 (different parameterization)
    - Direct weight: diversity=0.003, cross=0.999 (COLLAPSED)
    - exp(weight): diversity=0.821, cross=0.689 (PRESERVED)
    """
    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, torch.exp(self.weight.float()).to(x.dtype), self.eps)


class BiasAdd(nn.Module):
    """Simple module that adds a learnable bias."""
    def __init__(self, dim, device=None, dtype=None):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))

    def forward(self, x):
        return x + self.bias


class SSMBlock(nn.Module):
    """
    Mamba2-style Selective State Space Model block (reference: state-spaces/mamba).

    Architecture verified against reference Mamba2:
    - in_proj_qkv: Linear(2560, 8192) -> conv1d -> silu -> split into x(4096), B(2048), C(2048)
      where d_ssm=4096, ngroups*d_state=2048 each
    - in_proj_z: Linear(2560, 4096) -> gate z that BYPASSES conv1d
    - in_proj_b: Linear(2560, 32) -> per-group B bias / additional modulation
    - in_proj_a: Linear(2560, 32) -> per-group C bias / additional modulation
    - A_log: [32] -> state transition (nheads=32)
    - dt_bias: [32] -> discretization timestep bias (nheads=32)
    - conv1d: depthwise Conv1d(8192, 8192, kernel=4)
    - norm: RMSNorm(128) -> per-head norm (head_dim=128)
    - out_proj: Linear(4096, 2560) -> d_ssm -> hidden_size

    Dimensions: d_ssm=4096, nheads=32, head_dim=128, ngroups=32, d_state=64
    conv_dim = d_ssm + 2*ngroups*d_state = 4096 + 2*32*64 = 8192
    """
    def __init__(self, hidden_size=2560, d_inner=8192, n_groups=32,
                 d_gate=4096, conv_kernel=4, norm_dim=128,
                 device=None, dtype=None, ops=None):
        super().__init__()
        ops = ops or nn
        self.hidden_size = hidden_size
        self.d_inner = d_inner  # conv channels: d_ssm + 2*ngroups*d_state
        self.n_groups = n_groups  # also nheads (1 head per group)
        self.d_ssm = d_gate  # 4096 = nheads * head_dim
        self.head_dim = d_gate // n_groups  # 128
        self.d_state = (d_inner - d_gate) // (2 * n_groups)  # 64

        self.in_proj_qkv = ops.Linear(hidden_size, d_inner, bias=False, device=device, dtype=dtype)
        self.in_proj_z = ops.Linear(hidden_size, d_gate, bias=False, device=device, dtype=dtype)
        self.in_proj_a = ops.Linear(hidden_size, n_groups, bias=False, device=device, dtype=dtype)
        self.in_proj_b = ops.Linear(hidden_size, n_groups, bias=False, device=device, dtype=dtype)

        # Use ops.Conv1d for auto device/dtype casting
        self.conv1d = ops.Conv1d(
            d_inner, d_inner, conv_kernel, groups=d_inner,
            padding=conv_kernel - 1, bias=False, device=device, dtype=dtype
        )

        self.out_proj = ops.Linear(d_gate, hidden_size, bias=False, device=device, dtype=dtype)
        self.norm = RMSNorm(norm_dim, device=device, dtype=dtype)

        self.A_log = nn.Parameter(torch.zeros(n_groups, device=device, dtype=dtype))
        self.dt_bias = nn.Parameter(torch.zeros(n_groups, device=device, dtype=dtype))

    def _ssm_scan(self, x, B_state, C_state, dt_input, D_input):
        """
        Multi-head SSM scan with d_state > 1 (proper Mamba2 recurrence).

        x: [B, L, nheads, head_dim]  (the SSM input, nheads=32, head_dim=128)
        B_state: [B, L, ngroups, d_state]  (input matrix, ngroups=32, d_state=64)
        C_state: [B, L, ngroups, d_state]  (output matrix, ngroups=32, d_state=64)
        dt_input: [B, L, nheads]  (input-dependent discretization step)
        D_input: [B, L, nheads]   (input-dependent skip connection)

        SSM recurrence per head n (in group g):
            dt = softplus(dt_input + dt_bias)
            dA = exp(dt * A)
            h[n] = dA[n] * h[n] + dt[n] * (B[g] outer x[n])
            y[n] = (C[g] . h[n]) + D[n] * x[n]  (skip connection)

        State shape: [batch, nheads, head_dim, d_state]
        Returns: [B, L, nheads, head_dim]
        """
        batch, seq_len, nheads, head_dim = x.shape
        d_state = B_state.shape[-1]
        device = x.device
        compute_dtype = torch.float32

        # Move params to device
        A = -torch.exp(self.A_log.to(device=device).float())  # [nheads] (negative)
        dt_bias = self.dt_bias.to(device=device).float()  # [nheads]

        # State: [batch, nheads, head_dim, d_state]
        h = torch.zeros(batch, nheads, head_dim, d_state, device=device, dtype=compute_dtype)
        outputs = []

        x_f = x.float()
        B_f = B_state.float()  # [B, L, ngroups, d_state]
        C_f = C_state.float()  # [B, L, ngroups, d_state]
        dt_f = dt_input.float()  # [B, L, nheads]
        D_f = D_input.float()   # [B, L, nheads]

        for t in range(seq_len):
            x_t = x_f[:, t]  # [B, nheads, head_dim]
            B_t = B_f[:, t]  # [B, ngroups, d_state]
            C_t = C_f[:, t]  # [B, ngroups, d_state]

            # Input-dependent dt: softplus(dt_input + dt_bias) [B, nheads]
            dt_t = F.softplus(dt_f[:, t] + dt_bias)  # [B, nheads]

            # Discretize: dA = exp(A * dt) per head per batch
            dA_t = torch.exp(dt_t * A.unsqueeze(0))  # [B, nheads]

            # dBx = dt * outer(x_t, B_t): [B, nheads, head_dim, d_state]
            dt_expanded = dt_t.unsqueeze(-1).unsqueeze(-1)  # [B, nheads, 1, 1]
            dBx = dt_expanded * torch.einsum('bnh,bns->bnhs', x_t, B_t)

            # State update: h = dA * h + dBx
            dA_expanded = dA_t.unsqueeze(-1).unsqueeze(-1)  # [B, nheads, 1, 1]
            h = dA_expanded * h + dBx

            # Output: y_t = einsum(h, C_t) over d_state + D * x (skip)
            y_t = torch.einsum('bnhs,bns->bnh', h, C_t)  # [B, nheads, head_dim]
            y_t = y_t + D_f[:, t].unsqueeze(-1) * x_t  # D skip connection

            outputs.append(y_t)

        return torch.stack(outputs, dim=1).to(x.dtype)  # [B, L, nheads, head_dim]

    def forward(self, hidden_states):
        batch, seq_len, _ = hidden_states.shape

        # === Gate (bypasses conv1d, reference Mamba2 pattern) ===
        z = self.in_proj_z(hidden_states)  # [B, L, 4096] - the gate

        # === xBC goes through conv1d ===
        xBC = self.in_proj_qkv(hidden_states)  # [B, L, 8192]

        # in_proj_b -> input-dependent dt (time step for selective SSM)
        # in_proj_a -> input-dependent D (skip connection, no separate D param exists)
        dt_input = self.in_proj_b(hidden_states)  # [B, L, 32] (nheads)
        D_input = self.in_proj_a(hidden_states)   # [B, L, 32] (nheads)

        # Causal 1D convolution + activation
        xBC_conv = xBC.transpose(1, 2)  # [B, 8192, L]
        xBC_conv = self.conv1d(xBC_conv)[..., :seq_len]
        xBC_conv = F.silu(xBC_conv.transpose(1, 2))  # [B, L, 8192]

        # Split conv output: x(d_ssm=4096), B_conv(ngroups*d_state=2048), C_conv(2048)
        x, B_conv, C_conv = torch.split(
            xBC_conv,
            [self.d_ssm, self.n_groups * self.d_state, self.n_groups * self.d_state],
            dim=-1
        )

        # Reshape for SSM
        x = x.reshape(batch, seq_len, self.n_groups, self.head_dim)  # [B, L, 32, 128]
        B_state = B_conv.reshape(batch, seq_len, self.n_groups, self.d_state)  # [B, L, 32, 64]
        C_state = C_conv.reshape(batch, seq_len, self.n_groups, self.d_state)  # [B, L, 32, 64]

        # SSM scan (with input-dependent dt and D)
        y = self._ssm_scan(x, B_state, C_state, dt_input, D_input)  # [B, L, 32, 128]

        # Per-head RMSNorm (norm_dim=128=head_dim)
        y = self.norm(y)

        # Reshape and apply gating: y = norm(y) * silu(z) (RMSNormGated pattern)
        y = y.reshape(batch, seq_len, -1)  # [B, L, 4096]
        y = y * F.silu(z)

        # Output projection
        return self.out_proj(y)


class GatedSelfAttention(nn.Module):
    """
    Self-attention with gated Q projection.

    q_proj outputs Q(4096) + gate(4096) = 8192:
    - 16 attention heads with 256 head_dim
    - 4 KV heads with 256 head_dim (GQA ratio 4)
    - After attention: [B, L, 4096] gated by silu(gate) -> o_proj
    """
    def __init__(self, hidden_size=2560, num_heads=16, num_kv_heads=4,
                 head_dim=256, rope_theta=1000000.0,
                 device=None, dtype=None, ops=None):
        super().__init__()
        ops = ops or nn
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gqa_ratio = num_heads // num_kv_heads
        self.inner_dim = num_heads * head_dim  # 4096

        self.q_proj = ops.Linear(hidden_size, 2 * self.inner_dim, bias=False, device=device, dtype=dtype)
        self.k_proj = ops.Linear(hidden_size, num_kv_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = ops.Linear(hidden_size, num_kv_heads * head_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = ops.Linear(self.inner_dim, hidden_size, bias=False, device=device, dtype=dtype)

        self.q_norm = RMSNorm(head_dim, device=device, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, device=device, dtype=dtype)

    def forward(self, hidden_states, attention_mask=None, freqs_cis=None):
        B, L, _ = hidden_states.shape

        # Q projection with gate
        qg = self.q_proj(hidden_states)  # [B, L, 8192]
        q, gate = qg.chunk(2, dim=-1)  # [B, L, 4096] each

        # Reshape to heads
        q = q.view(B, L, self.num_heads, self.head_dim)  # [B, L, 16, 256]
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)  # [B, L, 4, 256]
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)  # [B, L, 4, 256]

        # Per-head norms
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose for attention: [B, H, L, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        if freqs_cis is not None:
            cos, sin = freqs_cis
            q = _apply_rotary_emb(q, cos, sin)
            k = _apply_rotary_emb(k, cos, sin)

        # GQA: expand K, V
        k = k.repeat_interleave(self.gqa_ratio, dim=1)  # [B, 16, L, 256]
        v = v.repeat_interleave(self.gqa_ratio, dim=1)  # [B, 16, L, 256]

        # Attention (ensure mask dtype matches query)
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.to(dtype=q.dtype)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=(attention_mask is None)
        )  # [B, 16, L, 256]

        # Reshape and gate
        attn_out = attn_out.transpose(1, 2).reshape(B, L, self.inner_dim)  # [B, L, 4096]
        attn_out = attn_out * F.silu(gate)

        return self.o_proj(attn_out)


def _apply_rotary_emb(x, cos, sin):
    """Apply rotary position embeddings."""
    # x: [B, H, L, D]
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


def _precompute_freqs_cis(head_dim, max_seq_len, theta=1000000.0, device=None, dtype=None):
    """Precompute RoPE frequencies."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, L, D/2]
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)  # [1, 1, L, D/2]
    # Duplicate for full head_dim
    cos = cos.repeat(1, 1, 1, 2)  # [1, 1, L, D]
    sin = sin.repeat(1, 1, 1, 2)  # [1, 1, L, D]
    if dtype is not None:
        cos = cos.to(dtype)
        sin = sin.to(dtype)
    return cos, sin


class MLP(nn.Module):
    """SwiGLU MLP."""
    def __init__(self, hidden_size=2560, intermediate_size=9216,
                 device=None, dtype=None, ops=None):
        super().__init__()
        ops = ops or nn
        self.gate_proj = ops.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = ops.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = ops.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HybridBlock(nn.Module):
    """
    A single transformer block that uses either SSM or self-attention.
    """
    def __init__(self, hidden_size=2560, intermediate_size=9216,
                 use_ssm=True, has_mlp=True,
                 device=None, dtype=None, ops=None):
        super().__init__()
        self.use_ssm = use_ssm
        self.has_mlp = has_mlp

        self.input_layernorm = RMSNorm(hidden_size, device=device, dtype=dtype)

        if use_ssm:
            self.linear_attn = SSMBlock(
                hidden_size=hidden_size,
                device=device, dtype=dtype, ops=ops
            )
        else:
            self.self_attn = GatedSelfAttention(
                hidden_size=hidden_size,
                device=device, dtype=dtype, ops=ops
            )

        if has_mlp:
            self.post_attention_layernorm = RMSNorm(hidden_size, device=device, dtype=dtype)
            self.mlp = MLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                device=device, dtype=dtype, ops=ops
            )

    def forward(self, x, attention_mask=None, freqs_cis=None):
        # Pre-norm + attention/SSM
        residual = x
        x_norm = self.input_layernorm(x)

        if self.use_ssm:
            x = residual + self.linear_attn(x_norm)
        else:
            x = residual + self.self_attn(x_norm, attention_mask=attention_mask, freqs_cis=freqs_cis)

        # Pre-norm + MLP
        if self.has_mlp:
            residual = x
            x = residual + self.mlp(self.post_attention_layernorm(x))

        return x


# ============================================================================
# Full Qwen 3.5 Hybrid Model
# ============================================================================

class Qwen35HybridModel(nn.Module):
    """
    Qwen 3.5 4B Hybrid Text Encoder.

    This is the official Qwen3.5-4B text backbone (confirmed by matching config.json
    from Qwen/Qwen3.5-4B: same vocab_size=248320, hidden_size=2560, 32 layers,
    same linear_attention/full_attention pattern every 4 layers).

    32 layers with alternating SSM/attention:
    - SSM layers: 0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22, 24,25,26, 28,29,30
    - Self-attention layers: 3,7,11,15,19,23,27,31
    - Layer 31: self-attention only (no MLP)
    - Final norm: Linear(2560->1024) + ExpRMSNorm(1024) + SiLU + Linear(1024->1024)

    The late norm uses exp(weight) parameterization for RMSNorm. The learned weights
    are near-zero (~-0.003), which with exp() gives scale ~ 0.997 ~ 1.0 (near-identity).
    Standard w*norm would collapse all tokens to the same vector (diversity=0.003).
    With exp(w)*norm, token diversity is preserved (diversity=0.82).
    """
    SELF_ATTN_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}
    NUM_LAYERS = 32
    HIDDEN_SIZE = 2560
    INTERMEDIATE_SIZE = 9216
    VOCAB_SIZE = 248320
    OUTPUT_DIM = 1024
    HEAD_DIM = 256  # For RoPE
    ROPE_THETA = 1000000.0
    DEFAULT_OUTPUT_SCALE = 1.0  # Raw output; use slider to experiment

    def __init__(self, config_dict=None, dtype=None, device=None, operations=None,
                 extension_dir=None):
        super().__init__()
        if config_dict is None:
            config_dict = {}
        ops = operations or nn

        self.num_layers = self.NUM_LAYERS
        self.dtype = dtype

        # Extension directory for loading calibration/alignment files
        self._extension_dir = extension_dir or os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        # Token embeddings
        self.embed_tokens = ops.Embedding(
            self.VOCAB_SIZE, self.HIDDEN_SIZE, device=device, dtype=dtype
        )

        # Transformer blocks
        self.layers = nn.ModuleList()
        for i in range(self.NUM_LAYERS):
            use_ssm = (i not in self.SELF_ATTN_LAYERS)
            has_mlp = (i != 31)  # Layer 31 has no MLP
            self.layers.append(HybridBlock(
                hidden_size=self.HIDDEN_SIZE,
                intermediate_size=self.INTERMEDIATE_SIZE,
                use_ssm=use_ssm,
                has_mlp=has_mlp,
                device=device, dtype=dtype, ops=ops
            ))

        # Output projection: Linear(2560->1024) + ExpRMSNorm + SiLU + Linear(1024->1024)
        self.norm = nn.Sequential(
            ops.Linear(self.HIDDEN_SIZE, self.OUTPUT_DIM, bias=True, device=device, dtype=dtype),
            ExpRMSNorm(self.OUTPUT_DIM, device=device, dtype=dtype),
            nn.SiLU(),
            ops.Linear(self.OUTPUT_DIM, self.OUTPUT_DIM, bias=True, device=device, dtype=dtype),
        )

        # Output scaling (set externally via config_dict or after construction)
        self._output_scale = config_dict.get("output_scale", self.DEFAULT_OUTPUT_SCALE)

        # Per-dimension affine calibration (computed by calibrate.py)
        self._calibration_scale = None  # [1024]
        self._calibration_bias = None   # [1024]
        self._use_calibration = config_dict.get("use_calibration", False)
        if self._use_calibration:
            self._load_calibration()

        # Procrustes rotation alignment (computed by compute_alignment.py)
        # Rotates 4B embedding directions to match 0.6B's concept space
        # This is a full 1024x1024 orthogonal rotation — preserves distances
        # but re-orients spatial/pose concepts to match what the adapter expects
        self._rotation_matrix = None   # [1024, 1024]
        self._rotation_mean_4b = None  # [1024]
        self._rotation_mean_06b = None # [1024]
        self._use_alignment = config_dict.get("use_alignment", False)
        self._alignment_strength = config_dict.get("alignment_strength", 1.0)  # 0..1 blend
        if self._use_alignment:
            self._load_alignment()

        # Pending visual embeddings for vision-text encoding
        # Set externally before forward(), cleared after
        self._pending_visual_embeds = None
        self._pending_vision_weight = 1.0   # Scale factor applied AFTER norm projection
        self._pending_vision_mode = "add"   # "add", "concat", or "replace_padding"

    @property
    def _calibration_file(self):
        return os.path.join(self._extension_dir, "calibration_params.safetensors")

    @property
    def _alignment_file(self):
        return os.path.join(self._extension_dir, "rotation_matrix.safetensors")

    def _load_calibration(self):
        """Load per-dimension affine calibration parameters."""
        cal_file = self._calibration_file
        if os.path.exists(cal_file):
            try:
                cal = safetensors_torch.load_file(cal_file)
                self._calibration_scale = cal["scale"].float()  # [1024]
                self._calibration_bias = cal["bias"].float()    # [1024]
                logger.info(f"[Qwen3.5-Anima] Loaded calibration: scale mean={self._calibration_scale.mean():.3f}, bias mean={self._calibration_bias.mean():.3f}")
            except Exception as e:
                logger.warning(f"[Qwen3.5-Anima] Failed to load calibration: {e}")
                self._use_calibration = False
        else:
            logger.warning(f"[Qwen3.5-Anima] Calibration file not found: {cal_file}")
            self._use_calibration = False

    def _load_alignment(self):
        """Load Procrustes rotation matrix for 4B→0.6B concept alignment."""
        align_file = self._alignment_file
        if os.path.exists(align_file):
            try:
                data = safetensors_torch.load_file(align_file)
                self._rotation_matrix = data["rotation"].float()     # [1024, 1024]
                self._rotation_mean_4b = data["mean_4b"].float()     # [1024]
                self._rotation_mean_06b = data["mean_06b"].float()   # [1024]
                # Use slogdet for numerical stability (det() on 1024x1024 float32 underflows to 0)
                sign, logabsdet = torch.linalg.slogdet(self._rotation_matrix.double())
                logger.info(
                    f"[Qwen3.5-Anima] Loaded Procrustes alignment: "
                    f"R shape={self._rotation_matrix.shape}, "
                    f"det={sign.item():+.0f}, "
                    f"mean_4b L2={self._rotation_mean_4b.norm():.1f}, "
                    f"mean_06b L2={self._rotation_mean_06b.norm():.1f}"
                )
            except Exception as e:
                logger.warning(f"[Qwen3.5-Anima] Failed to load alignment: {e}")
                self._use_alignment = False
        else:
            logger.warning(f"[Qwen3.5-Anima] Alignment file not found: {align_file}")
            self._use_alignment = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.embed_tokens = embeddings

    def forward(self, input_ids, attention_mask=None, embeds=None, num_tokens=None,
                intermediate_output=None, final_layer_norm_intermediate=True,
                dtype=None, embeds_info=None, **kwargs):
        # Get embeddings
        if embeds is not None:
            x = embeds
        else:
            x = self.embed_tokens(input_ids).to(dtype=dtype or torch.float32)

        # NOTE: Visual features (from ViT) are NOT fed through the backbone.
        # The backbone was trained on text only — visual prefixes would be
        # treated as noise, corrupting both visual and text representations.
        # Instead, visual features are projected through the norm (2560→1024)
        # separately and prepended to the output AFTER backbone processing.

        seq_len = x.shape[1]

        # Precompute RoPE frequencies for self-attention layers
        freqs_cis = _precompute_freqs_cis(
            self.HEAD_DIM, seq_len,
            theta=self.ROPE_THETA,
            device=x.device, dtype=x.dtype
        )

        # Build causal attention mask for self-attention layers
        attn_mask = None
        if attention_mask is not None:
            mask_fill = torch.finfo(x.dtype).min / 4
            causal = torch.empty(
                seq_len, seq_len, dtype=x.dtype, device=x.device
            ).fill_(mask_fill).triu_(1)
            pad_mask = 1.0 - attention_mask.to(x.dtype).reshape(
                attention_mask.shape[0], 1, -1, attention_mask.shape[-1]
            ).expand(attention_mask.shape[0], 1, seq_len, attention_mask.shape[-1])
            pad_mask = pad_mask.masked_fill(pad_mask.to(torch.bool), mask_fill)
            attn_mask = causal + pad_mask
        elif seq_len > 1:
            mask_fill = torch.finfo(x.dtype).min / 4
            attn_mask = torch.empty(
                seq_len, seq_len, dtype=x.dtype, device=x.device
            ).fill_(mask_fill).triu_(1)

        # Process through layers (text only)
        intermediate = None
        for i, layer in enumerate(self.layers):
            x = layer(x, attention_mask=attn_mask, freqs_cis=freqs_cis)

            # Capture intermediate output if requested
            if intermediate_output is not None:
                if isinstance(intermediate_output, int) and i == intermediate_output:
                    intermediate = x.clone()
                elif isinstance(intermediate_output, list) and i in intermediate_output:
                    if intermediate is None:
                        intermediate = {}
                    intermediate[i] = x.clone()

        # ── Visual injection in 2560-dim space (BEFORE norm) ──────────────
        # For "add" mode, inject here so the norm pipeline processes the
        # combined text+visual signal.  Different images perturb the hidden
        # states in different *directions*, and those directional differences
        # survive the norm even though magnitudes get normalised.
        # For "concat" / "replace_padding", we project visual tokens through
        # the full norm separately (they exist as distinct tokens).
        n_visual = 0
        _vis_for_post_norm = None  # visual embeds held for concat/replace_padding
        if self._pending_visual_embeds is not None:
            visual = self._pending_visual_embeds.to(device=x.device, dtype=x.dtype)
            n_visual = visual.shape[1]
            mode = self._pending_vision_mode
            weight = self._pending_vision_weight

            if mode == "add":
                # Mean-pool 196 ViT patches → single 2560-dim style vector
                style_vec_2560 = visual.mean(dim=1, keepdim=True)  # [B, 1, 2560]
                # Normalise style vector to match text hidden-state magnitude.
                # Raw ViT style is ~4× smaller than text (L2 3.5 vs 15),
                # so without scaling the perturbation is negligible after norm.
                # With this, weight=1.0 means "visual same magnitude as text".
                text_scale = x.norm(dim=-1).mean().clamp(min=1e-6)
                style_scale = style_vec_2560.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                style_vec_2560 = style_vec_2560 * (text_scale / style_scale)
                # Add as residual to every text hidden state before norm
                x = x + weight * style_vec_2560
                n_visual = 0  # no extra tokens in output
            else:
                # Keep visual embeds for projection after norm
                _vis_for_post_norm = (visual, n_visual, mode, weight)

        # Apply output projection (Linear -> ExpRMSNorm -> SiLU -> Linear)
        # Maps 2560-dim hidden states to 1024-dim adapter space
        x = self.norm(x)

        # Procrustes alignment: rotation + optional bias shift.
        # Decomposed into two parts:
        #   1. Rotation: R @ (x - m4b) — always applied when alignment is on.
        #      This re-orients concept directions (e.g. "from side") to match
        #      what the 0.6B-trained adapter expects.  Preserves norms.
        #   2. Bias shift: re-center from m4b to m06b (blended by alignment_strength).
        #      m06b has L2=70 vs m4b L2=5, so full shift dramatically changes
        #      output magnitude.  strength=0 keeps 4B's own scale, strength=1
        #      shifts fully to 0.6B's characteristic bias.
        if self._use_alignment and self._rotation_matrix is not None:
            R = self._rotation_matrix.to(device=x.device, dtype=x.dtype)
            m4b = self._rotation_mean_4b.to(device=x.device, dtype=x.dtype)
            m06b = self._rotation_mean_06b.to(device=x.device, dtype=x.dtype)
            alpha = self._alignment_strength
            # Always rotate (fixes concept directions)
            x_rotated = torch.einsum('ij,...j->...i', R, x - m4b)
            # Blend the re-centering: (1-α)*m4b + α*m06b
            x = x_rotated + (1.0 - alpha) * m4b + alpha * m06b

        # Per-dimension affine calibration
        if self._use_calibration and self._calibration_scale is not None:
            cal_scale = self._calibration_scale.to(device=x.device, dtype=x.dtype)
            cal_bias = self._calibration_bias.to(device=x.device, dtype=x.dtype)
            x = x * cal_scale + cal_bias

        # Additional uniform scaling
        if self._output_scale != 1.0:
            x = x * self._output_scale

        # ── Post-norm visual injection (concat / replace_padding) ─────────
        if _vis_for_post_norm is not None:
            visual, n_visual, mode, weight = _vis_for_post_norm

            # Project visual tokens through full norm (same as text)
            visual_projected = self.norm(visual)  # [B, N_vis, 1024]

            # Procrustes rotation on visual tokens too (same decomposition)
            if self._use_alignment and self._rotation_matrix is not None:
                R = self._rotation_matrix.to(device=visual_projected.device, dtype=visual_projected.dtype)
                m4b = self._rotation_mean_4b.to(device=visual_projected.device, dtype=visual_projected.dtype)
                m06b = self._rotation_mean_06b.to(device=visual_projected.device, dtype=visual_projected.dtype)
                alpha = self._alignment_strength
                vp_rotated = torch.einsum('ij,...j->...i', R, visual_projected - m4b)
                visual_projected = vp_rotated + (1.0 - alpha) * m4b + alpha * m06b

            if mode == "concat":
                if weight != 1.0:
                    visual_projected = visual_projected * weight
                x = torch.cat([visual_projected, x], dim=1)

            elif mode == "replace_padding":
                B, T, D = x.shape
                tok_norms = x.norm(dim=-1)  # [B, T]
                non_pad = (tok_norms[0] > 1.0).nonzero(as_tuple=True)[0]
                first_pad = (non_pad[-1].item() + 1) if len(non_pad) > 0 else 0
                n_pad_slots = T - first_pad

                if n_pad_slots > 0:
                    if weight != 1.0:
                        visual_projected = visual_projected * weight
                    if n_visual <= n_pad_slots:
                        x[:, first_pad:first_pad + n_visual, :] = visual_projected
                    else:
                        chunk_size = n_visual // n_pad_slots
                        for s in range(n_pad_slots):
                            start = s * chunk_size
                            end = min(start + chunk_size, n_visual) if s < n_pad_slots - 1 else n_visual
                            x[:, first_pad + s, :] = visual_projected[:, start:end, :].mean(dim=1)
                n_visual = 0

            else:
                logger.warning(f"[Qwen3.5-Vision] Unknown vision mode '{mode}', falling back to 'add'")
                style_vec = visual_projected.mean(dim=1, keepdim=True)
                x = x + weight * style_vec
                n_visual = 0

        self._last_n_visual = n_visual

        if intermediate is not None:
            return x, intermediate
        return x, None
