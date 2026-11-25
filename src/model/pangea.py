#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""PANGEA: the encoder that produces the landscape.

`PLAN.md`, Aim 1: "PANGEA takes a sequence, a molecule class and a vector of physical
conditions, and returns the per-position landscape with a predicted mean and variance for
every channel."

Four structural commitments from the plan are implemented here rather than approximated,
because each one is load-bearing and each would be easy to quietly drop.

**1. Reverse-complement equivariance is architectural, not learned.** The plan: "A
channel predicted with the wrong parity is wrong in a way no validation correlation will
reveal." So the trunk is applied twice with shared weights, once to the sequence and once
to its reverse complement, and the two are combined by each channel's *declared* parity:

    SYMMETRIC      (f + reverse(g)) / 2
    ANTISYMMETRIC  (f - reverse(g)) / 2
    STRANDED       f on the sense head, reverse(g) on the antisense head
    COMPOSITE      f alone

This makes equivariance exact to floating point for the first two classes, for any
weights, at any point in training. It costs a second forward pass, which is the honest
price; a single-pass approximation would make equivariance something to verify rather
than something guaranteed. COMPOSITE is passed through because those channels transform
under an index shift rather than a sign -- see `channels.Parity` -- and averaging them
would be wrong, not conservative.

**2. The output head is low rank, and the rank is a measurement.** The plan: "Rather than
projecting the trunk straight to the full channel set, it projects to a bottleneck of
rank r and then out... the smallest rank at which per-channel accuracy saturates is an
independent estimate of the intrinsic dimensionality of nucleic acid biophysics, obtained
for free while fitting."

**3. Outputs are heteroscedastic and the family weights are learned.** Every channel gets
a mean and a log-variance, trained under a Gaussian negative log-likelihood, and the
per-family loss weights are learned by uncertainty weighting rather than hand-set,
because "channels differ by orders of magnitude in scale and in label noise".

**4. The pyramid is a pyramid.** Per-base-pair channels come off the trunk at full
resolution, window channels off pooled representations at each rung of the ladder, and
whole-sequence channels off a final pool.

**What this file does not pretend to do.** The plan specifies a second wide branch with a
5 to 10 kb receptive field feeding the mesoscale channels. It is implemented and
`use_wide_branch` controls it, but the current fragment library is 61 bp, so there is no
context for it to see and it is off by default. Training it needs a longer-fragment
library. Stating that here is the point: a wide branch silently fed 61 bp windows would
train to nothing and look like a working component.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# Base order used everywhere. The complement is a permutation of these four, which is
# what makes reverse-complementing a one-hot tensor a flip on both axes.
BASES = "ACGT"
COMPLEMENT_PERM = torch.tensor([3, 2, 1, 0])      # A<->T, C<->G under "ACGT"


def one_hot(seqs: Sequence[str], device=None) -> torch.Tensor:
    """(batch, 4, length) one-hot. Any non-ACGT base becomes all-zero, not a guess."""
    idx = {b: i for i, b in enumerate(BASES)}
    n, L = len(seqs), len(seqs[0])
    x = torch.zeros(n, 4, L, device=device)
    for i, s in enumerate(seqs):
        for j, b in enumerate(s.upper()):
            k = idx.get(b)
            if k is not None:
                x[i, k, j] = 1.0
    return x


def reverse_complement(x: torch.Tensor) -> torch.Tensor:
    """Reverse-complement a one-hot (batch, 4, length) tensor.

    Complementing permutes the channel axis and reversing flips the position axis, so
    this is exact and has no learned part. Doing it on the tensor rather than on strings
    is what lets the second pass stay inside the autograd graph.
    """
    return x.flip(-1)[:, COMPLEMENT_PERM.to(x.device), :]


# ======================================================================================
# Conditioning
# ======================================================================================

class FiLM(nn.Module):
    """Feature-wise linear modulation from the condition and molecule-class vector.

    The plan puts conditions and molecule class into "FiLM layers at each block in both
    branches", so that "one set of weights serves every condition" and the model "learns
    how the physics responds to the environment rather than memorizing one environment".
    The scale is parameterised as 1 + delta so an all-zero condition vector leaves the
    features untouched, which keeps the untrained model well behaved and makes the
    no-condition case exactly the identity.
    """

    def __init__(self, n_cond: int, n_feat: int):
        super().__init__()
        self.proj = nn.Linear(n_cond, 2 * n_feat)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.n_feat = n_feat

    def forward(self, h: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        if cond is None:
            return h
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return h * (1.0 + gamma).unsqueeze(-1) + beta.unsqueeze(-1)


class DilatedBlock(nn.Module):
    """One residual block of the local trunk: dilated conv, norm, GELU, FiLM."""

    def __init__(self, width: int, dilation: int, n_cond: int, kernel: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(width, width, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 1)
        self.norm1 = nn.GroupNorm(8, width)
        self.norm2 = nn.GroupNorm(8, width)
        self.film = FiLM(n_cond, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        r = self.conv1(F.gelu(self.norm1(h)))
        r = self.film(r, cond)
        r = self.conv2(F.gelu(self.norm2(r)))
        return h + self.drop(r)


class LocalTrunk(nn.Module):
    """Dilated residual stack. The plan asks for a receptive field near 300 base pairs.

    With kernel 3 and dilations doubling to `max_dilation` and then repeating, the
    receptive field is 1 + 2 * sum(dilations), which `receptive_field` reports rather
    than leaving to be worked out from the hyperparameters.
    """

    def __init__(self, width: int = 192, n_blocks: int = 12, n_cond: int = 8,
                 max_dilation: int = 32, kernel: int = 3, dropout: float = 0.0):
        super().__init__()
        self.stem = nn.Conv1d(4, width, kernel, padding=kernel // 2)
        self.dilations: List[int] = []
        d = 1
        for _ in range(n_blocks):
            self.dilations.append(d)
            d = d * 2 if d < max_dilation else 1
        self.blocks = nn.ModuleList(
            DilatedBlock(width, dl, n_cond, kernel, dropout) for dl in self.dilations)
        self.out_norm = nn.GroupNorm(8, width)
        self.width = width
        self.kernel = kernel

    def receptive_field(self) -> int:
        return 1 + (self.kernel - 1) * (1 + sum(self.dilations))

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.stem(x)
        for b in self.blocks:
            h = b(h, cond)
        return F.gelu(self.out_norm(h))


class WideBranch(nn.Module):
    """Strided branch for channels that genuinely need kilobase context.

    Implemented, and **off by default**, because the current fragment library is 61 bp.
    The plan gives this branch a 5 to 10 kb receptive field to feed persistence length
    over long windows, nucleosome free energy, supercoiling response and R-loop
    formation. Fed 61 bp it would learn nothing and still look like a working component,
    which is the failure this note exists to prevent.
    """

    def __init__(self, width: int = 96, n_blocks: int = 8, n_cond: int = 8,
                 stride: int = 16):
        super().__init__()
        self.down = nn.Conv1d(4, width, stride * 2 + 1, stride=stride,
                              padding=stride)
        self.blocks = nn.ModuleList(
            DilatedBlock(width, 2 ** i, n_cond) for i in range(n_blocks))
        self.width = width

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor],
                out_len: int) -> torch.Tensor:
        h = self.down(x)
        for b in self.blocks:
            h = b(h, cond)
        return F.interpolate(h, size=out_len, mode="linear", align_corners=False)

    def receptive_field(self) -> int:
        """Receptive field in bp: (1 + 2 sum of dilations) strided steps plus the stride kernel.
        With the defaults (stride 16, kernel-3 blocks at dilations 1..128) 511 x 16 + 33 = 8,209 bp,
        inside the plan's 5 to 10 kb. (The first defaults, stride 4 and dilations 1..32, reached
        only 517 bp: a wide branch narrower than the local trunk's own ~300 bp scale.)"""
        k = self.down.kernel_size[0]; st = self.down.stride[0]
        return (1 + 2 * sum(2 ** i for i in range(len(self.blocks)))) * st + k


# ======================================================================================
# Heads
# ======================================================================================

class LowRankHead(nn.Module):
    """Project features through a rank-r bottleneck, then out to the channel set, twice.

    Two outputs per channel, a mean and a log-variance, so the model is heteroscedastic
    as the plan requires. The bottleneck is shared between them: the claim being encoded
    is that the landscape has few real degrees of freedom, and that claim applies to how
    uncertain the model is as much as to what it predicts.

    `rank` is the measurement the plan asks for. Sweeping it and finding where per-channel
    accuracy saturates is an estimate of the intrinsic dimensionality of the landscape,
    obtained while fitting rather than from a separate analysis.
    """

    def __init__(self, n_feat: int, n_channels: int, rank: int = 64):
        super().__init__()
        self.bottleneck = nn.Conv1d(n_feat, rank, 1)
        self.mean = nn.Conv1d(rank, n_channels, 1)
        self.logvar = nn.Conv1d(rank, n_channels, 1)
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)
        self.rank = rank
        self.n_channels = n_channels

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.bottleneck(h)
        # Clamped so a confident-but-wrong channel cannot drive the NLL to minus
        # infinity early in training and stall every other channel's gradient.
        return self.mean(z), self.logvar(z).clamp(-10.0, 10.0)


@dataclass
class ChannelPlan:
    """Which channels the model predicts, and how each behaves under reverse complement.

    Built from the dataset's own metadata (or the registry, where the physics is
    installed), so the model's output layout and the target layout are the same object.
    `per_bp_rc_shift` is the measured index map: position i reflects to n-1-i+shift.
    """

    per_bp_names: List[str]
    global_names: List[str]
    per_bp_parity: List[str]
    global_parity: List[str]
    per_bp_family: List[str]
    global_family: List[str]
    per_bp_rc_shift: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.per_bp_rc_shift:
            self.per_bp_rc_shift = [0] * len(self.per_bp_names)
        if len(self.per_bp_rc_shift) != len(self.per_bp_names):
            raise ValueError("per_bp_rc_shift must have one entry per per-bp channel")

    @staticmethod
    def from_registry(families: Sequence[str] = ("B", "C", "D", "E", "F", "L")
                      ) -> "ChannelPlan":
        import sys, os
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "physics"))
        import landscape as LS
        from channels import REGISTRY, Resolution

        pb = LS.channel_names(Resolution.PER_BP, families)
        gl = LS.channel_names(Resolution.GLOBAL, families)
        return ChannelPlan(
            per_bp_names=pb, global_names=gl,
            per_bp_parity=[REGISTRY[n].parity.value for n in pb],
            global_parity=[REGISTRY[n].parity.value for n in gl],
            per_bp_family=[REGISTRY[n].family for n in pb],
            global_family=[REGISTRY[n].family for n in gl],
            per_bp_rc_shift=[int(REGISTRY[n].rc_shift) for n in pb])

    @staticmethod
    def from_metadata(per_bp_names, global_names, per_bp_parity, global_parity,
                      per_bp_family, global_family, per_bp_rc_shift=None
                      ) -> "ChannelPlan":
        """Rebuild from a dataset's stored attributes, with no physics import."""
        return ChannelPlan(list(per_bp_names), list(global_names),
                           list(per_bp_parity), list(global_parity),
                           list(per_bp_family), list(global_family),
                           list(per_bp_rc_shift) if per_bp_rc_shift is not None else [])



def model_kwargs_from_args(args: dict) -> dict:
    """PANGEA constructor arguments from a run's stored training arguments (config.json).

    The one place that maps training flags to the model, so evaluation scripts rebuild
    exactly the trained architecture; defaults reproduce runs made before a flag existed.
    """
    return dict(width=args["width"], n_blocks=args["blocks"], rank=args["rank"],
                window_ladder=args["windows"], max_dilation=args["max_dilation"],
                dropout=args["dropout"], n_attn=args["attn_layers"],
                attn_heads=args["attn_heads"],
                global_readout=args.get("global_readout", "linear"),
                global_queries=args.get("global_queries", 4),
                pair_bias=args.get("pair_bias", False),
                pair_channels=args.get("pair_channels", 32),
                pair_layers=args.get("pair_layers", 4),
                pair_module=args.get("pair_module", "conv"),
                pair_heads=args.get("pair_heads", 4),
                strand_tie=args.get("strand_tie", False),
                pair_bpp=args.get("pair_bpp", False),
                wc_tie_global=tuple(args.get("wc_tie_global") or ()))


def _biased_encoder_layer(layer: nn.TransformerEncoderLayer, z: torch.Tensor,
                          bias: torch.Tensor) -> torch.Tensor:
    """A pre-norm encoder layer with a per-sample, per-head additive attention bias.

    Computed explicitly from the layer's own weights rather than by passing the bias
    as `src_mask`: PyTorch's fused inference path returned NaN for a (batch x heads)
    3-D float mask in eval mode while the training path was correct, which would have
    corrupted exactly the evaluation numbers. This is the same arithmetic as the
    layer's slow path (norm_first=True), checked against it in the tests.
    """
    attn = layer.self_attn
    b, n, e = z.shape
    h = attn.num_heads
    d = e // h
    x = layer.norm1(z)
    qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
    q, k, v = (t.view(b, n, h, d).transpose(1, 2) for t in qkv.chunk(3, dim=-1))
    o = F.scaled_dot_product_attention(
        q, k, v, attn_mask=bias.to(q.dtype),
        dropout_p=attn.dropout if attn.training else 0.0)
    o = attn.out_proj(o.transpose(1, 2).reshape(b, n, e))
    z = z + layer.dropout1(o)
    x = layer.norm2(z)
    return z + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(x)))))


def _attention_dtype(z: torch.Tensor) -> torch.dtype:
    """The dtype attention will compute in, so an additive mask can match it."""
    if z.is_cuda and torch.is_autocast_enabled():
        try:
            return torch.get_autocast_dtype("cuda")
        except AttributeError:                                  # older torch
            return torch.get_autocast_gpu_dtype()
    return z.dtype



class PairBias(nn.Module):
    """Per-head attention biases from a 2-D convolution over the base-pair map.

    Family L was the weakest family by a wide margin (per-base-pair median R-squared
    0.52 against 0.997 and above for B, C, E and F): positional entropy, unpaired
    probability, ensemble diversity and hairpin sizes are properties of which distant
    positions can pair, and the trunk has no direct representation of pairs. A stem is a
    run of complementary pairs along an anti-diagonal of the pair map, which is a local
    pattern in two dimensions, so the map is built from the identities of the two bases
    and their signed separation and passed through a small dilated 2-D convolution --
    the representation RNA structure predictors such as UFold use -- and projected to one
    additive bias per attention head per layer.

    The output projection starts at zero, so at initialisation the attention is exactly
    the baseline's; the bias is learned only as far as it helps. It is computed from each
    pass's own input, so the forward/reverse merge that enforces parity is unaffected.
    Cost at 61 bp is small: a 61 x 61 map at `channels` wide.
    """

    def __init__(self, n_bias: int, channels: int = 32, n_layers: int = 4,
                 max_dist: int = 64, bpp_head: bool = False):
        super().__init__()
        self.max_dist = max_dist
        self.n_bias = n_bias
        self.pair_embed = nn.Embedding(25, channels)            # (base_i, base_j), 5 x 5
        self.dist_embed = nn.Embedding(2 * max_dist + 1, channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for k in range(n_layers):
            d = (1, 2, 4, 8)[k % 4]
            self.convs.append(nn.Conv2d(channels, channels, 3, padding=d, dilation=d))
            self.norms.append(nn.GroupNorm(1, channels))
        self.out = nn.Conv2d(channels, n_bias, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # Optional base-pair head (as in PairStack): this pass's own transcript's pair
        # logits, started at the measured prior. Off by default, so every model built
        # before it existed rebuilds with exactly its old parameters.
        self.bpp_out = None
        if bpp_head:
            self.bpp_out = nn.Conv2d(channels, 1, 1)
            nn.init.constant_(self.bpp_out.bias, BPP_PRIOR_LOGIT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: one-hot (B, 4, L) -> biases (B, n_bias, L, L)."""
        b, _, n = x.shape
        known = x.sum(dim=1) > 0
        idx = torch.where(known, x.argmax(dim=1), torch.full_like(known, 4, dtype=torch.long))
        pair = idx[:, :, None] * 5 + idx[:, None, :]                       # (B, L, L)
        pos = torch.arange(n, device=x.device)
        rel = (pos[None, :] - pos[:, None]).clamp(-self.max_dist, self.max_dist) \
            + self.max_dist                                                # (L, L)
        z = (self.pair_embed(pair) + self.dist_embed(rel)[None]).permute(0, 3, 1, 2)
        for conv, norm in zip(self.convs, self.norms):
            z = z + conv(F.gelu(norm(z)))
        if self.bpp_out is not None:
            logits = self.bpp_out(z).squeeze(1).float()
            self.last_bpp_logits = 0.5 * (logits + logits.transpose(1, 2))
        return self.out(z)



# (plus prefix, minus prefix): channel pairs with minus(x) = plus(rc x) exactly, same frame
STRAND_PAIRS = (("L_p_", "L_m_"), ("K_hybp_", "K_hybm_"), ("J_p_", "J_m_"), ("N_p_", "N_m_"),
                # Family A: sector 5 / 4 and the Crick phosphate, stored in their own frame
                # (electrostatics_family.py), so A_s5(x) = A_s1(rc x) like L_m = L_p(rc).
                ("A_s1_", "A_s5_"), ("A_s2_", "A_s4_"), ("A_W_", "A_C_"),
                # Family K's A-form electrostatics on the RNA duplex: Family A's convention
                # (k_electrostatics.py); the hybrids are covered by K_hybp_/K_hybm_ above
                ("K_rna_es_s1_", "K_rna_es_s5_"), ("K_rna_es_s2_", "K_rna_es_s4_"), ("K_rna_es_W_", "K_rna_es_C_"),
                # Family N's modification electrostatics, per transcript in its own frame
                ("Nes_p_", "Nes_m_"),
                # Family A extras: dielectric image sectors, Family A's own-frame convention
                ("Ax_s1_", "Ax_s5_"), ("Ax_s2_", "Ax_s4_"))
MIN_PAIR_SEPARATION = 4   # ViennaRNA's minimum hairpin loop of 3 nt: j - i >= 4
BPP_PRIOR_LOGIT = -4.7    # log-odds of the mean pairable base-pair probability (0.0087)


def pairable_mask(n: int, device=None) -> torch.Tensor:
    """(n, n) True where positions i, j can form a base pair at all."""
    i = torch.arange(n, device=device)
    return (i[None, :] - i[:, None]).abs() >= MIN_PAIR_SEPARATION


def bpp_functionals(logits: torch.Tensor) -> torch.Tensor:
    """(B, N, N) pair logits -> (B, N, 3): per-base paired probability, positional entropy
    (bits) and ensemble-diversity contribution sum_j p_ij (1 - p_ij).

    The same arithmetic as ViennaRNA's targets, applied to sigmoid(logits) on the pairable
    positions: unpaired = 1 - sum_j p_ij and the positional entropy match the stored
    per-bp targets to 1e-7, and the third column sums over positions to the mean
    base-pair distance (`mean_bp_distance`, the ensemble-diversity channel) exactly.
    Clamped so an unsupervised or early prediction cannot produce NaN: a predicted row
    may sum above one, and the unpaired term is then floored.
    """
    n = logits.shape[-1]
    p = torch.sigmoid(logits.float()) * pairable_mask(n, logits.device)
    paired = p.sum(dim=-1)
    u = (1.0 - paired).clamp(min=1e-6, max=1.0)
    # p * log(max(p, 1e-12)), not xlogy(p, p): the derivative of xlogy at p = 0 is -inf,
    # and every unpairable entry is exactly 0, so xlogy turned the whole gradient into
    # NaN on the first step (seen in the smoke test). The clamp keeps it finite.
    plogp = p * torch.log(p.clamp(min=1e-12))
    ent = -(plogp.sum(dim=-1) + u * torch.log(u)) / math.log(2.0)
    div = (p * (1.0 - p)).sum(dim=-1)
    return torch.stack([paired, ent, div], dim=-1)


class AxialPairBlock(nn.Module):
    """One update of a pair representation: attention along rows, then columns, then an MLP."""

    def __init__(self, c: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.h = heads
        self.norm_r = nn.LayerNorm(c); self.qkv_r = nn.Linear(c, 3 * c); self.out_r = nn.Linear(c, c)
        self.norm_c = nn.LayerNorm(c); self.qkv_c = nn.Linear(c, 3 * c); self.out_c = nn.Linear(c, c)
        self.norm_f = nn.LayerNorm(c)
        self.ff = nn.Sequential(nn.Linear(c, 2 * c), nn.GELU(), nn.Linear(2 * c, c))
        self.drop = dropout

    def _rows(self, z, norm, qkv, out):
        b, n, m, c = z.shape
        d = c // self.h
        q, k, v = qkv(norm(z)).reshape(b * n, m, 3, self.h, d).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        return out(o.transpose(1, 2).reshape(b, n, m, c))

    def forward(self, z: torch.Tensor) -> torch.Tensor:          # (B, N, N, C)
        z = z + self._rows(z, self.norm_r, self.qkv_r, self.out_r)
        zt = z.transpose(1, 2)
        z = z + self._rows(zt, self.norm_c, self.qkv_c, self.out_c).transpose(1, 2)
        z = z + self.ff(self.norm_f(z))
        return 0.5 * (z + z.transpose(1, 2))                         # pairing is symmetric


class PairStack(nn.Module):
    """A pair representation that reasons over the base-pair map, for Family L.

    Folding is combinatorial pairing: whether base i is unpaired depends on whether any
    distant segment can pair with it, consistently with every other pair. The trunk with a
    static pair bias learned Family L only partially and overfitted it. This builds a pair
    representation from the sequence features of both positions, their base identities and
    their separation, updates it with attention along the rows and columns of the pair map
    (as RNAformer does for RNA folding) under a symmetry constraint, and returns
      (a) one additive bias per attention head per sequence-attention layer, and
      (b) a per-position summary (mean and max over partners) added to the per-bp
          features, which is where unpaired probability and positional entropy live.
    Both outputs are zero-initialised, so at initialisation the model is exactly the design
    without it. Each pass builds its own pair map from its own input, so the reverse-
    complement merge enforces every channel's parity unchanged. Gradient checkpointing
    keeps the (batch x 61 x 61 x channels) activations to one block at a time.
    """

    def __init__(self, feat: int, n_bias: int, channels: int = 64, n_layers: int = 4,
                 heads: int = 4, max_dist: int = 64, dropout: float = 0.0):
        super().__init__()
        self.max_dist = max_dist
        self.left = nn.Linear(feat, channels)
        self.right = nn.Linear(feat, channels)
        self.pair_embed = nn.Embedding(25, channels)
        self.dist_embed = nn.Embedding(2 * max_dist + 1, channels)
        self.blocks = nn.ModuleList(AxialPairBlock(channels, heads, dropout) for _ in range(n_layers))
        self.norm = nn.LayerNorm(channels)
        self.bias_out = nn.Linear(channels, n_bias)
        self.summary = nn.Linear(2 * channels, feat)
        # Base-pair probability logits of this pass's own (plus-strand) transcript, and
        # the two exact per-base functionals Family L's per-bp channels are made of:
        # paired probability sum_j p_ij, positional entropy in bits,
        # -sum_j p_ij log2 p_ij - u_i log2 u_i with u_i = 1 - sum_j p_ij, and the
        # ensemble-diversity contribution sum_j p_ij (1 - p_ij).
        self.bpp_out = nn.Linear(channels, 1)
        self.bpp_features = nn.Linear(3, feat)
        for lin in (self.bias_out, self.summary, self.bpp_features):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        # Start the pair logits at the measured prior: the mean probability of a pairable
        # entry is 0.0087 (logit -4.7; 16 shards of bpp_v2). From logit 0 the first
        # base-pair gradient was 158x the landscape loss's and fell to about 1x within
        # 50 steps (src/model/probe_bpp_weight.py); starting at the prior skips that
        # transient, during which global clipping starves every other family.
        nn.init.constant_(self.bpp_out.bias, BPP_PRIOR_LOGIT)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, n = h.shape
        hb = h.transpose(1, 2)
        known = x.sum(dim=1) > 0
        idx = torch.where(known, x.argmax(dim=1), torch.full_like(known, 4, dtype=torch.long))
        pair = idx[:, :, None] * 5 + idx[:, None, :]
        pos = torch.arange(n, device=h.device)
        rel = (pos[None, :] - pos[:, None]).clamp(-self.max_dist, self.max_dist) + self.max_dist
        z = (self.left(hb)[:, :, None, :] + self.right(hb)[:, None, :, :]
             + self.pair_embed(pair) + self.dist_embed(rel)[None])
        z = 0.5 * (z + z.transpose(1, 2))
        for blk in self.blocks:
            if self.training and torch.is_grad_enabled():
                z = torch.utils.checkpoint.checkpoint(blk, z, use_reentrant=False)
            else:
                z = blk(z)
        z = self.norm(z)
        bias = self.bias_out(z).permute(0, 3, 1, 2)                     # (B, n_bias, N, N)
        summ = self.summary(torch.cat([z.mean(dim=2), z.amax(dim=2)], dim=-1))
        logits = self.bpp_out(z).squeeze(-1).float()                    # (B, N, N)
        logits = 0.5 * (logits + logits.transpose(1, 2))
        summ = summ + self.bpp_features(bpp_functionals(logits)).to(summ.dtype)
        self.last_bpp_logits = logits
        return bias, summ.transpose(1, 2)                               # (B, F, N)


class GlobalReadout(nn.Module):
    """Whole-sequence outputs from mean, max and attention pools, through an MLP.

    The first readout was a low-rank *linear* map of the mean-pooled features. That can
    represent anything additive along the sequence -- a total enthalpy, a mean
    composition -- and nothing that is an extreme or a location. After five epochs the
    worst global channels, once the constant ones were removed, were exactly those: the
    bubble position (R-squared 0.002), the best G-quadruplex tetrad count (0.003), the
    Z-tract centre offset (0.003), the smallest hairpin (0.013), the best G4 loop. A
    maximum needs a max pool or a nonlinearity after pooling; a location needs pooling
    weights that depend on content and values that carry position.

    So: mean, max and `n_queries` learned attention pools, where the attention values
    carry a centred position coordinate u in [-1, 1] and u squared, concatenated and
    passed through a two-layer MLP before the low-rank head.

    **Equivariance is unaffected.** All three pools are invariant to permuting positions
    except through u, and u maps to -u under reverse complement exactly as a centre
    offset does, so the forward/reverse merge that enforces each global channel's
    declared parity applies unchanged: an antisymmetric location channel comes out
    antisymmetric by construction, as for every other head.
    """

    def __init__(self, n_feat: int, n_channels: int, rank: int = 64,
                 n_queries: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n_queries = n_queries
        self.score = nn.Conv1d(n_feat, n_queries, 1)
        self.value = nn.Conv1d(n_feat + 2, n_feat, 1)
        d_in = n_feat * (2 + n_queries)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, 2 * n_feat), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * n_feat, n_feat), nn.GELU())
        self.head = LowRankHead(n_feat, n_channels, rank)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, n = h.shape
        u = torch.linspace(-1.0, 1.0, n, device=h.device, dtype=h.dtype)
        u = u.view(1, 1, n).expand(b, 1, n)
        v = self.value(torch.cat([h, u, u * u], dim=1))                  # (B, F, N)
        a = torch.softmax(self.score(h).float(), dim=-1).to(v.dtype)     # (B, Q, N)
        att = torch.einsum("bqn,bfn->bqf", a, v).reshape(b, -1)
        z = torch.cat([h.mean(dim=-1), h.amax(dim=-1), att], dim=1)
        z = self.mlp(z).unsqueeze(-1)                                     # (B, F, 1)
        return self.head(z)


def window_index_offset(k: int) -> int:
    """Extra index shift a centred window of width k adds under reflection.

    The pooled window at position i covers [i-a, i+b] with a = (k-1)//2 and b = k-1-a.
    Reflecting maps it onto the window at n-1-i+s+(a-b), so an even width adds -1 and an
    odd width adds nothing. The plan's ladder rungs (10, 20, 50) are all even.
    """
    a = (k - 1) // 2
    return a - (k - 1 - a)


# ======================================================================================
# The model
# ======================================================================================

class PANGEA(nn.Module):
    """The encoder. Sequence, conditions and molecule class in; the landscape out.

    **The reverse-complement merge, and the two bugs it had.** `forward` runs the trunk on
    the sequence and on its reverse complement with shared weights, reflects the second
    set of outputs back into the forward frame, and combines them channel by channel by
    declared parity. That makes equivariance exact for any weights.

    The reflection must use each channel's *own* index map. The first version flipped
    every per-base-pair channel with i -> n-1-i, which is right for a base-pair quantity
    and wrong for a step quantity (i -> n-2-i) or a groove quantity (i -> n-1-i+-4). It
    averaged every step quantity with its neighbour's, and `C_twist` trained to R-squared
    0.09. Reflection is now by the measured `rc_shift`, and window outputs add the offset
    an even-width centred window introduces (`window_index_offset`).

    The log-variance was also merged by the *mean's* rule, so for an antisymmetric
    channel it became (fv - rv)/2 and was pinned near zero whatever the model predicted.
    The variance of an antisymmetric quantity is symmetric; only its mean changes sign.
    Log-variances are now always averaged.

    Positions whose reflected index falls off the end keep the forward prediction.
    STRANDED and COMPOSITE channels keep the forward prediction everywhere.

    **Optional attention.** `n_attn > 0` adds pre-norm transformer layers after the
    dilated convolutional trunk. The plan specifies the convolutional trunk; attention is
    a measured variant rather than a replacement, motivated by the RNA pairing channels,
    whose partners can sit anywhere in the fragment -- a pattern convolutions represent
    poorly at any depth.
    """

    def __init__(self, plan: ChannelPlan, width: int = 192, n_blocks: int = 12,
                 rank: int = 64, n_cond: int = 8, window_ladder: Sequence[int] = (10, 20, 50),
                 max_dilation: int = 32, dropout: float = 0.0,
                 use_wide_branch: bool = False, wide_width: int = 96,
                 n_attn: int = 0, attn_heads: int = 8,
                 global_readout: str = "linear", global_queries: int = 4,
                 pair_bias: bool = False, pair_channels: int = 32,
                 pair_layers: int = 4, window_readout: str = "head",
                 pair_module: str = "conv", pair_heads: int = 4,
                 strand_tie: bool = False, pair_bpp: bool = False,
                 wc_tie_global: Sequence[str] = (),
                 pb_window_rungs: Optional[Dict[int, Sequence[int]]] = None):
        super().__init__()
        self.plan = plan
        self.window_ladder = tuple(window_ladder)
        self.trunk = LocalTrunk(width, n_blocks, n_cond, max_dilation, dropout=dropout)
        self.use_wide_branch = use_wide_branch
        self.wide = WideBranch(wide_width, n_cond=n_cond) if use_wide_branch else None
        feat = width + (wide_width if use_wide_branch else 0)
        self.attn = nn.ModuleList()
        if n_attn > 0:
            if feat % attn_heads:
                raise ValueError(f"feature width {feat} not divisible by {attn_heads} heads")
            self.attn = nn.ModuleList(
                nn.TransformerEncoderLayer(feat, attn_heads, dim_feedforward=4 * feat,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
                for _ in range(n_attn))
            self.attn_norm = nn.LayerNorm(feat)
        self.attn_heads = attn_heads
        self.pair = None
        self.pair_stack = None
        if pair_bias:
            if n_attn == 0:
                raise ValueError("pair_bias needs attention layers to bias")
            if pair_module == "conv":
                self.pair = PairBias(n_attn * attn_heads, pair_channels, pair_layers,
                                     bpp_head=pair_bpp)
                if pair_bpp:
                    # the base-pair functionals enter the sequence features before the
                    # attention layers, zero-initialised as everywhere else
                    self.pair_feat = nn.Linear(3, feat)
                    nn.init.zeros_(self.pair_feat.weight); nn.init.zeros_(self.pair_feat.bias)
            elif pair_module == "axial":
                self.pair_stack = PairStack(feat, n_attn * attn_heads, pair_channels,
                                            pair_layers, pair_heads, dropout=dropout)
            else:
                raise ValueError(f"unknown pair module {pair_module!r}")

        n_pb, n_gl = len(plan.per_bp_names), len(plan.global_names)
        self.per_bp_head = LowRankHead(feat, n_pb, rank)
        if global_readout == "linear":
            self.global_head = LowRankHead(feat, n_gl, rank)
        elif global_readout == "attn_mlp":
            self.global_head = GlobalReadout(feat, n_gl, rank, n_queries=global_queries,
                                             dropout=dropout)
        else:
            raise ValueError(f"unknown global readout {global_readout!r}")
        self.global_readout = global_readout
        if window_readout not in ("head", "pooled"):
            raise ValueError(f"unknown window readout {window_readout!r}")
        self.window_readout = window_readout
        self.window_heads = nn.ModuleDict(
            {str(w): LowRankHead(feat, n_pb, rank) for w in self.window_ladder})
        # Extent-defined window tracks (Family W: bending over a window, the nucleosome window,
        # cyclizability and loop closure): PLAN.md, "window channels come off pooled
        # representations at each scale, drawing on the wide branch for the larger windows".
        # They are per-position tracks in the per-bp layout, but their values come from a head
        # on the features averaged over that window ({window bp: per-bp column indices}).
        self.pb_window_rungs = {int(w): list(map(int, idx)) for w, idx in (pb_window_rungs or {}).items()}
        self.pb_window_heads = nn.ModuleDict(
            {str(w): LowRankHead(feat, len(idx), rank) for w, idx in self.pb_window_rungs.items()})
        for w, idx in self.pb_window_rungs.items():
            self.register_buffer(f"_pbw_idx_{w}", torch.tensor(idx, dtype=torch.long), persistent=False)
        self.n_cond = n_cond

        # Parity masks and index-map groups, built once rather than every forward pass.
        def mask(par, tag):
            return torch.tensor([p == tag for p in par], dtype=torch.bool)
        self.register_buffer("pb_sym", mask(plan.per_bp_parity, "symmetric"), False)
        self.register_buffer("pb_anti", mask(plan.per_bp_parity, "antisymmetric"), False)
        self.register_buffer("gl_sym", mask(plan.global_parity, "symmetric"), False)
        self.register_buffer("gl_anti", mask(plan.global_parity, "antisymmetric"), False)
        shifts = torch.tensor(plan.per_bp_rc_shift, dtype=torch.long)
        self.register_buffer("pb_shift", shifts, False)
        # Strand tie (Family L). The minus-strand transcript of x is the plus-strand
        # transcript of rc(x), and its targets are stored in its own 5'->3' frame -- the
        # frame of the reverse-complement pass (verified against the stored profiles). So
        # L_m(x) = L_p(rc(x)) exactly, position by position, with no reflection: every
        # L_m output is read from the reverse pass's L_p output, and both strands' targets
        # train the one plus-transcript predictor. Exactly equivariant by construction.
        self.strand_tie = strand_tie
        empty = torch.zeros(0, dtype=torch.long)
        def ties(names):
            # Strand pairs whose minus member is exactly the plus member of the reverse
            # complement, stored in its own frame: Family L's transcripts, and Family K's
            # two hybrid orientations (DNA strand x or rc(x); duplex_classes).
            pos = {n: i for i, n in enumerate(names)}
            pairs = [(pos[n], pos[m + n[len(p):]]) for p, m in STRAND_PAIRS for n in names
                     if n.startswith(p) and m + n[len(p):] in pos]
            return (torch.tensor([a for a, _ in pairs], dtype=torch.long),
                    torch.tensor([b for _, b in pairs], dtype=torch.long))
        lp, lm = ties(plan.per_bp_names) if strand_tie else (empty, empty)
        gp, gm = ties(plan.global_names) if strand_tie else (empty, empty)
        # Watson/Crick tie for whole-sequence channels, listed by stem (X for the pair
        # X_W / X_C, including circular __sin/__cos columns): the Crick-strand value of x
        # is read from the reverse pass's Watson-strand value. Unlike L_m = L_p(rc), this
        # identity is a property of the structure pipeline, not of the definitions, so
        # only stems measured to satisfy it (src/eval/rc_consistency.py) are listed.
        pos = {n: i for i, n in enumerate(plan.global_names)}
        wc = [(pos[f"{stem}_W{suf}"], pos[f"{stem}_C{suf}"])
              for stem in wc_tie_global for suf in ("", "__sin", "__cos")
              if f"{stem}_W{suf}" in pos and f"{stem}_C{suf}" in pos]
        if wc_tie_global and not wc:
            raise ValueError(f"no W/C pair found for {list(wc_tie_global)}")
        self.wc_tie_global = list(wc_tie_global)
        if wc:
            gp = torch.cat([gp, torch.tensor([a for a, _ in wc], dtype=torch.long)])
            gm = torch.cat([gm, torch.tensor([b for _, b in wc], dtype=torch.long)])
        self.apply_ties = bool(len(lp) or len(gp))
        self.register_buffer("pb_tie_p", lp, False); self.register_buffer("pb_tie_m", lm, False)
        self.register_buffer("gl_tie_p", gp, False); self.register_buffer("gl_tie_m", gm, False)
        self._shift_values = sorted(set(int(v) for v in plan.per_bp_rc_shift))

    # ----------------------------------------------------------------------------------
    # Attention and the pair modules are quadratic in length: on inputs longer than ATTN_BLOCK they
    # run on overlapping blocks of ATTN_BLOCK (stride half a block) blended by triangular weights.
    # Inputs up to ATTN_BLOCK (every 61 bp fragment) are processed exactly as before; Family W's
    # 10,240 bp fragments and genome-scale sliding use the blocks. Equivariance is unaffected: it
    # comes from merging the forward and reverse-complement passes by parity.
    ATTN_BLOCK = 128

    def _features(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.trunk(x, cond)
        if self.wide is not None:
            h = torch.cat([h, self.wide(x, cond, h.shape[-1])], dim=1)
        if self.pair_stack is None and not len(self.attn):
            return h
        L, blk = h.shape[-1], self.ATTN_BLOCK
        if L <= blk:
            return self._attend(h, x)
        st = blk // 2
        starts = list(range(0, L - blk + 1, st))
        if starts[-1] != L - blk:
            starts.append(L - blk)
        B = h.shape[0]
        hb = torch.cat([h[..., s0:s0 + blk] for s0 in starts], 0)
        xb = torch.cat([x[..., s0:s0 + blk] for s0 in starts], 0)
        ob = self._attend(hb, xb)
        t = torch.arange(blk, device=h.device, dtype=torch.float32)
        wt = (1.0 - (t - (blk - 1) / 2).abs() / (blk / 2)).clamp(min=1e-3)
        out = torch.zeros(B, ob.shape[1], L, device=h.device, dtype=torch.float32)
        den = torch.zeros(L, device=h.device, dtype=torch.float32)
        for k, s0 in enumerate(starts):
            out[..., s0:s0 + blk] += ob[k * B:(k + 1) * B].float() * wt
            den[s0:s0 + blk] += wt
        return (out / den).to(h.dtype)

    def _attend(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        bias = None
        if self.pair_stack is not None:
            bias, summ = self.pair_stack(h, x)
            h = h + summ
        if len(self.attn):
            z = h.transpose(1, 2)
            if self.pair is not None:
                bias = self.pair(x)
                if self.pair.bpp_out is not None:
                    z = z + self.pair_feat(bpp_functionals(self.pair.last_bpp_logits)).to(z.dtype)
            for li, layer in enumerate(self.attn):
                if bias is None:
                    z = layer(z)
                else:
                    hh = self.attn_heads
                    z = _biased_encoder_layer(layer, z, bias[:, li * hh:(li + 1) * hh])
            h = self.attn_norm(z).transpose(1, 2)
        return h

    def _heads(self, h: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        out["per_bp"] = self.per_bp_head(h)
        if self.pb_window_rungs:
            pm, pv = out["per_bp"]
            pm, pv = pm.clone(), pv.clone()
            c = F.pad(h.float().cumsum(-1), (1, 0))          # box means in O(L) at any width
            for w in self.pb_window_rungs:
                k = min(int(w), h.shape[-1])
                hw = (c[..., k:] - c[..., :-k]) / k
                hw = F.pad(hw, ((k - 1) // 2, k - 1 - (k - 1) // 2), mode="replicate").to(h.dtype)
                wm, wv = self.pb_window_heads[str(w)](hw)
                idx = getattr(self, f"_pbw_idx_{w}")
                pm = pm.index_copy(1, idx, wm.to(pm.dtype)); pv = pv.index_copy(1, idx, wv.to(pv.dtype))
            out["per_bp"] = (pm, pv)
        if self.global_readout == "linear":
            gm, gv = self.global_head(h.mean(dim=-1, keepdim=True))
        else:
            gm, gv = self.global_head(h)
        out["global"] = (gm.squeeze(-1), gv.squeeze(-1))
        for w in self.window_ladder:
            k = min(int(w), h.shape[-1])
            hw = F.avg_pool1d(h, kernel_size=k, stride=1, padding=0)
            hw = F.pad(hw, ((k - 1) // 2, k - 1 - (k - 1) // 2), mode="replicate")
            out[f"window_{w}"] = self.window_heads[str(w)](hw)
        return out

    def reflect(self, rev: torch.Tensor, extra_shift: int = 0
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map reverse-pass positional outputs into the forward frame, channel by channel.

        Returns (reflected, valid): reflected[:, c, i] = rev[:, c, n-1-i+s_c] with
        s_c = rc_shift[c] + extra_shift, and valid[c, i] says whether that index exists.
        """
        n = rev.shape[-1]
        flipped = rev.flip(-1)                      # flipped[..., i] = rev[..., n-1-i]
        out = flipped
        valid = torch.ones(rev.shape[1], n, dtype=torch.bool, device=rev.device)
        for s0 in self._shift_values:
            s = s0 + extra_shift
            if s == 0:
                continue
            idx = (self.pb_shift == s0).nonzero(as_tuple=True)[0]
            rolled = torch.roll(flipped.index_select(1, idx), shifts=s, dims=-1)
            out = out.index_copy(1, idx, rolled)
            v = torch.ones(idx.numel(), n, dtype=torch.bool, device=rev.device)
            if s > 0:
                v[:, :s] = False
            else:
                v[:, n + s:] = False
            valid = valid.index_copy(0, idx, v)
        return out, valid

    @staticmethod
    def _merge_mean(f, r, sym, anti, valid=None):
        """Average by declared parity; the sign flips for antisymmetric channels."""
        if valid is None:
            ms, ma = sym[None, :], anti[None, :]
        else:
            ms, ma = (sym[:, None] & valid)[None], (anti[:, None] & valid)[None]
        out = torch.where(ms, 0.5 * (f + r), f)
        return torch.where(ma, 0.5 * (f - r), out)

    @staticmethod
    def _merge_logvar(f, r, sym, anti, valid=None):
        """Average for both symmetric and antisymmetric channels: a variance has no sign."""
        both = sym | anti
        m = both[None, :] if valid is None else (both[:, None] & valid)[None]
        return torch.where(m, 0.5 * (f + r), f)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None,
                equivariant: bool = True, return_bpp: bool = False
                ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Merged outputs; with `return_bpp`, also "bpp_logits": the pair stack's
        base-pair logits of each pass's own transcript, (2B, N, N) -- rows [:B] the
        forward pass (plus-strand transcript, forward frame), rows [B:] the reverse pass
        (minus-strand transcript, its own frame) -- or (B, N, N) without equivariance."""
        out = self._forward(x, cond, equivariant)
        if return_bpp:
            if self.pair_stack is not None:
                out["bpp_logits"] = self.pair_stack.last_bpp_logits
            elif self.pair is not None and self.pair.bpp_out is not None:
                out["bpp_logits"] = self.pair.last_bpp_logits
            else:
                raise ValueError("return_bpp needs the axial pair module or pair_bpp")
        return out

    def _forward(self, x: torch.Tensor, cond: Optional[torch.Tensor],
                 equivariant: bool) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        if not equivariant:
            return self._heads(self._features(x, cond))
        # One batched trunk pass over [x; rc(x)] instead of two, which halves the kernel
        # launches for a model this small.
        # The condition vector describes the molecule's environment, which is the same
        # for a sequence and its reverse complement, so it is repeated to match the
        # doubled batch rather than left at the original size.
        cond2 = None if cond is None else torch.cat([cond, cond], 0)
        both = self._heads(self._features(torch.cat([x, reverse_complement(x)], 0), cond2))
        B = x.shape[0]
        merged: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for key, (m, v) in both.items():
            fm, rm, fv, rv = m[:B], m[B:], v[:B], v[B:]
            if self.apply_ties:
                tp, tm = ((self.gl_tie_p, self.gl_tie_m) if key == "global"
                          else (self.pb_tie_p, self.pb_tie_m))
                if tp.numel():
                    fm, fv = fm.clone(), fv.clone()
                    fm[:, tm] = rm[:, tp]
                    fv[:, tm] = rv[:, tp]
            if key == "global":
                merged[key] = (self._merge_mean(fm, rm, self.gl_sym, self.gl_anti),
                               self._merge_logvar(fv, rv, self.gl_sym, self.gl_anti))
                continue
            extra = 0 if key == "per_bp" else window_index_offset(
                min(int(key.split("_")[1]), fm.shape[-1]))
            rm_r, valid = self.reflect(rm, extra)
            rv_r, _ = self.reflect(rv, extra)
            merged[key] = (self._merge_mean(fm, rm_r, self.pb_sym, self.pb_anti, valid),
                           self._merge_logvar(fv, rv_r, self.pb_sym, self.pb_anti, valid))
        if self.window_readout == "pooled":
            # Window means as the window average of the final per-bp means -- what the
            # window targets are by definition. Measured on validation at every ladder
            # rung, this beats the dedicated window heads (median R-squared at 200M:
            # 0.9667 -> 0.9893 at 10 bp, 0.9486 -> 0.9957 at 50 bp); the test side agrees.
            # Taken after the merge, so it is consistent with the per-bp output exactly.
            # The head's log-variance is kept; it was fitted around the head's mean, so
            # for the pooled mean it is conservative.
            from data import pooled_window_targets
            pb = merged["per_bp"][0]
            for w in self.window_ladder:
                key = f"window_{w}"
                if key in merged:
                    merged[key] = (pooled_window_targets(pb.float(), min(int(w), pb.shape[-1])
                                                         ).to(pb.dtype), merged[key][1])
        return merged

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ======================================================================================
# Loss
# ======================================================================================

class LandscapeLoss(nn.Module):
    """Heteroscedastic beta-NLL per channel, balanced across families, fully vectorised.

    **What the plan asks for, and why the first implementation of it was ill-posed.**
    `PLAN.md`: "Outputs are heteroscedastic, with loss weights across families learned
    through uncertainty weighting rather than set by hand." A per-channel learned
    variance in a Gaussian likelihood *is* uncertainty weighting -- it is exactly Kendall
    and Gal's task-uncertainty term, one task per channel -- so the per-channel log-variance
    head satisfies the requirement on its own.

    The first version stacked a second, per-family learned weight on top, in the form
    exp(-s) * L + s. That form assumes L is non-negative. A Gaussian NLL is not: it went
    to -31 within five epochs. With L < 0 the gradient with respect to s is always
    positive, so every family weight drifts toward the clamp together. Measured on the
    checkpoint: all six family weights sat between 4.07 and 4.78, carrying no per-family
    information, and still falling. It is removed. Families are instead balanced
    explicitly, each contributing its mean channel loss with equal weight, so Family C's
    167 per-base-pair channels do not drown Family F's three.

    **beta-NLL.** Seitzer et al., "On the pitfalls of heteroscedastic uncertainty
    estimation with probabilistic neural networks", ICLR 2022: under a plain Gaussian NLL
    the gradient on the mean is scaled by 1/sigma^2, so a network can lower the loss on a
    hard target by inflating its variance and then stop fitting its mean. Weighting each
    element's NLL by a stop-gradient sigma^(2 beta) removes that incentive; beta = 0.5 is
    their recommended default and beta = 0 recovers the plain NLL. That failure mode is
    exactly what low R-squared on the hardest families looks like from outside.

    **NaN targets are masked, never imputed**, and a channel with no finite target in a
    batch contributes nothing -- not a zero -- so a family with nothing to fit is absent
    from the loss rather than rewarded for it.
    """

    def __init__(self, per_bp_family: Sequence[str], global_family: Sequence[str],
                 beta: float = 0.5, window_weight: float = 1.0,
                 beta_per_bp: Optional[Sequence[float]] = None,
                 beta_global: Optional[Sequence[float]] = None):
        super().__init__()
        fams = sorted(set(per_bp_family) | set(global_family))
        self.families = fams
        self.beta = float(beta)
        # Optional per-channel beta (e.g. 1 for rare-event channels: the mean gradient then no
        # longer shrinks with the predicted spread, so a rare large target cannot be "fitted"
        # by inflating the variance). Window outputs pool per-bp channels and use beta_per_bp.
        self.per_channel_beta = beta_per_bp is not None or beta_global is not None
        if self.per_channel_beta:
            bp = torch.full((len(per_bp_family),), float(beta)) if beta_per_bp is None \
                else torch.as_tensor(list(beta_per_bp), dtype=torch.float32)
            bg = torch.full((len(global_family),), float(beta)) if beta_global is None \
                else torch.as_tensor(list(beta_global), dtype=torch.float32)
            self.register_buffer("beta_pb", bp, False)
            self.register_buffer("beta_gl", bg, False)
        self.window_weight = float(window_weight)
        idx = {f: i for i, f in enumerate(fams)}

        def onehot(fs):
            m = torch.zeros(len(fs), len(fams))
            for c, f in enumerate(fs):
                m[c, idx[f]] = 1.0
            return m
        self.register_buffer("M_pb", onehot(per_bp_family), False)
        self.register_buffer("M_gl", onehot(global_family), False)

    def channel_nll(self, mean: torch.Tensor, logvar: torch.Tensor,
                    target: torch.Tensor, beta: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-channel summed beta-NLL and per-channel count of finite targets."""
        mean, logvar, target = mean.float(), logvar.float(), target.float()
        valid = torch.isfinite(target)
        t = torch.where(valid, target, mean.detach())
        var = torch.exp(logvar)
        nll = 0.5 * (logvar + (t - mean) ** 2 / var)
        if beta is not None:
            shape = (1, -1, 1) if nll.dim() == 3 else (1, -1)
            nll = nll * var.detach() ** beta.view(shape)
        elif self.beta > 0:
            nll = nll * var.detach() ** self.beta
        nll = torch.where(valid, nll, torch.zeros_like(nll))
        dims = (0, 2) if nll.dim() == 3 else (0,)
        return nll.sum(dims), valid.sum(dims).to(nll.dtype)

    def family_means(self, s: torch.Tensor, n: torch.Tensor, M: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        has = (n > 0).to(s.dtype)
        ch = s / n.clamp(min=1.0) * has
        fam_sum, fam_cnt = ch @ M, has @ M
        return fam_sum / fam_cnt.clamp(min=1.0), fam_cnt > 0

    def forward(self, preds: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                targets: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total = None
        parts: Dict[str, torch.Tensor] = {}
        n_windows = max(1, sum(1 for k in preds if k.startswith("window_")))
        for key, (mean, logvar) in preds.items():
            tgt = targets.get(key)
            if tgt is None:
                continue
            b = None
            if self.per_channel_beta:
                b = self.beta_gl if key == "global" else self.beta_pb
                if b.numel() != mean.shape[1]:
                    b = None                    # a block whose columns differ (long windows)
            s, n = self.channel_nll(mean, logvar, tgt, b)
            M = self.M_gl if key == "global" else self.M_pb
            fam, present = self.family_means(s, n, M)
            if not bool(present.any()):
                continue
            term = fam[present].mean()
            w = self.window_weight / n_windows if key.startswith("window_") else 1.0
            total = w * term if total is None else total + w * term
            parts[key] = term.detach()
        if total is None:
            total = torch.zeros((), device=next(iter(preds.values()))[0].device)
        return total, parts

