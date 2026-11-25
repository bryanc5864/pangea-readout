#!/usr/bin/env python3
# MIT License, Bryan Cheng, 2026. Part of PANGEA/READOUT.
"""PANGEA's architectural guarantees, verified rather than asserted.

The central claim in `PLAN.md` about this model is that reverse-complement equivariance
is enforced *architecturally*: "A channel predicted with the wrong parity is wrong in a
way no validation correlation will reveal, so parity is checked channel by channel as a
unit test before anything downstream runs."

That claim is only worth anything if it holds for an untrained network with random
weights, because that is what makes it a property of the architecture rather than
something the model might learn. Every equivariance test here therefore runs on a freshly
initialised model, and each comes with a negative control showing the same model fails
the test with the equivariance path switched off.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "model"))
sys.path.insert(0, os.path.join(ROOT, "src", "physics"))

torch = pytest.importorskip("torch", reason="training environment not active")

import pangea as P  # noqa: E402

SEQS = ["ACGTACGTAGCTAGCTAGCTTACGATCGATCGGCTAGCTAGCATCGATCGATCGATTACGA"[:61],
        "GCGCAAATTTGCGCATATAGCGCAAATTTGCGCCCCTCCCTCCCTCCCAAATTTGCGCATA"]


# The channel plan for these tests is built from explicit metadata rather than from the
# registry, for the same reason the training job reads its layout from the dataset: the
# training environment has no physics stack installed, so a test that imported it could
# only ever run in the other environment. The architectural properties under test depend
# on having a representative mix of parities, not on the real channel names.
#
# `test_registry_plan_covers_all_four_parities` checks the real registry separately, and
# skips where the physics is unavailable.
_PARITIES = ["symmetric", "antisymmetric", "stranded", "composite"]


# Shifts cycle through the four index maps the real registry uses: base pair (0), step
# (-1), minor groove (+4), major groove (-4). The first synthetic plan had every shift
# at zero, which is why its equivariance tests passed while the real model was averaging
# step channels with their neighbours.
_SHIFTS = [0, -1, 4, -4]


def _synthetic_plan(n_per_bp: int = 24, n_global: int = 16) -> "P.ChannelPlan":
    pb_par = [_PARITIES[i % 4] for i in range(n_per_bp)]
    gl_par = [_PARITIES[i % 4] for i in range(n_global)]
    return P.ChannelPlan.from_metadata(
        [f"X_pb_{i}" for i in range(n_per_bp)],
        [f"X_gl_{i}" for i in range(n_global)],
        pb_par, gl_par,
        [["B", "C", "D", "E"][i % 4] for i in range(n_per_bp)],
        [["B", "C", "D", "E"][i % 4] for i in range(n_global)],
        [_SHIFTS[(i // 4) % 4] for i in range(n_per_bp)])


@pytest.fixture(scope="module")
def plan():
    return _synthetic_plan()


def test_registry_plan_covers_all_four_parities():
    """The real registry must exercise every parity class the model handles.

    If it did not, the equivariance merge would have an untested branch in production
    even though the synthetic tests above pass.
    """
    try:
        real = P.ChannelPlan.from_registry()
    except Exception as exc:                                       # noqa: BLE001
        pytest.skip(f"physics stack unavailable in this environment: {exc}")
    for res in (real.per_bp_parity, real.global_parity):
        assert set(res) <= set(_PARITIES)
    both = set(real.per_bp_parity) | set(real.global_parity)
    assert both == set(_PARITIES), f"registry only uses {sorted(both)}"


@pytest.fixture(scope="module")
def model(plan):
    torch.manual_seed(0)
    m = P.PANGEA(plan, width=64, n_blocks=6, rank=16, window_ladder=(10, 20),
                 max_dilation=8)
    m.eval()
    return m


@pytest.fixture(scope="module")
def x():
    return P.one_hot(SEQS)


# ======================================================================================
# The one-hot and reverse-complement primitives
# ======================================================================================

def test_reverse_complement_of_one_hot_matches_the_string_operation():
    """The tensor operation and the string operation must agree exactly.

    Doing it on the tensor is what keeps the second pass inside the autograd graph, but
    only if it is the same operation.
    """
    import channels as ch
    a = P.reverse_complement(P.one_hot(SEQS))
    b = P.one_hot([ch.reverse_complement(s) for s in SEQS])
    assert torch.equal(a, b)


def test_reverse_complement_is_an_involution(x):
    assert torch.equal(P.reverse_complement(P.reverse_complement(x)), x)


def test_one_hot_leaves_unknown_bases_empty():
    v = P.one_hot(["ACNGT"])
    assert v[0, :, 2].sum().item() == 0.0
    assert v.sum().item() == 4.0


# ======================================================================================
# Equivariance, on an untrained model
# ======================================================================================

def _rc_predictions(model, x, equivariant=True):
    """Predictions on a sequence and on its reverse complement, same weights."""
    with torch.no_grad():
        a = model(x, cond=None, equivariant=equivariant)
        b = model(P.reverse_complement(x), cond=None, equivariant=equivariant)
    return a, b


def _parity_residual(model, x, parity_tag, key, equivariant=True, which=0):
    """Largest violation of the declared parity across the channels carrying it.

    Positional outputs are reflected by each channel's own index map (plus the window
    offset), and only positions whose reflected index exists are compared.
    """
    a, b = _rc_predictions(model, x, equivariant)
    parities = (model.plan.global_parity if key == "global"
                else model.plan.per_bp_parity)
    idx = [i for i, p in enumerate(parities) if p == parity_tag]
    if not idx:
        return 0.0
    fm, rm = a[key][which], b[key][which]
    sign = -1.0 if parity_tag == "antisymmetric" and which == 0 else 1.0
    if key == "global":
        d = (fm[:, idx] - sign * rm[:, idx]).abs()
        return float((d.max() / fm[:, idx].abs().max().clamp(min=1e-6)).item())
    extra = 0 if key == "per_bp" else P.window_index_offset(int(key.split("_")[1]))
    rr, valid = model.reflect(rm, extra)
    m = valid[idx][None].expand(fm.shape[0], -1, -1)
    d = (fm[:, idx] - sign * rr[:, idx]).abs()[m]
    return float((d.max() / fm[:, idx].abs()[m].max().clamp(min=1e-6)).item())


@pytest.mark.parametrize("key", ["per_bp", "global"])
def test_symmetric_channels_are_exactly_equivariant_untrained(model, x, key):
    r = _parity_residual(model, x, "symmetric", key)
    assert r < 1e-5, f"{key} symmetric residual {r:.2e} on an untrained model"


@pytest.mark.parametrize("key", ["per_bp", "global"])
def test_antisymmetric_channels_are_exactly_equivariant_untrained(model, x, key):
    r = _parity_residual(model, x, "antisymmetric", key)
    assert r < 1e-5, f"{key} antisymmetric residual {r:.2e} on an untrained model"


@pytest.mark.parametrize("key", ["per_bp", "global"])
def test_without_the_equivariance_path_the_same_model_fails(model, x, key):
    """Negative control. Without this, the test above could pass for trivial reasons.

    A network whose outputs were near-constant, or whose parity masks were empty, would
    satisfy the equivariance check without the merge doing any work. Switching the merge
    off has to break it.
    """
    r = _parity_residual(model, x, "symmetric", key, equivariant=False)
    assert r > 1e-3, (
        f"{key}: disabling the equivariance merge did not break symmetry "
        f"(residual {r:.2e}); the test above may be passing vacuously")


def test_equivariance_holds_after_a_gradient_step(plan, x):
    """It is a property of the architecture, so training must not be able to break it."""
    torch.manual_seed(1)
    m = P.PANGEA(plan, width=64, n_blocks=4, rank=16, window_ladder=(10,),
                 max_dilation=4)
    out = m(x, cond=None)
    loss = sum(v[0].pow(2).mean() + v[1].pow(2).mean() for v in out.values())
    loss.backward()
    with torch.no_grad():
        for p in m.parameters():
            if p.grad is not None:
                p -= 0.1 * p.grad
    m.eval()
    assert _parity_residual(m, x, "symmetric", "per_bp") < 1e-5
    assert _parity_residual(m, x, "antisymmetric", "per_bp") < 1e-5


def test_stranded_channels_are_not_forced_symmetric(model, x):
    """STRANDED channels must genuinely differ between the strands.

    If the merge accidentally averaged them, a strand-specific quantity would be
    reported as a property of the duplex -- the same error the registry's STRANDED
    declaration exists to prevent.
    """
    r = _parity_residual(model, x, "stranded", "global")
    assert r > 1e-3, "STRANDED channels came out strand-symmetric"


# ======================================================================================
# Shapes, conditioning and the heads
# ======================================================================================

def test_output_shapes_match_the_channel_plan(model, x, plan):
    out = model(x, cond=None)
    n, L = x.shape[0], x.shape[-1]
    assert out["per_bp"][0].shape == (n, len(plan.per_bp_names), L)
    assert out["global"][0].shape == (n, len(plan.global_names))
    for w in model.window_ladder:
        assert out[f"window_{w}"][0].shape == (n, len(plan.per_bp_names), L)


def test_every_output_carries_a_variance(model, x):
    for key, (mean, logvar) in model(x, cond=None).items():
        assert mean.shape == logvar.shape, key
        assert torch.isfinite(logvar).all(), key


def test_film_with_zero_conditions_is_the_identity(plan, x):
    """A zero condition vector must leave the features untouched.

    The scale is parameterised as 1 + delta precisely so this holds, which keeps the
    no-condition case exact instead of approximately right.
    """
    torch.manual_seed(2)
    m = P.PANGEA(plan, width=64, n_blocks=4, rank=16, window_ladder=(10,),
                 max_dilation=4).eval()
    with torch.no_grad():
        a = m(x, cond=None)["per_bp"][0]
        b = m(x, cond=torch.zeros(x.shape[0], m.n_cond))["per_bp"][0]
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_conditions_actually_change_the_output(plan, x):
    """FiLM is only useful if a non-zero condition moves the prediction."""
    torch.manual_seed(3)
    m = P.PANGEA(plan, width=64, n_blocks=4, rank=16, window_ladder=(10,),
                 max_dilation=4).eval()
    # FiLM is initialised to zero so the untrained model ignores conditions by design;
    # perturb the projection to check the pathway is wired, not that it is pre-trained.
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, P.FiLM):
                mod.proj.weight.normal_(0, 0.1)
        a = m(x, cond=torch.zeros(x.shape[0], m.n_cond))["per_bp"][0]
        b = m(x, cond=torch.ones(x.shape[0], m.n_cond))["per_bp"][0]
    assert (a - b).abs().max() > 1e-4


def test_receptive_field_is_reported_and_reaches_the_planned_scale(plan):
    m = P.PANGEA(plan, width=32, n_blocks=12, rank=8, max_dilation=32)
    rf = m.trunk.receptive_field()
    assert rf >= 200, f"receptive field {rf} bp, PLAN.md asks for near 300"


def test_low_rank_head_is_actually_low_rank(plan):
    m = P.PANGEA(plan, width=64, n_blocks=4, rank=8, window_ladder=(10,))
    assert m.per_bp_head.bottleneck.out_channels == 8
    full = 64 * len(plan.per_bp_names)
    low = 64 * 8 + 8 * len(plan.per_bp_names)
    assert low < full


def test_wide_branch_is_off_by_default(plan):
    """It is implemented but must not run on 61 bp fragments without being asked for."""
    assert P.PANGEA(plan, width=32, n_blocks=2, rank=4).wide is None
    assert P.PANGEA(plan, width=32, n_blocks=2, rank=4,
                    use_wide_branch=True).wide is not None


# ======================================================================================
# Loss
# ======================================================================================

def test_nan_targets_are_masked_not_imputed():
    """A NaN target contributes nothing -- not a zero."""
    loss = P.LandscapeLoss(["C"] * 10, ["C"])
    mean, logvar = torch.zeros(4, 10), torch.zeros(4, 10)
    full = torch.full((4, 10), 2.0)
    part = full.clone()
    part[:, :5] = float("nan")
    s_full, n_full = loss.channel_nll(mean, logvar, full)
    s_part, n_part = loss.channel_nll(mean, logvar, part)
    assert torch.equal(n_part[:5], torch.zeros(5)) and torch.equal(n_part[5:], n_full[5:])
    torch.testing.assert_close(s_part[5:], s_full[5:])
    assert torch.equal(s_part[:5], torch.zeros(5))


def test_a_family_with_no_valid_targets_is_absent_from_the_loss():
    """Regression for the first loss: a family with nothing to fit must not appear."""
    loss = P.LandscapeLoss(["C", "E"], ["C", "E"])
    tgt = torch.zeros(8, 2)
    tgt[:, 0] = float("nan")
    preds = {"global": (torch.zeros(8, 2, requires_grad=True), torch.zeros(8, 2))}
    total, parts = loss(preds, {"global": tgt})
    total.backward()
    # Only family E contributes: its NLL at mean 0, target 0, logvar 0 is exactly 0.
    assert torch.isfinite(total) and float(total) == 0.0


def test_families_are_balanced_not_counted():
    """Three channels of one family must not outweigh one channel of another."""
    loss = P.LandscapeLoss(["C", "C", "C", "F"], ["C"], beta=0.0)
    mean = torch.zeros(4, 4, 5)
    logvar = torch.zeros(4, 4, 5)
    tgt = torch.zeros(4, 4, 5)
    tgt[:, 3] = 2.0                              # only the single F channel is wrong
    total, _ = loss({"per_bp": (mean, logvar)}, {"per_bp": tgt})
    # per-element NLL for the F channel is 0.5 * 4 = 2; C is 0; the family mean is 1.
    assert abs(float(total) - 1.0) < 1e-6


def test_beta_nll_keeps_fitting_the_mean_of_a_high_variance_point():
    """Seitzer et al. 2022: plain NLL scales the mean gradient by 1/sigma^2."""
    m = torch.zeros(1, 1, requires_grad=True)
    lv = torch.full((1, 1), 4.0)                 # sigma^2 = e^4: an "inflated" variance
    t = torch.ones(1, 1)
    grads = {}
    for beta in (0.0, 0.5):
        m.grad = None
        loss = P.LandscapeLoss(["C"], ["C"], beta=beta)
        s, _ = loss.channel_nll(m, lv, t)
        s.sum().backward()
        grads[beta] = float(m.grad.abs())
    assert grads[0.5] > 5 * grads[0.0], grads


def test_vectorised_loss_matches_a_reference_loop():
    torch.manual_seed(0)
    fam = ["B", "C", "C", "E", "E", "E"]
    loss = P.LandscapeLoss(fam, ["B"], beta=0.5)
    mean, lv, tgt = torch.randn(5, 6, 7), torch.randn(5, 6, 7) * 0.3, torch.randn(5, 6, 7)
    tgt[0, 1, :3] = float("nan")
    total, _ = loss({"per_bp": (mean, lv)}, {"per_bp": tgt})
    ref_fam = {}
    for c, f in enumerate(fam):
        v = torch.isfinite(tgt[:, c])
        var = lv[:, c].exp()
        e = 0.5 * (lv[:, c] + (tgt[:, c] - mean[:, c]) ** 2 / var) * var ** 0.5
        ref_fam.setdefault(f, []).append(e[v].mean())
    ref = torch.stack([torch.stack(v).mean() for v in ref_fam.values()]).mean()
    torch.testing.assert_close(total, ref, rtol=1e-5, atol=1e-6)


def test_logvar_of_an_antisymmetric_channel_is_merged_symmetrically(plan, x):
    """Regression: the first merge sign-flipped log-variances like means."""
    torch.manual_seed(5)
    m = P.PANGEA(plan, width=64, n_blocks=4, rank=16, window_ladder=(10,),
                 max_dilation=4).eval()
    r = _parity_residual(m, x, "antisymmetric", "per_bp", which=1)
    assert r < 1e-5, f"antisymmetric log-variance not symmetric: {r:.2e}"


def test_window_outputs_are_equivariant_under_their_offset(model, x):
    for w in model.window_ladder:
        for tag in ("symmetric", "antisymmetric"):
            r = _parity_residual(model, x, tag, f"window_{w}")
            assert r < 1e-5, f"window_{w} {tag}: {r:.2e}"


def test_window_offset_matches_the_target_pooling():
    """The merge's window offset must match how window targets are actually pooled."""
    import data as D
    torch.manual_seed(0)
    n = 31
    half = torch.randn(1, 1, 15)
    v = torch.cat([half, torch.randn(1, 1, 1), half.flip(-1)], -1)   # symmetric, shift 0
    for k in (10, 20, 9):
        pooled = D.pooled_window_targets(v, k)[0, 0]
        off = P.window_index_offset(k)
        idx = torch.arange(n).flip(0) + off
        ok = (idx >= 0) & (idx < n)
        assert torch.allclose(pooled[ok], pooled[idx[ok]], atol=1e-6), k


def test_attention_variant_is_equivariant_too(plan, x):
    torch.manual_seed(6)
    m = P.PANGEA(plan, width=64, n_blocks=3, rank=16, window_ladder=(10,),
                 max_dilation=4, n_attn=2, attn_heads=4).eval()
    assert _parity_residual(m, x, "symmetric", "per_bp") < 1e-5
    assert _parity_residual(m, x, "antisymmetric", "per_bp") < 1e-5


# ======================================================================================
# The split the QC gate's leniency depends on
# ======================================================================================

def test_similarity_split_puts_whole_groups_on_one_side():
    """A similarity group straddling the boundary defeats the point of grouping.

    The QC gate treats duplicate sequences as a warning rather than a failure, on the
    grounds that the similarity split assigns near-duplicates to one side. That claim is
    load-bearing -- it is what makes a leaky dataset safe to train on -- so it is tested
    here rather than asserted in the gate's message.

    This test lives with the model tests because it imports `train`, which imports torch.
    Its companion, that identical sequences land in the same group, lives in
    `tests/test_landscape.py` and needs only numpy.
    """
    sys.path.insert(0, os.path.join(ROOT, "src", "data"))
    import splits as sp
    import train as tr

    rng = np.random.default_rng(1)
    base = ["".join(rng.choice(list("ACGT"), 61)) for _ in range(60)]
    seqs = base + [base[5], base[5], base[22]]       # plant exact duplicates
    train_idx, val_idx, report = tr.similarity_split(seqs, 0.25, seed=0)

    assert report["n_train"] + report["n_val"] == len(seqs)
    assert report["n_val"] > 0 and report["n_train"] > 0
    labels, _ = sp.group_by_similarity(seqs, threshold=0.5, seed=0)
    train_groups = {labels[i] for i in train_idx}
    val_groups = {labels[i] for i in val_idx}
    assert not (train_groups & val_groups), (
        "a similarity group appears on both sides of the split, so the QC gate's "
        "warning-not-problem treatment of duplicate sequences is unsound")


def test_random_split_is_available_so_the_leak_can_be_measured():
    """`PLAN.md` wants the overlap quantified, which needs both splits to exist.

    Measured on this library: 3.53 percent of fragments sit in a near-duplicate group,
    almost all of them tiling neighbours of the same element. Comparing a random split
    against a similarity split is how that becomes a number rather than an argument.
    """
    import train as tr
    tr_idx, va_idx = tr.split_indices(1000, 0.2, seed=0)
    assert len(tr_idx) == 800 and len(va_idx) == 200
    assert not (set(tr_idx.tolist()) & set(va_idx.tolist()))


# ======================================================================================
# The nonlinear global readout
# ======================================================================================

def test_attn_mlp_readout_keeps_global_parity(plan, x):
    """Antisymmetric globals flip sign and symmetric ones do not, with the new readout.

    The readout feeds a centred position coordinate into its attention values, which is
    exactly the kind of input that could break equivariance if the merge did not handle
    it, so this is checked rather than argued -- and the same model with the merge
    switched off must fail, or the check would pass vacuously.
    """
    torch.manual_seed(0)
    m = P.PANGEA(plan, width=64, n_blocks=4, rank=16, window_ladder=(10,),
                 global_readout="attn_mlp").eval()
    for tag in ("symmetric", "antisymmetric"):
        assert _parity_residual(m, x, tag, "global") < 1e-5, tag
    assert _parity_residual(m, x, "antisymmetric", "global", equivariant=False) > 1e-3


def test_attn_mlp_readout_can_fit_a_location():
    """A content-dependent location -- where a motif sits -- is learnable by the readout.

    The linear mean-pool readout cannot represent it. A short fit on a synthetic task:
    the global target is the centred position of a planted GGGG; the attention readout
    should reach a low error where the linear one stays near the variance.
    """
    torch.manual_seed(0)
    n, L = 512, 40
    x = torch.zeros(n, 4, L)
    base = torch.randint(0, 2, (n, L))            # A/T background: no GGGG by chance
    x[torch.arange(n)[:, None], base * 3, torch.arange(L)[None, :]] = 1.0
    pos = torch.randint(2, L - 6, (n,))
    for i in range(n):
        x[i, :, pos[i]:pos[i] + 4] = 0.0
        x[i, 2, pos[i]:pos[i] + 4] = 1.0          # G
    centre = (pos.float() + 1.5 - (L - 1) / 2) / ((L - 1) / 2)

    def fit(readout):
        torch.manual_seed(1)
        head = P.GlobalReadout(32, 1, rank=8) if readout == "attn_mlp" else \
            P.LowRankHead(32, 1, rank=8)
        trunk = torch.nn.Sequential(torch.nn.Conv1d(4, 32, 5, padding=2), torch.nn.GELU(),
                                    torch.nn.Conv1d(32, 32, 5, padding=2), torch.nn.GELU())
        params = list(trunk.parameters()) + list(head.parameters())
        opt = torch.optim.Adam(params, lr=3e-3)
        for _ in range(300):
            h = trunk(x)
            pred = head(h)[0] if readout == "attn_mlp" else head(h.mean(-1, keepdim=True))[0]
            loss = ((pred.view(-1) - centre) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return float(loss) / float(centre.var())

    attn, lin = fit("attn_mlp"), fit("linear")
    assert attn < 0.1, attn
    assert attn < 0.5 * lin, (attn, lin)


# ======================================================================================
# The held-out test side
# ======================================================================================

def test_three_way_split_is_disjoint_and_keeps_groups_whole():
    sys.path.insert(0, os.path.join(ROOT, "src", "model"))
    import train as T
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3000, size=10000)
    tr, va, te, rep = T.three_way_split(len(labels), labels, 0.05, 0.05, 0, "similarity")
    assert len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0 \
        and len(set(va) & set(te)) == 0
    assert len(tr) + len(va) + len(te) == len(labels)
    for a, b in ((tr, te), (va, te), (tr, va)):
        assert not (set(labels[a]) & set(labels[b])), "a similarity group straddles sides"
    assert 0.04 < len(te) / len(labels) < 0.06 and 0.04 < len(va) / len(labels) < 0.06


def test_two_way_split_is_unchanged_when_test_fraction_is_zero():
    """Runs already in flight must resume onto exactly the split they started with."""
    sys.path.insert(0, os.path.join(ROOT, "src", "model"))
    import train as T
    labels = np.random.default_rng(1).integers(0, 500, size=4000)
    tr0, va0, _ = T.similarity_split_from_labels(labels, 0.05, 0)
    tr1, va1, te1, _ = T.three_way_split(len(labels), labels, 0.05, 0.0, 0, "similarity")
    assert np.array_equal(np.sort(tr0), np.sort(tr1)) and np.array_equal(np.sort(va0),
                                                                         np.sort(va1))
    assert len(te1) == 0
    trr0, var0 = T.split_indices(4000, 0.05, 0)
    trr1, var1, ter1, _ = T.three_way_split(4000, None, 0.05, 0.0, 0, "random")
    assert np.array_equal(trr0, trr1) and np.array_equal(var0, var1)


# ======================================================================================
# Pair-map attention bias
# ======================================================================================

def test_pair_bias_starts_as_the_baseline(plan, x):
    """Zero-initialised output projection: identical predictions to the same weights
    without the pair module, so the module can only add what it learns."""
    torch.manual_seed(0)
    a = P.PANGEA(plan, width=64, n_blocks=2, rank=8, window_ladder=(10,), n_attn=2,
                 attn_heads=4).eval()
    torch.manual_seed(0)
    b = P.PANGEA(plan, width=64, n_blocks=2, rank=8, window_ladder=(10,), n_attn=2,
                 attn_heads=4, pair_bias=True).eval()
    missing = b.load_state_dict(a.state_dict(), strict=False)
    assert all(k.startswith("pair.") for k in missing.missing_keys)
    with torch.no_grad():
        oa, ob = a(x), b(x)
    for key in oa:
        assert torch.allclose(oa[key][0], ob[key][0], atol=1e-5), key


def test_pair_bias_keeps_every_parity(plan, x):
    torch.manual_seed(0)
    m = P.PANGEA(plan, width=64, n_blocks=2, rank=8, window_ladder=(10,), n_attn=2,
                 attn_heads=4, pair_bias=True).eval()
    with torch.no_grad():                         # a non-trivial bias, not the zero init
        for p_ in m.pair.out.parameters():
            p_.normal_(0, 0.5)
    for key in ("per_bp", "global", "window_10"):
        for tag in ("symmetric", "antisymmetric"):
            assert _parity_residual(m, x, tag, key) < 1e-4, (key, tag)


def test_pair_bias_sees_complementarity():
    """The bias for a Watson-Crick pair differs from a mismatch once trained to: the
    module can represent base-pair identity, which a 1-D trunk cannot place in a pair."""
    torch.manual_seed(0)
    pb = P.PairBias(n_bias=1, channels=8, n_layers=1)
    x = P.one_hot(["ACGTACGTAC"])
    opt = torch.optim.Adam(pb.parameters(), lr=0.05)
    idx = x.argmax(1)[0]
    comp = torch.tensor([[1.0 if int(a) + int(b) == 3 else 0.0 for b in idx] for a in idx])
    for _ in range(200):
        loss = ((pb(x)[0, 0] - comp) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < 0.01


def test_pooled_window_readout_is_the_pooled_per_bp_output(plan, x):
    """Window means under the pooled readout are exactly the pooled per-bp means."""
    import sys as _s
    _s.path.insert(0, os.path.join(ROOT, "src", "model"))
    from data import pooled_window_targets
    torch.manual_seed(0)
    m = P.PANGEA(plan, width=64, n_blocks=2, rank=8, window_ladder=(10,)).eval()
    m.window_readout = "pooled"
    with torch.no_grad():
        out = m(x)
    assert torch.allclose(out["window_10"][0], pooled_window_targets(out["per_bp"][0], 10),
                          equal_nan=True)


def test_pooled_window_readout_is_equivariant_on_interior_windows(plan, x):
    """Exact on every window that lies inside the fragment; not at the partial edge windows.

    The window targets themselves are not reverse-complement symmetric at the ends for
    channels with a reflection shift: a partial window trims a different set of positions
    in the forward and the reflected profile (pooling an exactly reflected pair of
    profiles mismatches by 0.57 at the edges and 2e-7 inside). The pooled readout
    reproduces that definition; the head readout imposed a symmetry the targets lack.
    """
    torch.manual_seed(0)
    m = P.PANGEA(plan, width=64, n_blocks=2, rank=8, window_ladder=(10,)).eval()
    m.window_readout = "pooled"
    with torch.no_grad():
        a, b = m(x), m(P.reverse_complement(x))
    fm, rm = a["window_10"][0], b["window_10"][0]
    rr, valid = m.reflect(rm, P.window_index_offset(10))
    n, k = fm.shape[-1], 10
    valid[:, :k - 1] = False
    valid[:, n - k + 1:] = False
    sign = torch.tensor([-1.0 if q == "antisymmetric" else 1.0 for q in plan.per_bp_parity])
    keep = torch.tensor([q in ("symmetric", "antisymmetric") for q in plan.per_bp_parity])
    mm = (valid & keep[:, None])[None].expand(fm.shape[0], -1, -1)
    d = (fm - sign[None, :, None] * rr).abs()[mm]
    assert int(mm.sum()) > 500 and float(d.max()) < 1e-5


# ---------------------------------------------------------------------------------------
# PairStack (pair_module="axial")
def _axial(plan, **kw):
    torch.manual_seed(0)
    return P.PANGEA(plan, width=64, n_blocks=2, rank=8, n_attn=2, attn_heads=4,
                    pair_bias=True, pair_module="axial", pair_channels=16, pair_layers=2,
                    pair_heads=4, **kw)


def test_axial_pair_module_preserves_every_parity(plan, x):
    m = _axial(plan).eval()
    with torch.no_grad():                       # make the zero-initialised outputs live
        for p_ in (m.pair_stack.bias_out, m.pair_stack.summary):
            p_.weight.normal_(0, 0.05); p_.bias.normal_(0, 0.05)
    for tag in ("symmetric", "antisymmetric"):
        for key in ("per_bp", "global"):
            assert _parity_residual(m, x, tag, key) < 1e-4, (tag, key)


def test_axial_pair_module_starts_as_the_model_without_it(plan, x):
    """Zero-initialised outputs: at initialisation the pair stack changes nothing."""
    with_ = _axial(plan).eval()
    torch.manual_seed(0)
    without = P.PANGEA(plan, width=64, n_blocks=2, rank=8, n_attn=2, attn_heads=4).eval()
    sd = {k: v for k, v in with_.state_dict().items() if not k.startswith("pair_stack.")}
    without.load_state_dict(sd)
    with torch.no_grad():
        a, b = with_(x), without(x)
    for key in ("per_bp", "global"):
        assert torch.allclose(a[key][0], b[key][0], atol=1e-5), key


def test_axial_pair_map_is_symmetric_and_trains(plan, x):
    m = _axial(plan).train()
    h = torch.randn(x.shape[0], 64, x.shape[-1])
    with torch.no_grad():
        m.pair_stack.bias_out.weight.normal_(0, 0.05)
        bias, summ = m.pair_stack(h, x)
    assert torch.allclose(bias, bias.transpose(-1, -2), atol=1e-5)
    assert summ.shape == h.shape
    out = m(x)
    (out["per_bp"][0].nan_to_num().pow(2).mean() + out["global"][0].nan_to_num().pow(2).mean()).backward()
    g = [p_.grad for n_, p_ in m.named_parameters() if n_.startswith("pair_stack.")]
    assert all(t is not None for t in g) and any(float(t.abs().sum()) > 0 for t in g)


def test_model_kwargs_from_args_round_trips_old_and_new_configs():
    old = dict(width=64, blocks=2, rank=8, windows=[10], max_dilation=8, dropout=0.0,
               attn_layers=2, attn_heads=4)
    kw = P.model_kwargs_from_args(old)
    assert kw["pair_module"] == "conv" and kw["pair_bias"] is False
    new = dict(old, pair_bias=True, pair_module="axial", pair_heads=2)
    kw = P.model_kwargs_from_args(new)
    assert kw["pair_module"] == "axial" and kw["pair_heads"] == 2


# ---------------------------------------------------------------------------------------
# Strand tie and base-pair-probability head
def _stranded_plan():
    pb = ["L_p_unpaired_prob", "L_m_unpaired_prob", "L_p_positional_entropy",
          "L_m_positional_entropy", "X_sym", "X_anti", "X_str"]
    gl = ["L_p_mfe_kcal", "L_m_mfe_kcal", "X_gsym", "X_gstr"]
    return P.ChannelPlan.from_metadata(
        pb, gl, ["stranded"] * 4 + ["symmetric", "antisymmetric", "stranded"],
        ["stranded", "stranded", "symmetric", "stranded"],
        ["L"] * 4 + ["B", "B", "C"], ["L", "L", "B", "C"], [0] * len(pb))


def _tied(**kw):
    torch.manual_seed(0)
    return P.PANGEA(_stranded_plan(), width=64, n_blocks=2, rank=8, n_attn=2, attn_heads=4,
                    pair_bias=True, pair_module="axial", pair_channels=16, pair_layers=2,
                    window_ladder=(10,), **kw).eval()


def test_strand_tie_makes_minus_of_x_the_plus_of_its_reverse_complement(x):
    m = _tied(strand_tie=True)
    with torch.no_grad():
        a, b = m(x), m(P.reverse_complement(x))
    for key, (ip, im) in (("per_bp", ([0, 2], [1, 3])), ("global", ([0], [1]))):
        for k in (0, 1):                                   # means and log-variances
            assert torch.allclose(a[key][k][:, im], b[key][k][:, ip], atol=1e-5), (key, k)
            assert torch.allclose(a[key][k][:, ip], b[key][k][:, im], atol=1e-5), (key, k)


def test_strand_tie_leaves_every_other_channel_alone(x):
    tied, free = _tied(strand_tie=True), _tied(strand_tie=False)
    free.load_state_dict(tied.state_dict())
    with torch.no_grad():
        a, b = tied(x), free(x)
    assert torch.allclose(a["per_bp"][0][:, [0, 2, 4, 5, 6]], b["per_bp"][0][:, [0, 2, 4, 5, 6]])
    assert torch.allclose(a["global"][0][:, [0, 2, 3]], b["global"][0][:, [0, 2, 3]])
    assert not torch.allclose(a["per_bp"][0][:, 1], b["per_bp"][0][:, 1])


def test_bpp_functionals_match_the_vienna_formulas():
    torch.manual_seed(0)
    n = 30
    raw = torch.rand(2, n, n) * 0.015          # rows sum below one, as real matrices do
    p = (raw + raw.transpose(1, 2)) * P.pairable_mask(n)
    logits = torch.logit(p.clamp(1e-7, 1 - 1e-7))
    f = P.bpp_functionals(logits)
    pm = torch.sigmoid(logits) * P.pairable_mask(n)
    u = 1 - pm.sum(-1)
    ent = -((torch.where(pm > 0, pm * torch.log2(pm), torch.zeros_like(pm))).sum(-1)
            + u * torch.log2(u))
    assert torch.allclose(f[..., 0], pm.sum(-1), atol=1e-5)
    assert torch.allclose(f[..., 1], ent, atol=1e-4)
    iu = torch.triu_indices(n, n, 1)
    up = pm[:, iu[0], iu[1]]                       # mean_bp_distance = 2 sum_{i<j} p(1-p)
    assert torch.allclose(f[..., 2].sum(-1), 2 * (up * (1 - up)).sum(-1), atol=1e-4)
    big = P.bpp_functionals(torch.full((1, n, n), 30.0))     # rows summing far above one
    assert torch.isfinite(big).all()


def test_bpp_logits_of_both_passes(x):
    m = _tied(strand_tie=True)
    with torch.no_grad():
        a = m(x, return_bpp=True)["bpp_logits"]
        b = m(P.reverse_complement(x), return_bpp=True)["bpp_logits"]
    B = x.shape[0]
    assert a.shape == (2 * B, x.shape[-1], x.shape[-1])
    assert torch.allclose(a, a.transpose(1, 2))
    assert torch.allclose(a[B:], b[:B], atol=1e-5) and torch.allclose(a[:B], b[B:], atol=1e-5)


def test_bpp_loss_reaches_the_pair_stack(x):
    m = _tied(strand_tie=True).train()
    out = m(x, return_bpp=True)
    target = torch.rand_like(out["bpp_logits"])
    torch.nn.functional.binary_cross_entropy_with_logits(out["bpp_logits"], target).backward()
    assert float(m.pair_stack.bpp_out.weight.grad.abs().sum()) > 0
    assert m.pair_stack.blocks[0].qkv_r.weight.grad is not None


def test_bpp_functionals_have_finite_gradients_everywhere():
    """Unpairable entries are exactly zero; their entropy term must not poison the gradient."""
    torch.manual_seed(0)
    logits = (torch.randn(2, 25, 25) * 8).requires_grad_(True)
    with torch.no_grad():
        logits[0, 0, 5] = -200.0                          # sigmoid underflows to exactly 0
    P.bpp_functionals(logits).sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_conv_pair_bias_base_pair_head(x):
    plan = _stranded_plan()
    torch.manual_seed(0)
    m = P.PANGEA(plan, width=64, n_blocks=2, rank=8, n_attn=2, attn_heads=4, pair_bias=True,
                 pair_module="conv", pair_bpp=True, strand_tie=True, window_ladder=(10,)).eval()
    torch.manual_seed(0)
    old = P.PANGEA(plan, width=64, n_blocks=2, rank=8, n_attn=2, attn_heads=4, pair_bias=True,
                   pair_module="conv", strand_tie=True, window_ladder=(10,)).eval()
    extra = set(m.state_dict()) - set(old.state_dict())
    assert extra == {"pair.bpp_out.weight", "pair.bpp_out.bias", "pair_feat.weight",
                     "pair_feat.bias"}, extra                # old checkpoints still load
    with torch.no_grad():
        a = m(x, return_bpp=True)
        b = m(P.reverse_complement(x), return_bpp=True)
    B = x.shape[0]
    assert a["bpp_logits"].shape == (2 * B, x.shape[-1], x.shape[-1])
    assert torch.allclose(a["bpp_logits"][B:], b["bpp_logits"][:B], atol=1e-5)
    sd = {k: v for k, v in m.state_dict().items() if k in old.state_dict()}
    old.load_state_dict(sd)
    with torch.no_grad():
        c = old(x)
    assert torch.allclose(a["per_bp"][0], c["per_bp"][0], atol=1e-5)   # zero-init path


def test_watson_crick_tie_for_listed_global_stems(x):
    pb = ["L_p_unpaired_prob", "L_m_unpaired_prob", "X_sym"]
    gl = ["G_max_cleavage_W", "G_max_cleavage_C", "C_mean_az_W__sin", "C_mean_az_C__sin",
          "C_mean_az_W__cos", "C_mean_az_C__cos", "G_other_W", "G_other_C"]
    plan = P.ChannelPlan.from_metadata(pb, gl, ["stranded", "stranded", "symmetric"],
                                       ["stranded"] * len(gl), ["L", "L", "B"],
                                       ["G", "G", "C", "C", "C", "C", "G", "G"], [0, 0, 0])
    torch.manual_seed(0)
    m = P.PANGEA(plan, width=64, n_blocks=2, rank=8, window_ladder=(10,),
                 wc_tie_global=("G_max_cleavage", "C_mean_az")).eval()
    assert m.apply_ties and len(m.pb_tie_p) == 0          # no Family L tie unless asked
    with torch.no_grad():
        a, b = m(x), m(P.reverse_complement(x))
    for w, c in ((0, 1), (2, 3), (4, 5)):
        assert torch.allclose(a["global"][0][:, c], b["global"][0][:, w], atol=1e-5)
    # unlisted stem untouched: its C value is the forward pass's own
    torch.manual_seed(0)
    free = P.PANGEA(plan, width=64, n_blocks=2, rank=8, window_ladder=(10,)).eval()
    free.load_state_dict(m.state_dict())
    with torch.no_grad():
        f = free(x)
    assert torch.allclose(a["global"][0][:, 7], f["global"][0][:, 7])
    with pytest.raises(ValueError):
        P.PANGEA(plan, width=64, n_blocks=2, rank=8, window_ladder=(10,), wc_tie_global=("nope",))


def test_strand_tie_covers_family_k_hybrid_pairs(x):
    pb = ["K_hybp_twist", "K_hybm_twist", "K_rna_twist", "L_p_unpaired_prob", "L_m_unpaired_prob"]
    gl = ["K_hybp_duplex_dG37_kcal", "K_hybm_duplex_dG37_kcal", "X_g"]
    plan = P.ChannelPlan.from_metadata(pb, gl, ["stranded", "stranded", "symmetric", "stranded", "stranded"],
                                       ["stranded", "stranded", "symmetric"], ["K", "K", "K", "L", "L"],
                                       ["K", "K", "B"], [0, 0, -1, 0, 0])
    torch.manual_seed(0)
    m = P.PANGEA(plan, width=64, n_blocks=2, rank=8, window_ladder=(10,), strand_tie=True).eval()
    assert len(m.pb_tie_p) == 2 and len(m.gl_tie_p) == 1
    with torch.no_grad():
        a, b = m(x), m(P.reverse_complement(x))
    assert torch.allclose(a["per_bp"][0][:, 1], b["per_bp"][0][:, 0], atol=1e-5)
    assert torch.allclose(a["global"][0][:, 1], b["global"][0][:, 0], atol=1e-5)


def test_per_channel_beta_matches_scalar_and_frees_rare_channels():
    from pangea import LandscapeLoss
    torch.manual_seed(0)
    fam_pb, fam_gl = ["A", "A", "B"], ["A", "B"]
    pb_m = torch.randn(4, 3, 7, requires_grad=True); pb_v = torch.randn(4, 3, 7)
    gl_m = torch.randn(4, 2, requires_grad=True); gl_v = torch.randn(4, 2)
    tgt = {"per_bp": torch.randn(4, 3, 7), "global": torch.randn(4, 2)}
    preds = {"per_bp": (pb_m, pb_v), "global": (gl_m, gl_v)}
    l0, _ = LandscapeLoss(fam_pb, fam_gl, beta=0.5)(preds, tgt)
    l1, _ = LandscapeLoss(fam_pb, fam_gl, beta=0.5, beta_per_bp=[0.5] * 3, beta_global=[0.5] * 2)(preds, tgt)
    assert torch.allclose(l0, l1)
    # beta = 1 on one channel: its mean gradient no longer depends on the predicted variance
    lossA = LandscapeLoss(fam_pb, fam_gl, beta=0.5, beta_global=[1.0, 0.5])
    g = []
    for scale in (0.0, 3.0):
        m = gl_m.detach().clone().requires_grad_(True)
        v = gl_v.clone(); v[:, 0] += scale
        loss, _ = lossA({"per_bp": (pb_m.detach(), pb_v), "global": (m, v)}, tgt)
        loss.backward()
        g.append(m.grad[:, 0].clone())
    assert torch.allclose(g[0], g[1], atol=1e-5)
