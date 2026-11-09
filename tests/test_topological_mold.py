import sys

sys.path.insert(0, ".")

import pytest
import torch
from transformers import AutoModelForCausalLM

from topological_mold import build_report, freeze, load_tok, logit_lens, pour, record_activations, render_html

MODEL = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def setup():
    tok = load_tok(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.eval()
    base, intensity = pour(model, tok, "Hello world, this is a test", n_drops=3, max_len=12)
    return model, tok, base, intensity


def test_pour_normalized(setup):
    model, tok, base, intensity = setup
    for i in intensity:
        assert float(intensity[i].max()) <= 1.0 + 1e-6
        assert float(intensity[i].min()) >= 0.0


def test_freeze_shapes(setup):
    model, tok, base, intensity = setup
    occ, inten3, frozen = freeze(base, intensity, z_clusters=4, frozen_thresh=0.1)
    L = len(intensity)
    T = base[0].shape[0]
    assert frozen.shape == (L, T, 4)
    assert occ.shape == (L, T)
    assert set(frozen.unique().tolist()).issubset({0.0, 1.0})
    assert 0.0 <= float(occ.mean()) <= 1.0


def test_logit_lens(setup):
    model, tok, base, intensity = setup
    labels = logit_lens(model, tok, base)
    assert labels
    for i in labels:
        assert len(labels[i]) == base[i].shape[0]
        assert all(isinstance(x, str) for x in labels[i])


def test_html_and_report(setup, tmp_path):
    model, tok, base, intensity = setup
    occ, inten3, frozen = freeze(base, intensity, z_clusters=4, frozen_thresh=0.05)
    labels = logit_lens(model, tok, base)
    out = str(tmp_path / "mold_map.html")
    render_html(frozen, labels, out_path=out)
    with open(out) as f:
        html = f.read()
    assert "ANT-HILL" in html
    assert "plotly" in html.lower()
    report = build_report(intensity, occ, labels)
    assert report["n_layers"] == len(intensity)
    assert 0.0 <= report["dead_layer_fraction"] <= 1.0


def test_pour_rejects_zero_drops_before_model_use():
    with pytest.raises(ValueError, match="n_drops"):
        pour(None, None, None, n_drops=0)


def test_freeze_rejects_invalid_cluster_counts():
    base = {0: torch.ones(2, 5)}
    intensity = {0: torch.ones(2, 5)}
    with pytest.raises(ValueError, match="z_clusters"):
        freeze(base, intensity, z_clusters=0)
    with pytest.raises(ValueError, match="hidden dimension"):
        freeze(base, intensity, z_clusters=6)


def test_freeze_supports_uneven_cluster_widths():
    base = {0: torch.arange(10, dtype=torch.float32).reshape(2, 5)}
    intensity = {0: torch.ones(2, 5)}
    _, _, frozen = freeze(base, intensity, z_clusters=3)
    assert frozen.shape == (1, 2, 3)


def test_record_activations_removes_hooks_after_failure():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = type("ModelParts", (), {})()
            self.model.layers = torch.nn.ModuleList([torch.nn.Identity()])
            self.fail = True

        def forward(self, ids=None, inputs_embeds=None):
            value = ids if inputs_embeds is None else inputs_embeds
            for layer in self.model.layers:
                value = layer(value)
            if self.fail:
                raise RuntimeError("controlled forward failure")
            return value

    model = TinyModel()
    values = torch.ones(1, 2, 3)
    with pytest.raises(RuntimeError, match="controlled forward failure"):
        record_activations(model, ids=values)
    assert all(not layer._forward_hooks for layer in model.model.layers)

    model.fail = False
    activations = record_activations(model, ids=values)
    assert activations[0].shape == (2, 3)
    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_html_escapes_script_terminators_in_labels(tmp_path):
    frozen = torch.ones(1, 1, 1)
    malicious = "</script><script>alert('xss')</script>"
    out = tmp_path / "safe.html"
    render_html(frozen, {0: [malicious]}, out_path=str(out))
    html = out.read_text()
    assert malicious not in html
    assert "\\u003c/script\\u003e" in html
