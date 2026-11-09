import sys

sys.path.insert(0, ".")

import pytest
from transformers import AutoModelForCausalLM

from topological_mold import build_report, freeze, load_tok, logit_lens, pour, render_html

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
