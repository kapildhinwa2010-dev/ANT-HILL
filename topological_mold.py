"""ANT-HILL: pour a topological liquid into an AI, freeze it, read the 3D map.

Like the classic ant-colony experiment: pour cement into the hill, let it
reach every smallest corner, let it harden, then excavate and photograph the
full 3D structure. Here:

- POUR:   tiny random drops perturb the input gate (embedding); they flow
          wherever information actually flows in the network.
- FREEZE: flow sensitivity is measured at every layer x token position and
          thresholded into a binary occupancy mold.
- SNAP:   the frozen mold becomes a 3D voxel grid
          (layers x positions x channel clusters), and every occupied spot is
          labeled with a logit-lens readout of what the model is "thinking"
          there.

Artifacts: mold_report.json, mold_map.npz, mold_map.html (3D viewer).

CPU execution currently; GPU placement is not implemented. No training.
Bounded evaluation.
"""

__version__ = "0.1.0"

import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("unsupported architecture")


def load_tok(name):
    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception:
        return AutoTokenizer.from_pretrained("gpt2")


@torch.no_grad()
def record_activations(model, ids=None, embeds=None):
    """Run one forward pass; return {layer_index: (positions, dim) tensor}."""
    layers = get_layers(model)
    acts = {i: [] for i in range(len(layers))}
    handles = []

    def make_hook(idx):
        def hook(mod, inp, out):
            y = out[0] if isinstance(out, tuple) else out
            acts[idx].append(y.detach().float().cpu())
        return hook

    for i, lyr in enumerate(layers):
        handles.append(lyr.register_forward_hook(make_hook(i)))
    try:
        if embeds is None:
            model(ids)
        else:
            model(inputs_embeds=embeds)
    finally:
        for h in handles:
            h.remove()
    return {i: torch.cat(v, dim=0)[0] for i, v in acts.items()}


def pour(model, tok, prompt, n_drops=8, eps=0.05, max_len=48, seed=0):
    """Drop random liquid at the input gate; measure flow sensitivity per layer."""
    if n_drops <= 0:
        raise ValueError("n_drops must be greater than zero")
    ids = tok(prompt, return_tensors="pt").input_ids[:, :max_len]
    emb = model.get_input_embeddings()(ids)
    base = record_activations(model, ids=ids)

    g = torch.Generator().manual_seed(seed)
    flow = {i: torch.zeros_like(base[i]) for i in base}
    for _ in range(n_drops):
        noise = torch.randn(emb.shape, generator=g) * eps
        acts = record_activations(model, embeds=emb + noise)
        for i in base:
            flow[i] += (acts[i] - base[i]).abs()
    intensity = {}
    for i in flow:
        flow[i] /= n_drops
        intensity[i] = flow[i] / (flow[i].max() + 1e-12)
    return base, intensity


def freeze(base, intensity, z_clusters=8, occupancy_percentile=80, frozen_thresh=0.3):
    """Threshold flow into a binary 3D mold: (layers x positions x channel clusters)."""
    if z_clusters <= 0:
        raise ValueError("z_clusters must be greater than zero")
    keys = sorted(intensity.keys())
    L, T = len(keys), intensity[keys[0]].shape[0]
    if z_clusters > base[keys[0]].shape[1]:
        raise ValueError("z_clusters cannot exceed the hidden dimension")
    occ = torch.zeros(L, T)
    inten3 = torch.zeros(L, T, z_clusters)
    for k, i in enumerate(keys):
        row = intensity[i].mean(dim=1)
        if row.numel() > 1:
            thr = torch.quantile(row, 1.0 - occupancy_percentile / 100.0)
            occ[k] = (row > thr).float()
        else:
            occ[k] = 1.0
        B = base[i].abs()
        if z_clusters > B.shape[1]:
            raise ValueError("z_clusters cannot exceed the hidden dimension")
        # Unlike chunk(), tensor_split returns exactly z_clusters slices for
        # uneven widths as long as each requested slice can contain a channel.
        chunks = torch.stack([c.mean(dim=1) for c in torch.tensor_split(B, z_clusters, dim=1)], dim=1)
        inten3[k] = chunks
    inten3 = inten3 / (inten3.max() + 1e-12)
    frozen = (inten3 * occ.unsqueeze(-1) > frozen_thresh).float()
    return occ, inten3, frozen


@torch.no_grad()
def logit_lens(model, tok, base):
    """Decode what the model is 'thinking' at each spot."""
    head = getattr(model, "lm_head", None)
    if head is None:
        return {i: [] for i in base}
    labels = {}
    for i in base:
        logits = head(base[i])
        top = logits.argmax(dim=-1)
        labels[i] = [tok.decode([int(t)]) for t in top]
    return labels


def build_report(intensity, occ, labels):
    keys = sorted(intensity.keys())
    per_layer = []
    for k in range(len(keys)):
        per_layer.append({
            "layer": k,
            "occupied_fraction": round(float(occ[k].mean()), 4),
            "sample_thoughts": labels.get(k, [])[:8],
        })
    return {
        "n_layers": len(keys),
        "dead_layer_fraction": round(float((occ.mean(dim=1) == 0).float().mean()), 4),
        "per_layer": per_layer,
    }


def render_html(frozen, labels, out_path="mold_map.html"):
    """Interactive 3D voxel viewer (plotly scatter3d)."""
    L, T, Z = frozen.shape
    pts = []
    for l in range(L):
        for t in range(T):
            for z in range(Z):
                if frozen[l, t, z] > 0:
                    lab = labels.get(l, [])[t] if t < len(labels.get(l, [])) else ""
                    pts.append([l, t, z, lab])
    step = max(1, len(pts) // 2000)
    pts = pts[::step]
    xs = json.dumps([p[0] for p in pts])
    ys = json.dumps([p[1] for p in pts])
    zs = json.dumps([p[2] for p in pts])
    txt = json.dumps(["layer " + str(p[0]) + " | pos " + str(p[1]) + " | z " + str(p[2]) + " | " + p[3] for p in pts])
    # Prevent model/tokenizer-controlled strings from terminating the inline script.
    txt = txt.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    head = "<!DOCTYPE html>" + chr(10) + "<html><head><meta charset='utf-8'><title>ANT-HILL mold map</title>" + chr(10)
    head += "<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script></head><body>" + chr(10)
    head += "<h2>ANT-HILL - frozen topological mold</h2><div id='mold'></div><script>" + chr(10)
    body = "Plotly.newPlot('mold', [{type:'scatter3d', mode:'markers', x:" + xs + ", y:" + ys + ", z:" + zs + ", text:" + txt + ", hoverinfo:'text', marker:{size:3, opacity:0.8}}], "
    body += "{title:'ANT-HILL 3D mold map', scene:{xaxis:{title:'layer'}, yaxis:{title:'token position'}, zaxis:{title:'channel cluster'}}});" + chr(10)
    tail = "</script></body></html>" + chr(10)
    with open(out_path, "w") as f:
        f.write(head + body + tail)
    return out_path


def main():
    ap = argparse.ArgumentParser(prog="ant-hill", description="Pour, freeze and snap: a 3D topological mold of a language model.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default="Tokyo is the capital of Japan, a country in East Asia.")
    ap.add_argument("--drops", type=int, default=8)
    ap.add_argument("--z-clusters", type=int, default=8)
    ap.add_argument("--frozen-thresh", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tok = load_tok(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    model.eval()

    base, intensity = pour(model, tok, a.prompt, n_drops=a.drops)
    occ, inten3, frozen = freeze(base, intensity, z_clusters=a.z_clusters, frozen_thresh=a.frozen_thresh)
    labels = logit_lens(model, tok, base)
    report = build_report(intensity, occ, labels)

    np.savez("mold_map.npz", frozen=frozen.numpy(), intensity=inten3.numpy(), occupancy=occ.numpy())
    render_html(frozen, labels)
    with open("mold_report.json", "w") as f:
        json.dump(report, f, indent=2)

    total_voxels = int(frozen.sum())
    print("layers:", frozen.shape[0], "| positions:", frozen.shape[1], "| z clusters:", frozen.shape[2])
    print("frozen voxels:", total_voxels, "| dead layers:", report["dead_layer_fraction"])
    print("saved: mold_map.npz, mold_map.html, mold_report.json")


if __name__ == "__main__":
    main()
