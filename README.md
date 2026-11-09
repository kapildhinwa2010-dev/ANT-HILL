# 🐜 ANT-HILL

[![CI](https://github.com/kapildhinwa2010-dev/ANT-HILL/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/ANT-HILL/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/ANT-HILL/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/ANT-HILL/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 The full 3D map of the inside of an AI

Like the classic ant-colony experiment - pour cement into the hill, let it
reach every smallest corner, let it harden, then excavate and photograph the
complete 3D structure - ANT-HILL does this to a language model:

- 💧 **POUR** - tiny random drops perturb the input gate (embedding); they
  flow wherever information actually flows, into every live corner of the net
- ❄️ **FREEZE** - flow sensitivity is measured at every layer x token
  position and thresholded into a binary occupancy mold
- 📸 **SNAP** - the frozen mold becomes a 3D voxel grid
  (layers x positions x channel clusters), every occupied spot labeled with a
  🔍 **logit-lens** readout of what the model is "thinking" there

## 🌐 The 3D viewer

    mold_map.html - interactive 3D scatter of every frozen voxel
    x = layer | y = token position | z = channel cluster
    hover any voxel to see its logit-lens thought

## ⚡ Install

GPU (default):

    pip install torch
    pip install -r requirements.txt

<details>
<summary>🖥️ CPU-only variant</summary>

    pip install --index-url https://download.pytorch.org/whl/cpu torch
    pip install -r requirements.txt

</details>

## 💻 Usage

    python topological_mold.py --model Qwen/Qwen2.5-0.5B-Instruct

Outputs: 📊 mold_report.json, 🧊 mold_map.npz, 🌐 mold_map.html

## 🩺 What you can diagnose

- 💀 **Dead zones** - layers/positions the liquid never reached: no flow, no function
- 🌊 **Flow breaks** - where occupancy dies between two layers
- 🧠 **Thought map** - what the model is computing at each spot (logit lens)
- 🗺️ **The full naksha** - one 3D picture instead of 24 flat plots

## 🧪 Tests

    pytest -q

<details>
<summary>🔬 Honest limitations</summary>

- Input-gate perturbation measures first-order flow reachability, not every
  nonlinear interaction; it is a mold of the channels, not the water itself.
- Channel clusters are plain slices of the hidden dimension - a cheap 3D axis,
  not a learned decomposition.
- Logit-lens labels are the standard approximation: hidden states projected
  through the output head.

</details>

## ⚖️ License

MIT - see [LICENSE](LICENSE).
