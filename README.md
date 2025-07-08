# 🚀 Gated Recurrent Fusion (GRF): Efficient Multimodal Learning with Fewer Parameters

> 📄 Preprint: [arXiv:2507.02985](https://arxiv.org/pdf/2507.02985)  
> 🧪 Submitted to: ACMMM 2025 Workshop UAVM
> 🔥 TL;DR: We propose **GRF**, a lightweight gated recurrent fusion module that outperforms MulT on CMU-MOSI unaligned setting using 3× fewer parameters.

---

## 🧠 Overview

Multimodal models like MulT are powerful but suffer from computational overhead and sensitivity to alignment. We propose **Gated Recurrent Fusion (GRF)** — a sparse, parameter-efficient fusion module that captures cross-modal dynamics without relying on input alignment.

---

## 📦 Features

- ✅ Gated Recurrent Fusion (GRF) for unaligned multimodal fusion  
- ✅ Modular design compatible with any modality order  
- ✅ Includes ablation studies and visualization tools  
- ✅ Reproducible experiments using config files and MLflow

---

## 📊 Results

| Model     | F1 Score | Parameters | Dataset    | Alignment |
|-----------|----------|------------|------------|-----------|
| MulT      | **81**   | ~8M        | CMU-MOSI   | Unaligned |
| GRF (Ours)| 79       | 4.5M       | CMU-MOSI   | Unaligned |

For full details, see the [arXiv paper](https://arxiv.org/pdf/2507.02985).

---

## 📁 Project Structure
```
GRF
├── src/ # GRF model definitions, data handlers, utils, and engine
├── data/ # Dataloaders for CMU-MOSI
├── train.py # Training script
├── configs/ # YAML config files for training setups
├── requirements.txt # Python dependencies
└── README.md # You're here
```

## 🔧 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yushi2006/GRF.git
cd GRF
pip install -r requirements.txt
```

## 🏃‍♂️ Quick Start

### Train the GRF Model

```bash
chmod +x run_experiments.sh

./run_experiments.sh
```

## 📚 Citation
```
@misc{shihata2025gatedrecursivefusionstateful,
      title={Gated Recursive Fusion: A Stateful Approach to Scalable Multimodal Transformers}, 
      author={Yusuf Shihata},
      year={2025},
      eprint={2507.02985},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.02985}, 
}
```


