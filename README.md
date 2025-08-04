<div align="center">

<h1 style="font-family: 'Arial', sans-serif; font-size: 28px; font-weight: bold; color: #f0f0f0;">
    🔍 Has Machine Translation Evaluation Achieved Human Parity?<br>
    The Human Reference and the Limits of Progress
</h1>

[![Conference](https://img.shields.io/badge/ACL-2025-4b44ce
)](https://2025.aclweb.org/)
[![Paper](http://img.shields.io/badge/paper-ACL--anthology-B31B1B.svg)](https://aclanthology.org/2025.acl-short.63/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

[![](https://shields.io/badge/-MT%20Metrics%20Eval-green?style=flat&logo=github&labelColor=gray)](https://github.com/google-research/mt-metrics-eval)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)

</div>

## ⚙️ Setup

The code in this repo requires Python 3.10 or higher. We recommend creating a new [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) environment as follows:

```bash
conda create -n human-parity-mt-eval python=3.10
conda activate human-parity-mt-eval
pip install --upgrade pip
```

All scripts included within this repository require cloning and installing the [Google WMT Metrics evaluation repository](https://github.com/google-research/mt-metrics-eval). To do this, execute the following commands:

```bash
git clone https://github.com/google-research/mt-metrics-eval.git
cd mt-metrics-eval
pip install .
```

Then, download the WMT Metrics evaluation datasets:

```bash
alias mtme='python3 -m mt_metrics_eval.mtme'
mtme --download  # Puts ~2G of data into $HOME/.mt-metrics-eval.
```

## 📁 Data

The [data/](data/) directory contains all the information required to reproduce the analyses presented in our paper. The structure is organized by WMT evaluation year and language pair, and includes both human annotations and automatic metric outputs.

<details> <summary>📂 <strong>Click to expand the directory tree</strong></summary> <br>

```text
data
├── annotations
│   ├── wmt20
│   │   ├── en-de
│   │   │   ├── mqm-col1.pickle
│   │   │   ├── mqm-col2.pickle
│   │   │   ├── mqm-col3.pickle
│   │   │   ├── psqm-col1.pickle
│   │   │   ├── psqm-col2.pickle
│   │   │   └── psqm-col3.pickle
│   │   └── zh-en
│   │       ├── mqm-col1.pickle
│   │       ├── mqm-col2.pickle
│   │       ├── mqm-col3.pickle
│   │       ├── psqm-col1.pickle
│   │       ├── psqm-col2.pickle
│   │       └── psqm-col3.pickle
│   ├── wmt22
│   │   ├── en-de
│   │   │   ├── en-de.ESA-1.seg.score
│   │   │   ├── en-de.ESA-2.seg.score
│   │   │   ├── en-de.MQM-1.seg.score
│   │   │   ├── mqm-col1.pickle
│   │   │   ├── mqm-col2.pickle
│   │   │   └── mqm-col3.pickle
│   │   └── en-zh
│   │       ├── mqm-col1.pickle
│   │       ├── mqm-col2.pickle
│   │       └── mqm-col3.pickle
│   └── wmt23
│       ├── en-de
│       │   ├── mqm-col1_more_data.pickle
│       │   ├── mqm-col1.pickle
│       │   ├── mqm-col2_more_data.pickle
│       │   ├── mqm-col2.pickle
│       │   ├── mqm-col3_more_data.pickle
│       │   └── mqm-col3.pickle
│       └── zh-en
│           ├── mqm-col1.pickle
│           ├── mqm-col2.pickle
│           └── mqm-col3.pickle
├── metrics_info
│   ├── wmt20
│   │   └── out_paths.tsv
│   ├── wmt22
│   │   └── out_paths.tsv
│   └── wmt23
│       └── out_paths.tsv
├── metrics_outputs
│   ├── wmt20
│   │   ├── en-de
│   │   │   └── BLEURT-20
│   │   └── zh-en
│   │       └── BLEURT-20
│   ├── wmt22
│   │   ├── en-de
│   │   │   ├── CometKiwi-XL
│   │   │   ├── CometKiwi-XXL
│   │   │   ├── MetricX-23-QE-XXL
│   │   │   └── MetricX-23-XXL
│   │   └── en-zh
│   │       ├── CometKiwi-XL
│   │       ├── CometKiwi-XXL
│   │       ├── MetricX-23-QE-XXL
│   │       └── MetricX-23-XXL
│   └── wmt23
│       ├── en-de
│       │   ├── MetricX-23-QE-XXL
│       │   └── MetricX-23-XXL
│       └── zh-en
│           ├── MetricX-23-QE-XXL
│           └── MetricX-23-XXL
└── rankings
    ├── wmt20
    │   ├── en-de
    │   └── zh-en
    ├── wmt22
    │   ├── en-de
    │   └── en-zh
    ├── wmt23
    │   ├── en-de
    │   └── zh-en
    └── wmt24
        └── en-es
```

</details>

---

### 🧾 Description of the contents

- **📄 [annotations/](data/annotations)**  
  Contains **human annotations following MT evaluation protocols** (e.g., MQM, PSQM, ESA) across multiple WMT editions and language pairs. These are the human MT evaluators used in our analysis.

- **ℹ️ [metrics_info/](data/metrics_info)**  
  Stores **metadata about the additional automatic metrics** we included in our study (beyond those originally submitted to WMT). These metadata consist of metric names and output file paths.

- **📈 [metrics_outputs/](data/metrics_outputs)**  
  Includes the **actual outputs of the additional automatic metrics** for each WMT year and language pair.

- **🏆 [rankings/](data/rankings)**  
  This folder is used for the **final rankings of all evaluators (both automatic metrics and humans)**, as generated by the `run_mt_meta_eval.py` script.

## 🏃‍♂️ Running the code

To reproduce the results presented in our paper, you can run the `run_mt_meta_eval.py` script, which performs the meta-evaluation considering both automatic MT metrics and human evaluators.

---

### 📊 Reproducing Meta-Evaluation Results

<details>
<summary><strong>WMT20</strong> (click to expand)</summary>


<br>

#### 🌍 Language Pair: `en-de`

```bash
python scripts/run_mt_meta_eval.py \
    --wmt-year wmt20 \
    --lp en-de \
    --new-human-annotations-dir data/annotations/wmt20 \
    --gold-name --mqm-col1 \
    --new-metrics-path data/metrics_info/wmt20/out_paths.tsv > data/rankings/wmt20/en-de/ranking.txt
```

#### 🌍 Language Pair: `zh-en`

```bash
python scripts/run_mt_meta_eval.py \
    --wmt-year wmt20 \
    --lp zh-en \
    --new-human-annotations-dir data/annotations/wmt20 \
    --gold-name --mqm-col1 \
    --new-metrics-path data/metrics_info/wmt20/out_paths.tsv > data/rankings/wmt20/zh-en/ranking.txt
```

</details>

<details>
<summary><strong>WMT22</strong> (click to expand)</summary>


<br>

#### 🌍 Language Pair: `en-de`

```bash
python scripts/run_mt_meta_eval.py \
    --wmt-year wmt22 \
    --lp en-de \
    --new-human-annotations-dir data/annotations/wmt22 \
    --gold-name --mqm-col1 \
    --new-metrics-path data/metrics_info/wmt22/out_paths.tsv > data/rankings/wmt22/en-de/ranking.txt
```

#### 🌍 Language Pair: `en-zh`

```bash
python scripts/run_mt_meta_eval.py \
    --wmt-year wmt22 \
    --lp en-zh \
    --new-human-annotations-dir data/annotations/wmt22 \
    --gold-name --mqm-col1 \
    --new-metrics-path data/metrics_info/wmt22/out_paths.tsv > data/rankings/wmt22/en-zh/ranking.txt
```

</details>

<details>
<summary><strong>WMT23</strong> (click to expand)</summary>


<br>

#### 🌍 Language Pair: `en-de`

```bash
python scripts/run_mt_meta_eval.py \
    --wmt-year wmt23 \
    --lp en-de \
    --new-human-annotations-dir data/annotations/wmt23 \
    --gold-name --mqm-col1 \
    --new-metrics-path data/metrics_info/wmt23/out_paths.tsv > data/rankings/wmt23/en-de/ranking.txt
```

#### 🌍 Language Pair: `zh-en`

```bash
python scripts/run_mt_meta_eval.py \
    --wmt-year wmt23 \
    --lp zh-en \
    --new-human-annotations-dir data/annotations/wmt23 \
    --gold-name --mqm-col1 \
    --new-metrics-path data/metrics_info/wmt23/out_paths.tsv > data/rankings/wmt23/zh-en/ranking.txt
```

</details>

<details>
<summary><strong>WMT24</strong> (click to expand)</summary>


<br>

#### 🌍 Language Pair: `en-es`

```bash
python scripts/run_mt_meta_eval.py \
    --wmt-year wmt24 \
    --lp en-es \
    --gold-name --mqm > data/rankings/wmt24/en-es/ranking.txt
```

</details>

## Cite this work
This work has been published at [ACL 2025 (Main Conference)](https://2025.aclweb.org/program/main_papers/). If you use any part, please consider citing our paper as follows:

```bibtex
@misc{proietti2025machinetranslationevaluationachieved,
      title={Has Machine Translation Evaluation Achieved Human Parity? The Human Reference and the Limits of Progress}, 
      author={Lorenzo Proietti and Stefano Perrella and Roberto Navigli},
      year={2025},
      eprint={2506.19571},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.19571}, 
}
```

## License
This work is licensed under [Creative Commons Attribution-ShareAlike-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
