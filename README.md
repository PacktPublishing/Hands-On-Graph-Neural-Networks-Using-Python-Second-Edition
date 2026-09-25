# Hands-On Graph Neural Networks Using Python — Second Edition

Companion code for *Hands-On Graph Neural Networks Using Python, Second Edition*,
published by Packt.

Each chapter is a self-contained folder with a script you can run start to
finish, a notebook generated from that script, and the code that draws the
chapter's figures.

## Chapters

| | Chapter | Runs |
| --- | --- | --- |
| 02 | _(title to confirm)_ | figures only |
| 03 | Creating Node Representations with DeepWalk | `run.py` |
| 04 | Improving Embeddings with Biased Random Walks in Node2Vec | `run.py` |
| 05 | Including Node Features with Vanilla Neural Networks | `run.py` |
| 06 | Introducing Graph Convolutional Networks | `run.py` |
| 07 | Graph Attention Networks | `run.py` |
| 08 | Scaling Up Graph Neural Networks with GraphSAGE | `run.py` |
| 09 | Bridging Graph Databases and GNNs for Scalability | `run.py` + Neo4j |
| 10 | Graph Transformers: Attention Beyond Local Neighborhoods | `run.py` |
| 11 | Defining Expressiveness for Graph Classification | `run.py` |
| 12 | Predicting Links with Graph Neural Networks | `run.py` |
| 13 | Learning from Heterogeneous Graphs | `run.py` |
| 14 | Temporal Graph Neural Networks | `run.py` |
| 15 | Explaining Graph Neural Networks | `run.py` |
| 16 | Forecasting Traffic Using A3T-GCN | `run.py` |
| 17 | Detecting Anomalies Using Heterogeneous GNNs | `run.py` |
| 18 | Building a Recommender System Using LightGCN | `run.py` |
| 19 | Large Language Models Meet Graph Neural Networks | `run.py` + model download |
| 20 | Unlocking the Potential of Graph Foundation Models | figures only |

Every chapter folder holds:

- `run.py` — the chapter's code, start to finish
- `ChapterXX.ipynb` — the same code as a notebook, generated from `run.py`
- `requirements.txt` — what that chapter needs
- `figures/generate_figures.py` — the code behind the chapter's figures
- `figures/INDEX.md` — which file is which figure, and what produces it

## Setup

Python 3.11 is what the book was verified against.

```bash
git clone https://github.com/PacktPublishing/Hands-On-Graph-Neural-Networks-Using-Python-Second-Edition.git
cd Hands-On-Graph-Neural-Networks-Using-Python-Second-Edition

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The root `requirements.txt` pins one set of versions that every chapter runs
against, so you install once and work through the whole book. Each chapter
also states its own minimum versions, should you prefer separate environments.

**On macOS**, run `/Applications/Python 3.11/Install Certificates.command`
once. The python.org build ships without CA certificates, and Chapters 14 and
16 download their data with `urllib`, which then fails with
`CERTIFICATE_VERIFY_FAILED`.

## Running a chapter

Scripts read and write paths relative to their own chapter, so run them from
inside it:

```bash
cd Chapter06
python run.py                          # the chapter, start to finish
python figures/generate_figures.py     # redraw its figures
```

The datasets are not in this repository: `run.py` downloads what it needs on
first run. Expect that to take a while, and a few GB of disk, for the larger
chapters. Chapter 09 additionally needs Neo4j, which its own README explains;
Chapter 19 downloads a language model.

Times vary with hardware. Most chapters finish in a few minutes on a laptop;
Chapters 08, 10, 14, 16 and 17 train for considerably longer.

## Tools

| Command | What it does |
| --- | --- |
| `python tools/run_all.py` | runs every chapter and reports what passed |
| `python tools/run_all.py --figures` | redraws every figure instead |
| `python tools/make_notebooks.py` | regenerates the notebooks from `run.py` |
| `python tools/make_notebooks.py --check` | reports notebooks that no longer match |
| `python tools/make_figure_index.py` | rewrites each `figures/INDEX.md` |
| `python tools/check_requirements.py` | checks the chapters against the pinned versions |
| `tools/backup.sh <dir>` | full backup of the working tree plus a git bundle |

## License

See [LICENSE](LICENSE).
