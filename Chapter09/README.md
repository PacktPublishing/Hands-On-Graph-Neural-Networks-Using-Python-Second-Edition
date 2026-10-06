# Chapter 9 – Bridging Graph Databases and GNNs for Scalability

Two ways to keep a graph in a database and still train a GNN on it: Neo4j with
the Graph Data Science plugin as a server, and Kùzu as an embedded store. Both
parts run GraphSAGE on OGBN-arxiv, so the comparison is between the storage
layers, not the models.

## Services

Neo4j is started by `docker-compose.yml`:

```bash
cd Chapter09
docker compose up -d
```

Give it about 30 seconds on the first run: it boots and installs the Graph Data
Science plugin. Confirm at http://localhost:7475 (user `neo4j`, password
`password`).

The container is published on the +1 ports — bolt on **7688**, browser on
**7475** — so it does not clash with a Neo4j you may already be running
locally. `run.py` defaults to `bolt://localhost:7688` to match.

Kùzu, used in Part 2, is embedded. It ships as a Python wheel and needs no
container or server.

## Python environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running it

```bash
python run.py                        # runs both Part 1 and Part 2
python figures/generate_figures.py   # regenerates figures
```

Ingestion is idempotent: rerunning the script skips reloading Neo4j if the
paper count is already correct, and a partial load left by an interrupted run
is cleared before it starts.

Part 2 is skipped automatically if `kuzu` is not installed, so Part 1 alone
works with a subset of the dependencies.

## Environment overrides

`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.

## What to expect

Part 1 reaches about 71–72% test accuracy on OGBN-arxiv after 100 epochs of
GraphSAGE training. Part 2 uses the same model but is intentionally capped at
10 epochs to keep the example concise, so its accuracy is lower by design.

## Caveats

- `id()` is deprecated in Neo4j 5 in favor of `elementId()`. `run.py` uses
  `id()` because it matches the integer node id returned by the GDS client.
  See the chapter note for the migration path.
- We pin `kuzu==0.10.0`. Kùzu Inc. archived the project in October 2025;
  0.10.0 is the last release with pre-compiled wheels for Linux, macOS
  (Intel and Apple Silicon, macOS 11.0 and later), and Windows. The API
  used in this chapter is identical in the community-maintained fork
  LadybugDB (github.com/LadybugDB/ladybug) for readers who prefer to
  depend on a maintained project: change `import kuzu` to `import ladybug`
  and the class names accordingly.
- The full Kùzu graph is extracted with `get_as_df()`, which transfers the
  result in bulk. `get_as_torch_geometric()` iterates one row at a time and
  does not finish in reasonable time on 1.16M edges; it is kept for the small
  year-filtered subgraph only.

## Requirements

Covered by the pinned environment in the repository root. This chapter needs:

```
torch>=2.5
torch-geometric>=2.6
ogb>=1.3.6
pandas>=2.0
neo4j>=5.20
graphdatascience>=1.11,<2.0
kuzu==0.10.0
matplotlib>=3.8
numpy>=1.26
```
