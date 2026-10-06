
# Python environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Wait ~30 seconds for Neo4j to boot and install GDS on first run. Confirm at
http://localhost:7474 (user: `neo4j`, password: `password`).

Kùzu (used in Part 2) is embedded. It ships as a Python wheel and needs no
container or server.

## Run

```bash
python run.py                # runs both Part 1 and Part 2
python figures/generate_figures.py   # regenerates figures
```

Ingestion is idempotent: rerunning the script skips reloading Neo4j if the
paper count is already correct.

Part 2 is skipped automatically if `kuzu` is not installed, so Part 1 alone
works with a subset of the dependencies.

## Environment overrides

`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.

## Caveats

- `id()` is deprecated in Neo4j 5 in favor of `elementId()`. `run.py` uses
  `id()` because it matches the integer node id returned by the GDS client.
  See the chapter note for the migration path.
- We pin `kuzu==0.10.0`. Kùzu Inc. archived the project in October 2025;
  0.10.0 is the last release with pre-compiled wheels for Linux, macOS
  (Intel and Apple Silicon, macOS 11.0 and later), and Windows. The API
  used in this chapter is identical in the community-maintained fork
  LadybugDB (github.com/LadybugDB/ladybug) for readers who prefer to
  depend on a maintained project.
- Part 1 reaches about 71–72% test accuracy on OGBN-arxiv after 100 epochs
  of GraphSAGE training. Part 2 uses the same model but is intentionally
  capped at 10 epochs to keep the example concise.