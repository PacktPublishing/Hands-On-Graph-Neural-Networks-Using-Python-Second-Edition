"""
Chapter 9 – Bridging Graph Databases and GNNs for Scalability
Hands-On Graph Neural Networks Using Python (2nd Edition)

Requirements:
    pip install torch torch-geometric ogb pandas
    pip install neo4j graphdatascience
    pip install kuzu==0.10.0             # for Part 2 (optional)

Prerequisites (services):
    Neo4j 5 with the Graph Data Science plugin, started by docker-compose.yml:

        docker compose up -d

    It is published on the +1 ports, so it does not clash with a Neo4j you
    may already be running: bolt on 7688 and the browser on
    http://localhost:7475. Override NEO4J_URI to point somewhere else.

    Kùzu is embedded and needs no container.

New in this chapter (no first-edition counterpart):
  - Part 1: Neo4j + Graph Data Science + GraphSAGE + Cypher writeback
  - Part 2: Kùzu as embedded storage + Cypher-based PyG extraction

Fixes applied after the technical review:
  1. PyTorch >= 2.6 changed torch.load to weights_only=True by default, and
     the cached OGB pickle contains PyG classes that are not on the safe
     list, so loading OGBN-arxiv raised UnpicklingError. The PyG classes are
     registered with torch.serialization.add_safe_globals before loading.
  2. Topology is streamed with gds.graph.relationships.stream, which returns
     a RelationshipsDataFrame exposing by_rel_type(). The gds.beta namespace
     used before is deprecated since GDS 2.5.
  3. The random-walk-with-restarts sample is included, with an
     exists-then-drop guard and concurrency=1, which randomSeed requires.
  4. A partial Neo4j load left by an interrupted run is cleared before
     ingestion, so rerunning never trips the uniqueness constraint.
  5. Kuzu: the full graph is extracted with get_as_df(), which transfers
     the result in bulk. get_as_torch_geometric() iterates the result one
     row at a time in Python and does not finish in reasonable time on
     1.16M edges; it is kept for the small year-filtered subgraph only.
     The year column is now exported, since the filtered query needs it.
  6. Training time is printed for both parts.

Design notes:
  - Ingestion is idempotent: rerunning the script does not reload the
    graph if the paper count is already correct.
  - GraphSAGE outputs raw logits and the loss is CrossEntropyLoss.
    No log_softmax in the model.
  - Predictions are written back to Neo4j via id() rather than
    elementId(): id() is deprecated in Neo4j 5, but matches the
    integer node id that the GDS client currently returns as `nodeId`,
    keeping the mapping trivial for a self-contained example.
  - Part 2 pins kuzu==0.10.0. Kùzu Inc. archived the project in
    October 2025 after an acquisition. This release provides
    pre-compiled wheels for Linux, macOS (Intel and Apple Silicon),
    and Windows. Readers building long-term projects can migrate to
    LadybugDB, the community-maintained fork with an identical API,
    by changing `import kuzu` to `import ladybug` and the class names
    accordingly.
"""

import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

NEO4J_URI      = os.environ.get('NEO4J_URI',      'bolt://localhost:7688')
NEO4J_USER     = os.environ.get('NEO4J_USER',     'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'password')
NEO4J_IMPORT_DIR = os.path.abspath('./neo4j_import')
os.makedirs(NEO4J_IMPORT_DIR, exist_ok=True)


# =============================================================================
# PART 1 – GraphSAGE on OGBN-arxiv via Neo4j + GDS
# =============================================================================

print("\n" + "=" * 60)
print("PART 1 – GraphSAGE on OGBN-arxiv via Neo4j + GDS")
print("=" * 60)

from ogb.nodeproppred import PygNodePropPredDataset
from neo4j import GraphDatabase
from graphdatascience import GraphDataScience
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected

# ── Download and inspect the dataset ─────────────────────────────────────────

# PyTorch >= 2.6 loads with weights_only=True by default, and the cached OGB
# pickle contains PyG classes that are not on the default safe list.
if hasattr(torch.serialization, 'add_safe_globals'):
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    torch.serialization.add_safe_globals(
        [DataEdgeAttr, DataTensorAttr, GlobalStorage])

dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data')
data_ogb = dataset[0]
print(f"\nDataset: {data_ogb}")

# ── Convert to CSV for Neo4j ingestion ───────────────────────────────────────

papers_csv    = f"{NEO4J_IMPORT_DIR}/papers.csv"
citations_csv = f"{NEO4J_IMPORT_DIR}/citations.csv"

if os.path.exists(papers_csv) and os.path.exists(citations_csv):
    print("\nCSVs already present; skipping regeneration.")
else:
    print("\nWriting papers.csv and citations.csv …")
    papers = pd.DataFrame({
        'id':      range(data_ogb.num_nodes),
        'year':    data_ogb.node_year.squeeze().tolist(),
        'subject': data_ogb.y.squeeze().tolist(),
    })
    papers['features'] = [
        ';'.join(f'{v:.6f}' for v in row) for row in data_ogb.x.tolist()
    ]
    papers.to_csv(papers_csv, index=False)

    src, dst = data_ogb.edge_index
    pd.DataFrame({'src': src.tolist(), 'dst': dst.tolist()}) \
      .to_csv(citations_csv, index=False)

# ── Load into Neo4j (idempotent) ─────────────────────────────────────────────

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

with driver.session() as session:
    n_papers = session.run("MATCH (p:Paper) RETURN count(p) AS n").single()['n']

if n_papers == data_ogb.num_nodes:
    print(f"\nNeo4j already has {n_papers} Paper nodes — skipping ingestion.")
else:
    print(f"\nIngesting into Neo4j (current node count: {n_papers}) …")
    with driver.session() as session:
        # A run interrupted mid-ingestion leaves a partial graph behind, and
        # reloading on top of it would violate the uniqueness constraint.
        if n_papers > 0:
            print("  clearing a partial load first …")
            session.run("""
                MATCH (p:Paper)
                CALL { WITH p DETACH DELETE p } IN TRANSACTIONS OF 10000 ROWS
            """)
        session.run(
            "CREATE CONSTRAINT paper_id IF NOT EXISTS "
            "FOR (p:Paper) REQUIRE p.id IS UNIQUE")

        t0 = time.time()
        session.run("""
            LOAD CSV WITH HEADERS FROM 'file:///papers.csv' AS row
            CALL {
                WITH row
                CREATE (:Paper {
                    id: toInteger(row.id),
                    year: toInteger(row.year),
                    subject: toInteger(row.subject),
                    features: [v IN split(row.features, ';') | toFloat(v)]
                })
            } IN TRANSACTIONS OF 10000 ROWS
        """)
        print(f"  papers loaded in {time.time()-t0:.1f}s")

        t0 = time.time()
        session.run("""
            LOAD CSV WITH HEADERS FROM 'file:///citations.csv' AS row
            CALL {
                WITH row
                MATCH (src:Paper {id: toInteger(row.src)})
                MATCH (dst:Paper {id: toInteger(row.dst)})
                CREATE (src)-[:CITES]->(dst)
            } IN TRANSACTIONS OF 10000 ROWS
        """)
        print(f"  citations loaded in {time.time()-t0:.1f}s")

with driver.session() as session:
    result = session.run("""
        MATCH (p:Paper)
        OPTIONAL MATCH (p)-[r:CITES]->()
        RETURN count(DISTINCT p) AS num_papers, count(r) AS num_citations
    """).single()
    print(f"Neo4j has {result['num_papers']} papers "
          f"and {result['num_citations']} citations")

# ── Project the graph into GDS ───────────────────────────────────────────────

gds = GraphDataScience(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

if gds.graph.exists('arxiv')['exists']:
    gds.graph.drop('arxiv')

G, _ = gds.graph.project(
    'arxiv',
    {'Paper': {'properties': ['id', 'features', 'subject', 'year']}},
    {'CITES': {'orientation': 'NATURAL'}},
)
print(f"\nProjected {G.node_count()} nodes "
      f"and {G.relationship_count()} relationships into GDS")

# ── Stream topology and node properties client-side ─────────────────────────

# gds.graph.relationships.stream returns a RelationshipsDataFrame, whose
# by_rel_type() maps each relationship type to [[sources], [targets]]:
#   {'CITES': [[42, 7, ...], [1053, 42, ...]]}
# relationshipProperties.stream streams property values instead, and its
# result has no by_rel_type().
topology_df    = gds.graph.relationships.stream(G)
edge_index_raw = topology_df.by_rel_type()['CITES']
print(f"Streamed {len(edge_index_raw[0])} edges")

node_props = gds.graph.nodeProperties.stream(
    G, ['id', 'features', 'subject', 'year'], separate_property_columns=True)
print(f"Streamed properties for {len(node_props)} nodes")

# ── Remap GDS ids to contiguous PyG indices ─────────────────────────────────

gds_ids    = node_props['nodeId'].tolist()
gds_to_pyg = {gid: i for i, gid in enumerate(gds_ids)}

src        = [gds_to_pyg[i] for i in edge_index_raw[0]]
dst        = [gds_to_pyg[i] for i in edge_index_raw[1]]
edge_index = torch.tensor([src, dst], dtype=torch.long)
# Convert to undirected. OGB provides edge_index directionally (paper A cites B),
# but node classification on citation graphs treats connections as symmetric:
# a paper is influenced by both what it cites and what cites it. This matches
# the OGBN-arxiv leaderboard convention and is essential for the model to
# reach the ~70% test accuracy typical of GraphSAGE on this dataset.
edge_index = to_undirected(edge_index)
print(f"Edge index after to_undirected: {edge_index.shape[1]} edges")

x = torch.tensor(node_props['features'].tolist(), dtype=torch.float)
y = torch.tensor(node_props['subject'].tolist(), dtype=torch.long)

data = Data(x=x, y=y, edge_index=edge_index)

# ── Attach the OGB train/valid/test splits ───────────────────────────────────

# Critical: Neo4j+GDS reordered the nodes relative to OGB's original array.
# The 'id' property carries the OGB index of each paper, so we look up each
# PyG-indexed row's OGB id and mark it as train/valid/test accordingly.
ogb_ids = node_props['id'].astype(int).tolist()
split   = dataset.get_idx_split()
for name in ('train', 'valid', 'test'):
    split_set = set(split[name].tolist())
    mask      = torch.tensor([oid in split_set for oid in ogb_ids],
                              dtype=torch.bool)
    setattr(data, f'{name}_mask', mask)

print(f"\nData: {data}")
print(f"Train: {int(data.train_mask.sum())}, "
      f"Valid: {int(data.valid_mask.sum())}, "
      f"Test: {int(data.test_mask.sum())}")


# ── GraphSAGE ────────────────────────────────────────────────────────────────

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1   = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2   = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Standard OGB baseline pattern: conv -> BN -> relu -> dropout,
        # repeated across hidden layers, then a plain conv for the logits.
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # Raw logits — CrossEntropyLoss expects logits.
        return self.conv3(x, edge_index)


num_classes = int(data.y.max().item()) + 1
model       = GraphSAGE(data.x.size(1), 256, num_classes).to(device)
optimizer   = torch.optim.Adam(model.parameters(), lr=0.01)
criterion   = torch.nn.CrossEntropyLoss()
print(f"\n{model}")

train_loader = NeighborLoader(
    data, num_neighbors=[15, 10, 5], batch_size=1024,
    input_nodes=data.train_mask, shuffle=True)

print("\nTraining GraphSAGE (100 epochs) …")
n_train = int(data.train_mask.sum())
t_train = time.time()
for epoch in range(1, 101):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index)[:batch.batch_size]
        loss = criterion(out, batch.y[:batch.batch_size])
        loss.backward(); optimizer.step()
        total_loss += float(loss) * batch.batch_size
    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:03d}/100 | Loss: {total_loss/n_train:.4f}")
print(f"Training time: {time.time()-t_train:.0f}s on {device}")

model.eval()
with torch.no_grad():
    # Full-graph evaluation: GraphSAGE is inductive, so the same model
    # applies to every node in the graph without retraining.
    out  = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=1).cpu()

for name in ('train', 'valid', 'test'):
    mask = getattr(data, f'{name}_mask')
    acc  = (pred[mask] == data.y[mask]).float().mean().item()
    print(f"{name.capitalize()} accuracy: {acc:.4f}")

# ── Write predictions back to Neo4j ─────────────────────────────────────────

print("\nWriting predictions back to Neo4j …")
predictions = [
    {'gds_id': int(gid), 'predicted': int(p)}
    for gid, p in zip(gds_ids, pred.tolist())
]

write_query = """
    UNWIND $rows AS row
    MATCH (p:Paper) WHERE id(p) = row.gds_id
    SET p.predicted_subject = row.predicted
"""

with driver.session() as session:
    for i in range(0, len(predictions), 10000):
        session.run(write_query, rows=predictions[i:i+10000])
print(f"  wrote {len(predictions)} predictions")

# ── Mixed query: traversal + predicted labels ────────────────────────────────

print("\nMixed query — class-0 papers citing >=3 class-16 papers:")
with driver.session() as session:
    rows = session.run("""
        MATCH (p:Paper {predicted_subject: 0})
              -[:CITES]->(c:Paper {predicted_subject: 16})
        WITH p, count(c) AS n
        WHERE n >= 3
        RETURN p.id AS paper_id, n AS citations
        ORDER BY n DESC LIMIT 5
    """).data()
if rows:
    for row in rows:
        print(f"  paper {row['paper_id']}: {row['citations']} citations")
else:
    print("  (no papers match this specific pattern)")

# ── When the graph does not fit: random walk with restarts ──────────────────

# randomSeed is only accepted together with concurrency=1, and the sample is
# a named graph in the catalog, so it needs the same exists-then-drop guard
# as the main projection to make the cell safe to rerun.
if gds.graph.exists('arxiv_sample')['exists']:
    gds.graph.drop('arxiv_sample')

G_sample, sample_stats = gds.graph.sample.rwr(
    'arxiv_sample',
    G,
    samplingRatio=0.1,
    randomSeed=42,
    concurrency=1,
)
print(f"\nSampled {G_sample.node_count()} nodes "
      f"and {G_sample.relationship_count()} relationships.")

driver.close()


# =============================================================================
# PART 2 – Kùzu as a PyG-ready graph store (OGBN-arxiv)
# =============================================================================
#
# In Part 1 we streamed the OGBN-arxiv graph out of Neo4j+GDS in one shot.
# Neo4j is powerful but heavy: it runs as a server, needs a JVM heap, and is
# overkill for training a single-machine model against a graph that fits on
# a laptop's disk.
#
# Kùzu is an embedded columnar graph database. It runs in the same process
# as our Python script, stores the graph in a single directory on disk, and
# ships a Cypher engine plus a first-class PyTorch Geometric bridge. In
# this section we ingest OGBN-arxiv into a Kùzu database, then use a
# Cypher query to pull out a PyG Data object we can train against.
#
# Compared to Part 1 we exchange server infrastructure for embedded storage,
# and we exchange full-graph streaming for query-level flexibility: any
# subgraph expressible in Cypher can be materialised into a PyG Data object
# on demand.
#
# Kùzu Inc. archived the project in October 2025 following an acquisition.
# We pin kuzu==0.10.0 as the last release with a stable PyG integration
# across every mainstream platform. The community fork LadybugDB
# (github.com/LadybugDB/ladybug) continues active development and is a
# drop-in API replacement for readers building long-term projects.

print("\n" + "=" * 60)
print("PART 2 – Kùzu as a PyG-ready graph store (OGBN-arxiv)")
print("=" * 60)

try:
    import kuzu
except ImportError as e:
    print(f"\nSkipping Part 2 — could not import Kùzu: {e}")
    print("Install with:  pip install kuzu==0.10.0")
else:
    import csv, shutil
    from pathlib import Path

    KUZU_PATH = Path("./arxiv_kuzu")
    CSV_DIR   = Path("./kuzu_import")
    CSV_DIR.mkdir(exist_ok=True)

    # ── Bulk export the OGBN-arxiv graph to CSV ─────────────────────────────
    #
    # Kùzu loads much faster from CSV via COPY than from per-row CREATE.
    # The features column is written as a bracketed list (e.g.
    # "[0.12,0.87,...]") which Kùzu parses as an ARRAY of DOUBLE on ingest.

    nodes_csv = CSV_DIR / "arxiv_nodes.csv"
    edges_csv = CSV_DIR / "arxiv_edges.csv"

    # Regenerate if missing, or if left over from a version without 'year'
    stale = nodes_csv.exists() and \
        open(nodes_csv).readline().strip() != 'id,x,y,year'
    if not nodes_csv.exists() or stale:
        print(f"\nExporting OGBN-arxiv to CSV under {CSV_DIR} …")
        t0 = time.time()
        with open(nodes_csv, 'w') as f:
            w = csv.writer(f)
            w.writerow(['id', 'x', 'y', 'year'])
            x_all    = data_ogb.x.tolist()
            y_all    = data_ogb.y.squeeze().tolist()
            year_all = data_ogb.node_year.squeeze().tolist()
            for i in range(data_ogb.num_nodes):
                w.writerow([i,
                            '[' + ','.join(f'{v:.6f}' for v in x_all[i]) + ']',
                            y_all[i],
                            year_all[i]])
        with open(edges_csv, 'w') as f:
            w = csv.writer(f)
            w.writerow(['from', 'to'])
            src_e, dst_e = data_ogb.edge_index
            for s, d in zip(src_e.tolist(), dst_e.tolist()):
                w.writerow([s, d])
        print(f"  exported in {time.time()-t0:.1f}s")

    # ── Load the graph into Kùzu ────────────────────────────────────────────

    # A clean database each run keeps the schema definition simple. In
    # production you would persist KUZU_PATH between runs.
    if KUZU_PATH.exists():
        shutil.rmtree(KUZU_PATH)

    print(f"\nCreating Kùzu database at {KUZU_PATH} …")
    kz_db   = kuzu.Database(str(KUZU_PATH))
    kz_conn = kuzu.Connection(kz_db)

    feat_dim = data_ogb.x.size(1)
    # DOUBLE[dim] is what Kùzu's PyG converter recognises as a feature vector.
    kz_conn.execute(
        f"CREATE NODE TABLE paper "
        f"(id INT64, x DOUBLE[{feat_dim}], y INT64, year INT64, PRIMARY KEY (id))"
    )
    kz_conn.execute(
        "CREATE REL TABLE cites (FROM paper TO paper)"
    )

    print("Bulk-loading nodes and edges into Kùzu …")
    t0 = time.time()
    kz_conn.execute(f"COPY paper FROM '{nodes_csv.resolve()}' (HEADER=true)")
    kz_conn.execute(f"COPY cites FROM '{edges_csv.resolve()}' (HEADER=true)")
    n_nodes_kz = kz_conn.execute("MATCH (n:paper) RETURN count(n)").get_next()[0]
    n_edges_kz = kz_conn.execute("MATCH ()-[r:cites]->() RETURN count(r)").get_next()[0]
    print(f"  loaded {n_nodes_kz} nodes and {n_edges_kz} edges "
          f"in {time.time()-t0:.1f}s")

    # ── Extract the graph as a PyG Data object via Cypher ───────────────────
    #
    # Two Cypher queries, one for nodes and one for edges, each transferred in
    # bulk with get_as_df(). get_as_torch_geometric() would build the Data
    # object in one call, but it iterates the result one row at a time in
    # Python, which does not finish in reasonable time on 1.16M edges.
    # Node ids are the OGB indices 0..N-1, so ORDER BY p.id makes row i of the
    # tensors correspond to OGB node i and the splits apply directly.

    print("\nExtracting the full arxiv graph from Kùzu via Cypher …")
    t0 = time.time()
    nodes_df = kz_conn.execute(
        "MATCH (p:paper) RETURN p.id AS id, p.x AS x, p.y AS y "
        "ORDER BY p.id"
    ).get_as_df()
    edges_df = kz_conn.execute(
        "MATCH (a:paper)-[:cites]->(b:paper) RETURN a.id AS src, b.id AS dst"
    ).get_as_df()

    kz_data = Data(
        x=torch.tensor(np.stack(nodes_df['x'].to_numpy()), dtype=torch.float),
        y=torch.tensor(nodes_df['y'].to_numpy(), dtype=torch.long),
        edge_index=torch.tensor(
            np.stack([edges_df['src'].to_numpy(), edges_df['dst'].to_numpy()]),
            dtype=torch.long),
    )
    print(f"  extracted in {time.time()-t0:.1f}s: {kz_data}")

    kz_data.edge_index = to_undirected(kz_data.edge_index)
    split_kz = dataset.get_idx_split()
    for name in ('train', 'valid', 'test'):
        mask = torch.zeros(kz_data.num_nodes, dtype=torch.bool)
        mask[split_kz[name]] = True
        setattr(kz_data, f'{name}_mask', mask)
    print(f"Train: {int(kz_data.train_mask.sum())}, "
          f"Valid: {int(kz_data.valid_mask.sum())}, "
          f"Test: {int(kz_data.test_mask.sum())}")

    # ── A filtered subgraph, where get_as_torch_geometric() is the right tool
    #
    # Any subgraph expressible in Cypher can be materialised on demand. On a
    # small result like this one the one-call converter is fast enough.
    # Note that OGBN-arxiv is split by year (train <= 2017, valid 2018,
    # test >= 2019), so this subgraph holds no training nodes: it is shown
    # for extraction, not for training.
    t0 = time.time()
    result_recent = kz_conn.execute(
        "MATCH (a:paper)-[r:cites]->(b:paper) "
        "WHERE a.year >= 2018 AND b.year >= 2018 "
        "RETURN a, r, b"
    )
    recent_data, _, _, _ = result_recent.get_as_torch_geometric()
    print(f"\nRecent subgraph extracted in {time.time()-t0:.1f}s: {recent_data}")

    # ── Train a GraphSAGE on the Kùzu-backed Data ──────────────────────────
    #
    # From here on the code is deliberately the same as Part 1: the Data
    # object came from a different backend, but the model, the sampler, and
    # the training loop are unchanged.

    model_kz     = GraphSAGE(kz_data.x.size(1), 256, num_classes).to(device)
    optimizer_kz = torch.optim.Adam(model_kz.parameters(), lr=0.01)

    train_loader_kz = NeighborLoader(
        kz_data, num_neighbors=[15, 10, 5], batch_size=1024,
        input_nodes=kz_data.train_mask, shuffle=True)

    print("\nTraining GraphSAGE on the Kùzu-backed graph (10 epochs) …")
    n_train_kz = int(kz_data.train_mask.sum())
    t_train = time.time()
    for epoch in range(1, 11):
        model_kz.train()
        total_loss = 0.0
        for batch in train_loader_kz:
            batch = batch.to(device)
            optimizer_kz.zero_grad()
            out  = model_kz(batch.x.float(), batch.edge_index)[:batch.batch_size]
            loss = criterion(out, batch.y[:batch.batch_size].long())
            loss.backward(); optimizer_kz.step()
            total_loss += float(loss) * batch.batch_size
        print(f"  Epoch {epoch:02d}/10 | Loss: {total_loss/n_train_kz:.4f}")
    print(f"Training time: {time.time()-t_train:.0f}s on {device}")

    model_kz.eval()
    with torch.no_grad():
        out_kz  = model_kz(kz_data.x.float().to(device),
                            kz_data.edge_index.to(device))
        pred_kz = out_kz.argmax(dim=1).cpu()
    for name in ('train', 'valid', 'test'):
        mask = getattr(kz_data, f'{name}_mask')
        acc  = (pred_kz[mask] == kz_data.y[mask]).float().mean().item()
        print(f"{name.capitalize()} accuracy: {acc:.4f}")

print("\nDone.")