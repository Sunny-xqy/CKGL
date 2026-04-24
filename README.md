# CKGL
A Coalition-based Knowledge Graph Learning Method for Actual Controller Disclosure

## 1. Install Dependencies

```bash
pip install torch numpy networkx joblib pyyaml tqdm
```

---

## 2. Prepare Data

You need two files:

### (1) Dynamic Graph

File: `edges_dynamic.txt`

Format:

```
source_node,target_node,relation_type,timestamp
```

Example:

```
1,2,shareholding,2019
2,3,kinship,2020
3,4,-shareholding,2021
```

Notes:

* Positive relation → edge added
* Negative relation → edge removed

---

### (2) History Node Embeddings

File: `pretrained_embeddings.npy`

* Format: NumPy array
* Shape: `[num_nodes, embedding_dim]`

---

## 3. Run

Main entry:

```bash
python run_update.py
```

Running `run_update.py` executes:

```
1. Load dynamic graph
2. Multi-relational aggregation
3. Metapath-based random walk
4. Skip-gram training
5. Control prediction (CoNN)
6. Relation-aware update
7. Output embeddings and predictions
```

---

## 4. Optional Commands

Generate dynamic graph:

```bash
python dynamic_relation.py
```

Train update module only:

```bash
python train_RU.py
```

---

## 5. Output

Output path is defined in:

```
shareholding.yaml -> output_path
```

Includes:

* Updated node embeddings
* Control prediction results

---

## 6. Common Issues

### CUDA error

```yaml
device: cpu
```

### File not found

Check paths in:

```
shareholding.yaml
```

### Slow training

Reduce:

```yaml
batch_size
time_buckets
epochs
```

---

## 7. One-line Run

```bash
python run_update.py
```

