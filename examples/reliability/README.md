## Reliability Training

This use-case reproduces the path-level regression experiments from Section 5.2.2 of the GNN4CIRCUITS paper, inspired by the GNN4REL framework [Alrahis et al. (2022)](https://ieeexplore.ieee.org/document/9852805). The goal is to estimate process-variation-induced timing degradation for critical paths in gate-level netlists.

Each graph in this experiment represents a 1-hop subgraph around a timing path in an integrated circuit (IC) netlist. The regression task predicts the degradation of the delay of each path — a critical aspect in designing reliable ICs under process variation.

The dataset is derived from select EPFL benchmarks, where 1,000 worst-slack paths were extracted per design and augmented using Monte-Carlo simulations of process variation. The data, found [here](https://github.com/lilasrahis/GNN4REL), has been preprocessed using the GNN4CIRCUITS pipeline and converted into a format suitable for graph regression.

#### Example:
```bash
python GNN4CIRCUITS.py train -class node -task_type regression -hdim 64 -n_layers 4 -epochs 500 -lr 0.001 -batch_size 32 -input examples/reliability/adder_files4training
```
This command trains a GNN regression model (GCN, GIN, or PNA) on the extracted subgraphs for path-level degradation prediction.

#### Input
The input directory (e.g., examples/security/) must contain:

- node_features.csv: Numerical features for each node

- graph_edges.csv: Edge list describing graph structure

- graph_properties.csv: Graph-level properties such as graph IDs and labels

#### Output
The training script will:

- Train a graph-level regression model to predict degradation values

- Log training and validation losses (MAE)

- Save metrics in output file
