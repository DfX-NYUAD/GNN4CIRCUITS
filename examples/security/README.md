## Security Training

This use-case reproduces the node-level classification experiments described in Section 5.2 of the [GNN4IC paper](https://arxiv.org/abs/2201.06848), focused on hardware reverse engineering.

The training pipeline uses graph representations of gate-level netlists. Each graph corresponds to an integrated circuit (IC) design, and the goal is to classify the type of gate or recover functional properties from the structure — a task relevant to reverse engineering.

The dataset used in this experiment is adapted from [Alrahis et al. (2021)](https://ieeexplore.ieee.org/document/9530566), originally released as part of the [GNN-RE project](https://github.com/DfX-NYUAD/GNN-RE). The version in this repository under `examples/security/` has been preprocessed using the GNN4CIRCUITS parser - the steps can be found under `examples/matrix_format/` .

#### Example:

```bash
python GNN4CIRCUITS.py train -class node -task classification -hdim 256 -n_layers 5 -epochs 2000 -input examples/security
```
This command trains a GNN model on node-level reverse engineering tasks using the security dataset.

#### Input

The input directory (e.g., examples/security/) must contain:

- node_features.csv: Numerical features for each node

- graph_edges.csv: Edge list describing graph structure

- graph_properties.csv: Graph-level properties such as graph IDs and labels

#### Output
The training script will:

- Train a GNN model (e.g., GCN, GIN, or PNA) on the selected dataset

- Log training and validation accuracy/loss

- Save performance metrics and model checkpoints if specified in the script
