## EDA Training

This use-case reproduces the node-level learning experiments described in Section 5.2 of the [GNN4IC paper](https://arxiv.org/abs/2201.06848), using graph representations of high-level synthesis (HLS) dataflow graphs.

The training pipeline takes as input graph-structured data derived from HLS circuit designs. Each graph corresponds to a dataflow graph (DFG), control-dataflow graph (CDFG), or real-case application, and includes node-level features and multi-label annotations.

The dataset used in this experiment is adapted* from [Wu et al.](https://github.com/lydiawunan/HLS-Perf-Prediction-with-GNNs/tree/main), originally published in their paper on HLS performance prediction. The version provided under `examples/EDA/` has been preprocessed to meet the input requirements of GNN4CIRCUITS.

#### Example:

```bash
python GNN4CIRCUITS.py train -class node -hdim 300 -n_layers 5 -input examples/EDA/dfg
```

To train on different subsets of the dataset, simply replace dfg with cdfg or realcase.

#### Input

The input directory (e.g., examples/EDA/dfg/) must contain:

- node_features.csv: Numerical features for each node

- graph_edges.csv: Edge list describing graph structure

- graph_properties.csv: Graph-level properties such as graph IDs and labels

#### Output
The training script will:

- Train a GNN model (e.g., GCN, GIN, or PNA) on the selected dataset

- Log training and validation accuracy/loss

- Save performance metrics and model checkpoints if specified in the script

*The dataset provided under examples/EDA/ has been adapted from the original JSON-based graph format released by Wu et al. To make it compatible with the GNN4CIRCUITS pipeline, we parsed each JSON graph into a networkx object, extracted relevant node features (bitwidth, m_delay, etc.), and assigned multi-label classification targets based on FPGA resource usage (FF, LUT, DSP). We then converted these graphs into DGL format and exported them as three CSV files: node_features.csv, graph_edges.csv, and graph_properties.csv. This preprocessing ensures that the dataset can be directly used in the GNN4CIRCUITS training and evaluation pipeline for node-level classification tasks.
