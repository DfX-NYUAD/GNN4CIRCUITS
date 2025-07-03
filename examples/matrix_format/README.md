## Matrix Format Parsing

This use-case demonstrates how to parse hardware design graphs represented in matrix format into a format compatible with the GNN4CIRCUITS pipeline. It is useful for datasets where the graph structure and features are pre-encoded in plain `.txt` files.

The dataset provided in `examples/matrix_format/` is extracted data taken from [Alrahis et al.](https://ieeexplore.ieee.org/document/9530566), originally released as part of the [GNN-RE project](https://github.com/DfX-NYUAD/GNN-RE). 

#### Example:

```bash
python GNN4CIRCUITS.py parse_txt -path examples/matrix_format
```
This command parses the txt files and generates a graph object suitable for GNN-based training and evaluation tasks such as reverse engineering.

#### Input
The input directory (e.g., examples/matrix_format/) must contain the following files:

- row.txt: Source node indices for edges
- col.txt: Destination node indices for edges
- feat.txt: Tab-separated node feature vectors (one row per node)
- label.txt: Integer class labels for each node
- cell.txt: Maps each node to its original cell name and source file


#### Output
The parsing script will generate csv files needed for graph representation:
- node_features.csv: Numerical features for each node
- graph_edges.csv: Edge list describing graph structure
- graph_properties.csv: Graph-level properties such as graph IDs and labels

This use-case showcases how GNN4CIRCUITS supports flexible input formats beyond Verilog.
