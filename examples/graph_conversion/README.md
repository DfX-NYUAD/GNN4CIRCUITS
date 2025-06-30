## Graph conversion

This use-case converts gate-level netlist designs into graph representations, which can be used for downstream tasks such as machine learning model training.

The graph conversion pipeline parses standard digital hardware designs and outputs graph structures with metadata including gate types, primary inputs/outputs, and hierarchical connections.

The input Verilog files used in this use case come from open-source benchmark suites:
- [ISCAS-85](https://ddd.fit.cvut.cz/www/prj/Benchmarks/ISCAS85/)
- [ITC-99](https://github.com/ccsl-uaegean/ITC99-RTL-Verilog)

#### Example:

```bash
python GNN4CIRCUITS.py parse -ver Original_Designs/ -hw GL -class graph -lib NangateOpenCellLibrary.v -id -od -gt -pi -po
```

This will create a directory called files4training/ which contains 3 files: node_features.csv, graph_edges.csv and graph_properties.csv. You may use this directory as input for the training and evaluation step of the pipeline.
