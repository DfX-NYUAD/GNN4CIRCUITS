## EDA Experiment

To reproduce the work as in the GNN4IC paper, under section 5.2, you can run the following command:

```python GNN4CIRCUITS.py train -class node -hdim 300 -n_layers 5 -input Experiments/EDA/dfg```

replace "dfg" with "cdfg" and "realcase" to test the rest of the dataset.

This dataset has been taken from [Wu et al.](https://arxiv.org/abs/2201.06848), and the original can be downloaded [here](https://github.com/lydiawunan/HLS-Perf-Prediction-with-GNNs/tree/main). The dataset in this repository has been adapted from the original to fit the input requirements of GNN4CIRCUITS and can be found under Experiments/EDA/.
