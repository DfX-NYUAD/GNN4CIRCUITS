## Security Experiment

To reproduce the work as in the GNN4IC paper, under section 5.2, you can run the following command:

```python GNN4CIRCUITS.py train -class node -hdim 256 -n_layers 5 -epochs 2000 -input examples/security```

This dataset has been taken from [Alrahis et al.](https://ieeexplore.ieee.org/document/9530566), and the original can be downloaded [here](https://github.com/DfX-NYUAD/GNN-RE?tab=readme-ov-file#Citation-&-Acknowledgement). The dataset in this repository has been adapted from the original to fit the input requirements of GNN4CIRCUITS and can be found under Experiments/security/.
