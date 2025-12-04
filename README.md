# Omics Project (2025)

## Setup
```bash
# create env (conda)
conda create --name omics python=3.11 -y
conda activate omics
conda install -y jupyter jupyterlab ipykernel
conda install -c conda-forge numpy scikit-learn matplotlib pandas tqdm scipy seaborn umap-learn -y
