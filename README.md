# TeG-DRec: Inductive Text-Graph Learning for Unseen Node Scientific Dataset Recommendation

## Requirements

Make sure you have Python 3.8+ installed. Then install the following dependencies:

```
pip install torch torchvision torchaudio
pip install torch-geometric


pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
**where ${TORCH} and ${CUDA} should be replaced by the specific PyTorch and CUDA versions, respectively**

pip install faiss-cpu
pip install pandas tqdm
pip install numpy==1.26.4
```

## How to Run

First, clone this repository.

Enter command below in CLI based on the model available:

For GAT:

```python main.py --encoder gat --lr 5e-3```

For RGCN:

```python main.py --encoder rgcn --lr 5e-3```

For GraphSAGE:

```python main.py --encoder sage --lr 5e-2```



