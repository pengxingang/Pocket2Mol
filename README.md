# Pocket2Mol: Efficient Molecular Sampling Based on 3D Protein Pockets

[Pocket2Mol](https://arxiv.org/abs/2205.07249) used equivariant graph neural networks to improve efficiency and molecule quality of [previous structure-based drug design model](https://arxiv.org/abs/2203.10446).

<img src="./assets/model.jpg" alt="model"  width="70%"/>


🚧**Still in progress ...**


## Installation
### Dependency
The code has been tested in the following environment:
Package  | Version
--- | ---
Python | 3.8.12
PyTorch | 1.10.1
CUDA | 11.3.1
PyTorch Geometric | **1.7.2**
RDKit | 2022.09.5
<!-- OpenBabel | 3.1.0
BioPython | 1.79 -->
NOTE: Current implementation relies on PyTorch Geometric (PyG) < 2.0.0. We will fix compatability issues for the latest PyG version in the future.
### Install via conda yaml file (cuda 11.3)
```bash
conda env create -f env_cuda113.yml
conda activate Pocket2Mol
```

### Manually installation

For sampling only:
``` bash
conda create -n Pocket2Mol python=3.8
conda activate Pocket2Mol

conda install pytorch==1.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.10.1+cu113.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.10.1+cu113.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install torch-geometric==1.7.2

conda install -c conda-forge rdkit
conda install pyyaml easydict python-lmdb -c conda-forge
```

For training, we recommend to install [`apex` ](https://github.com/NVIDIA/apex) for lower gpu memory usage. If  so, change the value of `train/use_apex` in the `configs/train.yml` file.


## Datasets

Please refer to [`README.md`](./data/README.md) in the `data` folder.

## Sampling

### Sampling for pockets in the testset
To sample molecules for the i-th pocket in the testset, please first download the trained models following [`README.md`](./ckpt/README.md) in the `ckpt` folder. 
Then, run the following command:
```bash
python sample.py --data_id {i} --outdir ./outputs  # Replace {i} with the index of the data. i should be between 0 and 99 for the testset.
```
We recommend to specify the GPU device number and restrict the cpu cores using command like:
```bash
CUDA_VISIBLE_DIVICES=0  taskset -c 0 python sample.py --data_id 0 --outdir ./outputs
```
### Sampling for PDB pockets 
TODO

## Training
```
python train.py
```

## Citation
```
@inproceedings{peng2022pocket2mol,
  title={Pocket2Mol: Efficient Molecular Sampling Based on 3D Protein Pockets},
  author={Xingang Peng and Shitong Luo and Jiaqi Guan and Qi Xie and Jian Peng and Jianzhu Ma},
  booktitle={International Conference on Machine Learning},
  year={2022}
}
```
