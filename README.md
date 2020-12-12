# NewGS
Predicting drug-target affinity based on recurrent neural networks and graph convolutional neural networks

# Source
- README.md: this file.
- data/davis/folds/test_fold_setting1.txt,train_fold_setting1.txt; data/davis/Y,ligands_can.txt,proteins.txt data/kiba/folds/test_fold_setting1.txt,train_fold_setting1.txt; data/kiba/Y,ligands_can.txt,proteins.txt Dataset
- embed : Dictionaries
- create_data.py: create data in pytorch format
- utils.py: include TestbedDataset used by create_data.py to create data, and performance measures.
- training.py: train a  model.

# Running
## Install Python libraries needed
Create a conda virtual environment and intall libs
- Install pytorch
- Install pytorch_geometric following instruction at https://github.com/rusty1s/pytorch_geometric
- Install rdkit: conda install -y -c conda-forge rdkit
- Install tensorflow : pip install tensorflow-gpu==1.15
- Install keras: pip install keras ==2.3.1
- Install transformers pip install transformers
## Create data files
```python
python create_data.py 
```
It will create four .pt files in the data/processed folder
## training
```python
python training.py 0 0
```
where the first argument is for the index of the datasets, 0/1 for 'davis' or 'kiba', respectively; the second argument is for the index of the models, 0/1/2/3 for GINConvNet, GATNet, GAT_GCN, or GCNNet
This returns result.csv, containing the performance of the proposed models on the two datasets. The measures include rmse, mse, pearson, spearman, and ci.


