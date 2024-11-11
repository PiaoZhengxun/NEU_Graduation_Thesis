# LB-GLAT

This is the code repository associated with the paper titled "LB-GLAT: Long-Term Bi-Graph Layer Attention Convolutional Network for Anti-Money Laundering in Transactional Blockchain." We present the PyTorch implementation of LB-GLAT, along with the code and the results obtained from ablation and comparison experiments conducted in this paper.

In our research, we have a focus on money laundering detection in UTXO-based blockchains. The dataset used for the experiments is the [Elliptic dataset](https://www.elliptic.co/media-center/elliptic-releases-bitcoin-transactions-data)，which is the open-source dataset of illicit Bitcoin activities.

## Overview

The repository is organised as follows:

- `config/` contains the configuration files for dataset and GNN models.
- `data/` contains the Elliptic dataset and the generated .npz files after processing.
- `dataset/` contains the preprocessing files for the Elliptic dataset and the data loading files for all GNN models and machine learning algorithms.
- `error/` contains the custom error class files.
- `models/` contains the implementation of GNN models, including LB-GLAT, GAT, GCN, GraphSAGE, DeeperGCN and their variants are involved in the paper. Other than LB-GLAT, in the names of the models within the files, "Bi" stands for Bi-Graph, "LTLA" represents Long-Term Layer Attention mechanism, and "FC" signifies Fully Connected (classification head).
- `result/` contains the experimental results of the paper recorded by tensorboard.
- `train/` contains the training startup files of the model in the experiments.
- `utils/` contains the tool files composed of frequently used functions.

## Dependencies

The script has been tested running under Python 3.10.10, with the following packages(dependencies) installed:

- `pandas==2.0.3`
- `numpy==1.25.1`
- `torch==2.0.0+cu118` (CUDA 11.8)
- `tensorboard==2.13.0`
- `scikit-learn==1.3.0`
- `scipy==1.11.1`
- `tqdm==4.65.0`

## Parameters

The model parameters are all configured in the configuration files, with some of the key parameters presented below:

- The key parameters of the Bi-Graph module:

  - `gnns_forward_hidden` and `gnns_reverse_hidden`: list

    The size of each GNN hidden layer for the forward and reverse graphs. Both by default `32 32 32 32`.

  - `gnn_forward_layer_num` and `gnn_reverse_layer_num`: int

    The length of GNN hidden layers for the forward and reverse graphs. Both by default `4`.

  - `gnn_do_bn`: bool

    Whether to apply BatchNorm in GNN, default `True`.

  - `gnn_dropout`: double

    The ratio of Dropout in GNN, default `0.5`.

- The key parameters of the LTLA module:

  - `project_hidden`: int

    The projection layer size, typically the same size as Transformer, default `32`.

  - `tsf_dim`: int

    The Transformer size, default `32`.

  - `tsf_mlp_hidden`: int

    The MLP size of the Transformer, typically twice the size of Transformer, default `64`.

  - `tsf_depth`: int

    The number of alternating layers (K) of multi-headed self-attention (MSA) and multi-layer perception (MLP) in Transformer, default `6`.

  - `tsf_heads`: int

    The head number of MSA, default `4`.

  - `tsf_head_dim`: int

    The head size of MSA, default `8`.

  - `tsf_dropout`: double

    The ratio of Dropout in Transaction, default `0.5`.

  - `vit_emb_dropout`: double

    The ratio of Dropout in VIT, default `0.5`.

  - `vit_pool`: string

    either `cls` token pooling or `mean` pooling, default `mean`.

- The key parameters of the classification heads:

  - `linears_hidden`: list

    The size of each hidden layer in classification heads, default `64 32`.

  - `linear_layer_num`: int

    The length of hidden layers in classification heads, default `2`.

  - `linear_do_bn`: bool

    Whether to apply BatchNorm in classification heads, default `True`.

  - `linear_dropout`: double

    The ratio of Dropout in classification heads, default `0.5`.

- The other key parameters:

  - `opt`: string

    The model optimizer, with selectable values `Adam`, `AdamW`, `SGD` and `RMSprop`, default `Adam`.

  - `lr0`: double

    The initial value of learning rate, default `0.01`.

  - `decay_rate`: double

    The decay rate of learning rate, default `1`.

  - `weight_decay`: double

    The weight decay learning rate, default `0.0005`.

  - `epochs`: int

    The number of training epochs, default `200`.

  - `model_folder`: string

    The name of the folder `result/GNN/xxx` where the results of the model training are stored, default `LB-GLAT`.

  - `model_name`: string

    The name of GNN model that you want to train, default `LB-GLAT`.

## Usage

First, unzip the files located in the `data/` folder, including `elliptic_txs_features.7z` and `data_all_time_np_list.7z`. If you wish to modify parameters within the model, you can alter the variable values in the dataset.conf and GNN.conf files. After configuring the parameters in the configuration file, you can simply launch "train_GNN.py." If you prefer to use Tensorboard, you can start "train_GNN_tensorboard.py."

```
python train\GNN\train_GNN.py 
```
```
python train\GNN\train_GNN_tensorboard.py 
```

## Experimental Results

The experimental results from our paper are stored in the `result/` folder. You can run Tensorboard under the path of a specific model folder `result/GNN/xxx/logs/xxx/xxx` to view our experimental outcomes.

```
tensorboard --logdir ./ --bind_all 
```

The folders under the `result/GNN/xxx/logs/` directory are named as `GnnLayerX/` where X represents the number of convolutional layers in GNN. In this paper, it is used to test the degree of over-smoothing. If `_` appears, it indicates that default parameter settings have been modified. The numeric or word following `_` represents the altered value of a specific parameter. However, if you run the "train" file of this program, the generated result file names can be quite lengthy. You can find the specific naming conventions in the "logs_subfolder" and "results_dir" variables within the corresponding "train" file.