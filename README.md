# Less but More: Linear Adaptive Graph Learning Empowering Spatiotemporal Forecasting (NIPS 2025 submission paper 425)
This is the official repository of our NIPS 2025 submission paper 425. The effectiveness of Spatiotemporal Graph Convolutional Networks (STGCNs) critically hinges on the quality of the underlying graph topology. While end-to-end adaptive graph learning methods have demonstrated promising results in capturing latent spatiotemporal dependencies, they often suffer from high computational complexity and limited expressive capacity. In this paper, we propose MAGE for efficient spatiotemporal forecasting. We first conduct a theoretical analysis demonstrating that the ReLU activation function employed in existing methods amplifies edge-level noise during graph topology learning, thereby compromising the fidelity of the learned graph structures. To enhance model expressiveness, we introduce a sparse yet balanced mixture-of-experts strategy, where each expert perceives the unique underlying graph through kernel-based functions and operates with linear complexity relative to the number of nodes. The sparsity mechanism ensures that each node interacts exclusively with compatible experts, while the balancing mechanism promotes uniform activation across all experts, enabling diverse and adaptive graph representations. Furthermore, we theoretically establish that a single graph convolution using the learned graph in MAGE is mathematically equivalent to multiple convolutional steps under conventional graphs. We evaluate MAGE against 14 state-of-the-art baselines on 17 real-world spatiotemporal datasets. MAGE achieves SOTA performance on 94% (48/51) of the evaluation metrics. Notably, on the SD dataset, MAGE achieves an impressive 5.15% performance improvement, while also improving memory efficiency by 10X and training efficiency by 20X.

<br>

We show the pseudocode for the main algorithms, including <b>the Fréchet embedding process</b>, <b>STONE forward process</b>, and <b>the training process of STONE</b>. We will add it to the new version for a clear presentation.

<img src='img/Spatial Fréchet Embedding Layer.png' width='300px' alt='The algorithm of Spatial Fréchet Embedding Layer'>

<img src='img/optimization flow.png' width='300px' alt='Optimization flow of STONE during training'>

<img src='img/STONE.png' width='300px' alt='Framework of STONE'>

## 1. Introduction about the datasets
### 1.1 Generating the SD and GBA sub-datasets from CA dataset
In the experiments of our paper, we used SD and GBA datasets with years from 2019 to 2021, which were generated from CA dataset, followed by [LargeST](https://github.com/liuxu77/LargeST/blob/main). For example, you can download CA dataset from the provided [link](https://www.kaggle.com/datasets/liuxu77/largest) and please place the downloaded `archive.zip` file in the `data/ca` folder and unzip the file. 

First of all, you should go through a jupyter notebook `process_ca_his.ipynb` in the folder `data/ca` to process and generate a cleaned version of the flow data. Then, please go through all the cells in the provided jupyter notebooks `generate_sd_dataset.ipynb` in the folder `data/sd` and `generate_gla_dataset.ipynb` in the folder `data/gla` respectively. Finally use the commands below to generate traffic flow data for our experiments. 
```
python data/generate_data_for_training.py --dataset sd_gba --years 2019_2020_2021
```
Moreover, you can also generate the other years of data, as well as the two additional remaining subdatasets. 

### 1.2 Generating the additional PM2.5 Knowair dataset
We implement extra experiments on [Knowair](https://github.com/shuowang-ai/PM2.5-GNN). For example, you can download Knowair dataset from the provided [link](https://drive.google.com/file/d/1R6hS5VAgjJQ_wu8i5qoLjIxY0BG7RD1L/view) and please place the downloaded `Knowair.npy` file in the `Knowair` folder and complete the files in the `Knowair/data` folder.

<br>

## 2. Environmental Requirments
The experiment requires the same environment as [LargeST](https://github.com/liuxu77/LargeST/blob/main), and need to add the libraries mentioned in the requirements in [Knowair](https://github.com/shuowang-ai/PM2.5-GNN).

<br>

## 3. Model Running
To run STONE on <b>LargeST</b>, for example, you may execute this command in the terminal:
```
bash experiments/stone/run.sh
```
or directly execute the Python file in the terminal:
```
python experiments/stone/main.py --device cuda:0 --dataset SD --years 2019 --model_name stone --seed 0 --bs 64
```
To run STONE on <b>Knowair</b>, you may directly execute the Pyhon file in the terminal:
```
python Knowair/train.py
```

