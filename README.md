# GraphLeak: A realistic dataset to detect and locate leaks in water distribution networks

This repository contains the GraphLeak dataset, a comprehensive dataset designed for locating and identifying leaks in water distribution networks (WDN). The dataset is intended to support researchers in developing and evaluating water leak detection models, particularly those utilizing deep learning techniques.

**Note:** Please refer to the corresponding folder in the folder list above for information about a specific publication. All of them use the same data generation and structure proposed in GraphLeak.

## Abstract

The management of water resources and the reduction of water losses due to leaks are crucial for human life and industrial processes. To improve the efficiency of leak detection algorithms, a realistic dataset with reliable values is essential. GraphLeak is a dataset created through realistic simulations using the EPANET-MATLAB toolkit. It includes various WDN scenarios and topologies, with each node representing a measurement point within the network.

## Index Terms
- Dataset
- Water leak detection
- Deep learning
- EPANET simulation

## Dataset Description
Deep learning algorithms rely on high-quality data for accurate training and evaluation. GraphLeak provides a comprehensive dataset in tabular format, where each column represents a specific variable measured by individual sensors. The dataset includes information on pressure, flow, volume, label, and localization. The simulations are conducted using the EPANET WDN modeling software, and the datasets are exported to CSV (Comma-Separated Values) files.





![WDS_topologie](/WDS_top.png)

### Evaluation

The results obtained by a Multi-layer Perceptron are evaluated by the ain classification metrics of confusion matrix, such as accuracy, precision, reacall and F1-score.

The Mean Absolute Error (MAPE) is used to analyze the error between predictions and the correct values.

# Raw Data Download

All the contents of GraphLeak are public and can be acessed [here](https://drive.google.com/drive/folders/1Q_JQO2OZhejQEd0BMdx0UGcRaDo85ENC?usp=share_link)

# PreProcess python file

### Prerequisites
- Python3
- All the libraries in <code>requirements.txt</code>

### Data generation

From raw Data, generate the dataset by running:

<pre><code> python3 main.py </pre></code>

### Configurations

**Meansurements Content** - You can choose which measure values contain in the dataset
- <code>Pressure: True or False</code>
- <code>Flow: True or False</code>
- <code>Volume: True or False</code>

**Noise** - If you want a Gaussian noise in the data, set noise as True.
- <code>Noise: True</code>

**Noise specification** - If there is noise in the data, specify the configuration bellow:
- <code>mu: 0 </code> mean default
- <code>sigma: 0.1 </code> standard deviation default

**Nodes Normalization** - Set True (recommended) to normalize values between nodes.
- <code>Node_normalization: True</code>

**Data Normalization** - Set True (recommended) to normalize values in the range 0 to 1.
- <code>Data_normalization: True</code>

# License

# Authors
Lucas Roberto Tomazini;
Weliton do Carmo Rodrigues;
Rodrigo Pita Rolle;
Alexandre da Silva Simões;
Esther Luna Colombini;
Eduardo Paciência Godoy;


# Citation 
Please cite one of the following papers if you use this code for your researches:

<pre><code>@article{xx,
  title={GraphLeak: A realistic dataset to detect and locate leaks in water distribution networks},
  author={xx},
  journal={xx},
  volume={xx},
  pages={xx},
  year={xx},
  publisher={xx}
}
</pre></code>

