# Welcome to GraphLeak: A realistic dataset to detect and locate leaks in water distribution networks

This repository contains the GraphLeak dataset, a comprehensive dataset designed for locating and identifying leaks in water distribution networks (WDN). The dataset is intended to support researchers in developing and evaluating water leak detection models, particularly those utilizing deep learning techniques.

In this GitHub, we share the datasets used in our research and also the source files/scripts for researchers who want to customize or build their own WDN models. Our WDN modeling framework can be adapted and replicated in various contexts. We use EPANET as the WDN modeling tool, and Matlab (with EPANET-MATLAB Toolkit) to create simulation scripts and export the datasets.

**Note:** Please refer to the corresponding folder in the folder list above for information about a specific publication. All of them use the same data generation and structure proposed in GraphLeak.

## Dataset Description
Deep learning algorithms rely on high-quality data for accurate training and evaluation. GraphLeak provides a comprehensive dataset in tabular format, where each column represents a specific variable measured by individual sensors. The dataset includes information on pressure, flow, volume, label, and localization. The datasets are exported to CSV (Comma-Separated Values) files.



<figure>
  <img src="/WDS_top.png" alt="WDN Topology" width="750">
  <figcaption>Figure 1: Water Distribution Network Topology (case study)</figcaption>
</figure>

## Data generation workflow

Each model contains daily demand patterns that emulate different consumption profiles. Water demands are uncertain and difficult to predict, thus these patterns preserve some usual aspects (for example, reduced water consumption over the night, when most people are sleeping) and one or two peak consumption periods during the day. Before every simulation day starts, a consumption profile is randomly assigned to each house. Also, a node base demand parameter is randomized within a reasonable range to represent the normal oscillations in the water demands WDN-wide.

<figure>
  <img src="/data_gen_flowchart.png" alt="Data generation flowchart" width="400" style="background-color: white; padding: 10px; border-radius: 8px;">
  <figcaption>Data generation workflow (scripted on Matlab/Matlab-EPANET Toolkit)</figcaption>
</figure>

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


# Authors
Lucas Roberto Tomazini;
Weliton do Carmo Rodrigues;
Rodrigo Pita Rolle;
Alexandre da Silva Simões;
Esther Luna Colombini;
Eduardo Paciência Godoy;


# Citation 
Please cite one of the following papers if you use this code for your researches:

<pre><code>@inproceedings{tomazini2024graphleak,
  title={GraphLeak: A Realistic Dataset for Analyzing Leaks in Water Distribution Systems},
  author={Tomazini, Lucas Roberto and Rolle, Rodrigo Pita and Godoy, Eduardo Paci{\^e}ncia and Colombini, Esther Luna and da Silva Simoes, Alexandre},
  booktitle={XXV Congresso Brasileiro de Autom{\'a}tica (CBA 2024)},
  year={2024}
}
</code></pre>

