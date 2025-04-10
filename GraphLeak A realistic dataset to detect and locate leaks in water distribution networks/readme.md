# Full paper 
The full paper is available at: https://www.sba.org.br/cba2024/papers/paper_7042.pdf
<pre><code>@inproceedings{tomazini2024graphleak,
  title={GraphLeak: A Realistic Dataset for Analyzing Leaks in Water Distribution Systems},
  author={Tomazini, Lucas Roberto and Rolle, Rodrigo Pita and Godoy, Eduardo Paci{\^e}ncia and Colombini, Esther Luna and da Silva Simoes, Alexandre},
  booktitle={XXV Congresso Brasileiro de Autom{\'a}tica (CBA 2024)},
  year={2024}
}
</code></pre>


## Dataset Description
Deep learning algorithms rely on high-quality data for accurate training and evaluation. GraphLeak provides a comprehensive dataset in tabular format, where each column represents a specific variable measured by individual sensors. The dataset includes information on pressure, flow, volume, label, and localization. The datasets are exported to CSV (Comma-Separated Values) files.



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


# Authors
Lucas Roberto Tomazini;
Weliton do Carmo Rodrigues;
Rodrigo Pita Rolle;
Alexandre da Silva Simões;
Esther Luna Colombini;
Eduardo Paciência Godoy;
