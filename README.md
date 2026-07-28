# Welcome to GraphLeak: Realistic modeling of water distribution networks

Welcome to GraphLeak, a comprehensive dataset designed for the realistic simulation of Water Distribution Networks (WDNs). The dataset is intended to support researchers in developing and evaluating leakage detection and localization models. It is particularly suited for Deep Learning (DL) algorithms, which typically require large amounts of data for training and validation.

In addition to the ready-to-use CSV files of the WDNs developed for our own research, we also share source files and scripts for researchers who want to customize or build their own WDN models. Our WDN modeling framework can be adapted and replicated in various contexts. We use EPANET as the WDN modeling tool and Matlab (with the EPANET-Matlab Toolkit) to create simulation scripts and export the datasets.

**Note:** This GitHub is organized in two main folders: 
- Publications: Please refer to this folder if you are looking for information about a specific publication of our research group. 
- WDN Datasets: Please refer to this folder to find ready-to-use CSV files, Matlab scripts and EPANET models of the WDN models developed within our research.


# GraphLeak's scope and overview
Deep learning algorithms rely on high-quality data for accurate training and evaluation. GraphLeak provides a comprehensive dataset in tabular format, where each column represents a specific variable measured by individual sensors. The dataset includes information on pressure, flow, volume, leakage label, and location. The data is exported as CSV (Comma-Separated Values) files.  

<figure>
  <img src="./Publications/GraphLeak A realistic dataset to detect and locate leaks in water distribution networks/WDS_top.png" alt="Example of a WDN" width="750">
  <figcaption>Figure 1: Water Distribution Network Topology (case study)</figcaption>
</figure>

Each model contains daily demand patterns that emulate different consumption profiles. Because water demands are uncertain and difficult to predict, these patterns preserve typical characteristics (e.g., reduced consumption at night) alongside one or two daily peak consumption periods. Before each simulation day begins, a consumption profile is randomly assigned to each household. Additionally, the base demand parameter of each node is randomized within a realistic range to represent normal demand oscillations across the WDN.

<figure>
  <img src="./Publications/GraphLeak A realistic dataset to detect and locate leaks in water distribution networks/data_gen_flowchart.png" alt="Data generation flowchart" width="400" style="background-color: white; padding: 10px; border-radius: 8px;">
  <figcaption>Figure 2: Data generation workflow (scripted on Matlab/Matlab-EPANET Toolkit)</figcaption>
</figure>

## Prerequisites

To run our scripts and recreate the datasets locally, or to use our source codes as a template for your own WDN models, you might need the following tools:
- Matlab
- Matlab-EPANET Toolkit [(click here)](https://github.com/OpenWaterAnalytics/EPANET-Matlab-Toolkit)
- EPANET [(click here)](https://github.com/USEPA/EPANET2.2)
- Optional but useful: add our custom Matlab functions that we use in our scripts by downloading the "matlab" folder to your Matlab path.

## Using the data

Explore the folders to see examples of our datasets and how we use them to perform various leakage analyses in water distribution networks. Each folder contains distinct insights and tools you can adopt for your own work! Start [here](./Publications/GraphLeak%20A%20realistic%20dataset%20to%20detect%20and%20locate%20leaks%20in%20water%20distribution%20networks) for an in-depth explanation of the dataset, and feel free to explore our research group's other publications.

## Developers and contributors
- Eduardo Paciência Godoy (eduardo.godoy@unesp.br) - Associate Professor, Unesp Sorocaba
- Rodrigo Pita Rolle (rodrigo.rolle@unesp.br) - Assistant Professor, Unesp Itapeva
- Weliton do Carmo Rodrigues (wc.rodrigues@unesp.br) - Ph.D. student, Unesp Sorocaba
- Lucas Roberto Tomazini (lucas.tomazini@unesp.br) - M.Sc. student, Unesp Sorocaba

Contributors: Alexandre da Silva Simões (Unesp Sorocaba), Esther Luna Colombini (Unicamp).

## Funding
This research project is financed in part by the following entities/grants:
- Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP) - grant 2025/09285-5
- Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq)

# Citation 
Please cite one of the following papers if you use our content for your research:

<pre><code>
  @INPROCEEDINGS{rodrigues2025estudo,
  author={Rodrigues, Weliton C. and Rolle, Rodrigo P. and Godoy, Eduardo P.},
  booktitle={2025 16th IEEE International Conference on Industry Applications (INDUSCON)}, 
  title={Estudo da Ponderação em Redes Neurais de Grafo para Localização de Vazamentos}, 
  year={2025},
  pages={516-521},
  doi={10.1109/INDUSCON66435.2025.11241242}}

  @INPROCEEDINGS{rolle2024leveraging,
  author={Rolle, Rodrigo P. and Rodrigues, Weliton C. and Tomazini, Lucas R. and Monteiro, Lucas N. and Godoy, Eduardo P.},
  booktitle={2024 IEEE International Workshop on Metrology for Industry 4.0 & IoT (MetroInd4.0 & IoT)}, 
  title={Leveraging graph-based leak localization in water distribution networks}, 
  year={2024},
  pages={192-197},
  doi={10.1109/MetroInd4.0IoT61288.2024.10584129}}

  @inproceedings{rodrigues2024leak,
  title={Leak Location in Water Distribution Networks based on Sampling and Graph Aggregation (GraphSAGE)},
  author={Rodrigues, Weliton C and Rolle, Rodrigo P and Godoy, Eduardo P},
  booktitle={Congresso Brasileiro de Autom{\'a}tica-CBA},
  volume={4},
  number={1},
  year={2024}
  }  
  
  @inproceedings{tomazini2024graphleak,
  title={GraphLeak: A Realistic Dataset for Analyzing Leaks in Water Distribution Systems},
  author={Tomazini, Lucas Roberto and Rolle, Rodrigo Pita and Godoy, Eduardo Paci{\^e}ncia and Colombini, Esther Luna and da Silva Simoes, Alexandre},
  booktitle={XXV Congresso Brasileiro de Autom{\'a}tica (CBA 2024)},
  year={2024}
}
</code></pre>

