# Welcome!

Folder containing data sets and additional files related to the paper "Graph Neural Networks and LPWAN for Water Leakage Localization".

---

## MATLAB code

The MATLAB code used to produce the CSV datasets is in the 'm' files inside each folder. They use the EPANET-MATLAB Toolkit (https://github.com/OpenWaterAnalytics/EPANET-Matlab-Toolkit ) and some custom functions that we provided in the file 'CustomFunctions.rar'. Extract them to your MATLAB path.

---

## Python files

The Python files perform preprocessing (convert CSV to Tensor files) and the algorithm execution itself. Please check if your Python environment has the needed libraries and remember to include the 'model' folder in your Python folder. It contains the GGNN model (based on https://github.com/tfjonas/ggnn_fault_loc ). 

