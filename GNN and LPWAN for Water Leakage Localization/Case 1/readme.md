## Files for Case 1

---

## WDN overview
![WDN Map](GNN and LPWAN for Water Leakage Localization/Case 1/map-case1.png)

This simple WDN contains 4 consumption units (yellow dots) and 3 joint nodes (black dots). Leakage can be simulated in the red nodes. We did not enable two or more leakage points at once in our research. Leakage is pressure-dependant, using the emitter coefficient feature on EPANET. 

The CSV file is structured as follows:
* Column 1: time (in seconds)
* Columns 2-9: flow on monitored nodes (N2-N9)
* Columns 10-17: pressure on monitored nodes (N2-N9)
* Columns 18-25: volume on monitored nodes (N2-N9)
* Columns 26-33: x coordinates of monitored nodes (N2-N9)
* Columns 34-41: y coordinates of monitored nodes (N2-N9)
* Columns 42-49: z coordinates of monitored nodes (N2-N9)
* Columns 50-54: indicate absence (0) or presence (1) of leakage on nodes N10-N14
* Columns 55-57: x-y-z coordinates of the active leakage (if there are none, they are filled with zeros)
* Column 58: weekday (reserve/future use)

---

## EPANET model

The editable WDN model is the file 'case1.inp'. You can open and edit it using EPANET (free, open-source). 

---

## MATLAB code

The MATLAB code used to produce the CSV datasets is in the file 'case1.m'. It uses the EPANET-MATLAB Toolkit (https://github.com/OpenWaterAnalytics/EPANET-Matlab-Toolkit) and some custom functions that we provided in the root folder (CustomFunctions.rar).

---

## Python files

The file 'preprocess-case1' converts the CSV data sets to Tensor format. To use it, place the CSV files inside a folder named 'Original_Data' and create an empty folder named 'work_dir'. The Python code will output the tensor files to the 'work_dir' folder.

The file'GGNN-case1' runs the GGNN algorithm using the tensor files. It outputs the training/validation statistics of each training/validation pair to the folder 'work_dir2' and a binary Pickle file to the folder 'results'. Thus, you can open these files to generate the box plots and perform other kinds of analysis.
