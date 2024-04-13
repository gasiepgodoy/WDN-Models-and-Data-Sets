## Files for Case 1

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
