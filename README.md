# HG-P-LBA
## **Project Introduction**  
**HG-P-LBA** is a a graph neural network-based tool for predicting protein ligand affinity.

## Installation & Execution for HPC Deployment

### **1. Build the Environment**
```bash
conda env create -f environment.yml
```

### **2. Usage&Example**

**Prepare Raw Data:**
- **Step 1:** Prepare the raw data, for example, please visit the official website of PDBbind dataset to download the raw data, and then put the downloaded data into the "data" directory of the project.

- **Step 2:** Convert the raw data downloaded to the data directory into training, validation, and testing datasets in a certain proportion.


**Preprocessing Raw Data:** 
- **Note:** Generate graph datasets(train.pkl、val.pkl、test.pkl) using already partitioned raw datasets(train、val、test) separately.
```shell
python process.py --raw_data_path ./data/sample_set/ --graph_data ./data/sample.pkl
```

**Model Training:**
```shell
python train.py
```
  
  

