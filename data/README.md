# Data Folder

This directory is intended to store the dataset files required to run the experiments.

## Required Files

* `variables_final.csv`: This is the primary dataset used by the preprocessing scripts.

## Setup Instructions

Due to its sensitive nature and access restrictions, the MIMIC-IV dataset is **not** included in this repository.

1.  **Obtain Access:** You must first apply for and be granted access to the [MIMIC-IV database](https://physionet.org/content/mimiciv/2.2/).
2.  **Generate Dataset:** Run the `1_Data_Generation_and_EDA.ipynb` notebook located in the `/notebooks` directory. This will process the raw MIMIC-IV data and generate the required `variables_final.csv` file.
3.  **Place File Here:** Ensure the final `variables_final.csv` is placed in this `/data` directory.