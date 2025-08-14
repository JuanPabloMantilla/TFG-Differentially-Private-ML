# Differentially Private Machine Learning: Implementation and Analysis

![Language](https://img.shields.io/badge/Language-Python-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Project Overview

[cite_start]This repository contains the official implementation for the final degree thesis, "Differentially Private Machine Learning: Implementation and Analysis of Gradient and Dataset Perturbation Techniques"[cite: 6]. [cite_start]The project investigates, implements, and empirically evaluates two core techniques for integrating the formal privacy guarantees of Differential Privacy (DP) into machine learning pipelines[cite: 26].

[cite_start]The primary goal is to analyze the fundamental trade-off between model utility and the strength of privacy guarantees when training on sensitive, real-world medical data from the **MIMIC-IV database**[cite: 29, 649].

## Key Features & Methodology

* **Two Core DP Techniques Implemented:**
    1.  [cite_start]**Dataset Perturbation:** Calibrated Gaussian noise is added directly to the training data before model fitting[cite: 40].
    2.  [cite_start]**Gradient Perturbation:** Noise is injected into gradients during training via Differentially Private Stochastic Gradient Descent (DP-SGD)[cite: 41, 583].
* [cite_start]**Real-World Sensitive Data:** The experiments use a multi-class classification task on a dataset derived from the MIMIC-IV critical care database[cite: 29].
* [cite_start]**Rigorous Evaluation:** Models are evaluated against a non-private baseline, using appropriate metrics for class imbalance like **Macro F1-Score** and **Macro OVO AUC** to measure the privacy-utility trade-off[cite: 30, 43].

## Key Results

The project confirms the inverse correlation between privacy and utility. [cite_start]The DP-SGD model with **ε = 7.27** was identified as the optimal choice for this medical context, providing a meaningful privacy guarantee (ε < 10) while maintaining the highest utility among the strongly private models[cite: 904].

*(Aquí, añade una captura de pantalla de la Tabla 4.4 de tu memoria, guárdala en una carpeta `assets/` o `img/` y referénciala así)*
![Comparative Results](assets/results_comparison.png)

## Repository Structure

```
.
├── .gitignore
├── LICENSE
├── README.md
├── environment.yml
├── requirements.txt
├── data/              # -> Not tracked by git. For local data storage.
├── notebooks/         # -> Jupyter notebooks for EDA and results visualization.
├── results/           # -> Not tracked by git. For generated results.
└── src/               # -> All source code for experiments and utilities.
```

## Setup & Usage

**1. Clone the repository:**
```bash
git clone [https://github.com/JuanPabloMantilla/TFG-Differentially-Private-ML.git](https://github.com/JuanPabloMantilla/TFG-Differentially-Private-ML.git)
cd tu-repositorio
```

**2. Create and activate the environment:**

* **Using Conda:**
    ```bash
    conda env create -f environment.yml
    conda activate your-env-name
    ```
* **Using pip and venv:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

**3. Data Setup:**
This project uses the MIMIC-IV database, which requires obtaining credentials for access. Due to its sensitive nature, the data is not included in this repository. Please place the generated `.csv` file inside the `/data` directory.

**4. Run Experiments:**
The core experiment scripts are located in the `/src` directory.
```bash
# Example for running the gradient perturbation experiment
python src/gradient_perturbation_experiment.py
```

## Author

* **Juan Pablo Mantilla Carreño**
