# Multimodal Explainable Automated Diagnosis of Autism Spectrum Disorder

This repository contains the implementation for the **model training** and **data processing** components of the paper titled **"Multimodal Explainable Automated Diagnosis of Autism Spectrum Disorder."**

The research was conducted during a five-month internship at the [Computer Science Research Institute of Toulouse (IRIT)]([URL](https://www.irit.fr/en/home/)), within the [Generalised Information Systems team (SIG)](https://www.irit.fr/en/departement/dep-data-management/sig-team/), 
and supervised by:
- **Dr. Moncef Garouani**, whose work specializes in the automatic selection and parametrization of machine learning algorithms, as well as AI explainability.
- **Dr. Julien Aligon**, who specializes in post-hoc methods for prediction explanation.

The research was carried out by **Meryem Ben Yahia**, M.Sc Machine Learning and Data Mining at Université Jean Monnet.

---
## Repository Structure

Multimodal-Explainable-Automated-Diagnosis-of-Autistic-Spectrum-Disorder/
├── data/                                   # dataset-related files
│   ├── 0. measure generator using BASC122.ipynb  # Notebook for generating measures using the BASC122 atlas
│   ├── Phenotypic_V1_0b_preprocessed1.csv     
│   └──  totaldata.csv                        
│ 
├── notebooks/                              # Jupyter notebooks for analysis and model development
│   ├── 1. Exploratory Data Analysis on ABIDE and Phenotypic Feature Selection.ipynb  
│   ├── 2. Test of Classical Models on Phenotypic Features.ipynb 
│   └── Neural Networks (fMRI, Multimodal, Pheno)/  # Specialized directory for neural network models
│       ├── 1. Neural Network on RFE fMRI data.ipynb  # Neural network model training on fMRI data
│       ├── 2. Neural Network on RFE Multimodal data.ipynb  # Neural network model training on multimodal data
│       ├── 3. Neural Network on RFE Phenotypic data.ipynb  # Neural network model training on phenotypic data
│       ├── best_model_FMRI_RFE.h5          # Saved model trained on fMRI data
│       ├── best_model_Multimodal_RFE.h5    # Saved model trained on multimodal data
│       ├── best_model_Pheno_RFE.h5         # Saved model trained on phenotypic data
│       ├── hyperparameters.json            # JSON file containing best model hyperparameters
│       ├── model.py                        # Neural network architecture definition
│       ├── plotting.py                     # Functions for plotting metrics
│       ├── rfe_data.csv                    # RFE processed data for feature selection
│  
├── results/                               
│   ├── evaluation_metrics.json             # JSON file with evaluation metrics
│   └── hyperparameter_tuning_results.json  # File with results from hyperparameter tuning for classical models
│   
├── src/                                    # Source code for data processing, model building, and evaluation
│   ├── __pycache__/                        # Compiled Python files for optimization
│   ├── __init__.py                         
│   ├── data_preprocessing.py               # Data preprocessing functions
│   ├── fetchfmri.py                        # Fetching fMRI data
│   ├── model_evaluation.py                 # Evaluation of model performance
│   ├── model_explainability.py             # Explainability functions (e.g., SHAP)
│   ├── model_training.py                   # Model architecture and training routines
│   └── model_tuning.py                     # Hyperparameter tuning for the model
│
├── script_fetch_abide.py                   # Script to fetch ABIDE pre-processed CPAC Pipeline
└── README.md                               # Overview of the project, installation instructions, and usage

---
## Results

- **Mean Accuracy**: 98.64% (±0.86%)
- **AUC**: 1.00
- **Recall**: 99%
- **Precision**: Near-perfect for both ASD and Non-ASD classes.


---
## License

This project is licensed under the MIT License. 

