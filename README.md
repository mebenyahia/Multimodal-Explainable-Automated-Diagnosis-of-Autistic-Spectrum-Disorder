# Multimodal Explainable Automated Diagnosis of Autism Spectrum Disorder

This repository contains the implementation for the **model training** and **data processing** components of the paper titled **"Multimodal Explainable Automated Diagnosis of Autism Spectrum Disorder."**

The research was conducted during a five-month internship at the [Computer Science Research Institute of Toulouse (IRIT)](https://www.irit.fr/en/home/), within the [Generalised Information Systems team (SIG)](https://www.irit.fr/en/departement/dep-data-management/sig-team/), 
and supervised by:
- **Dr. Moncef Garouani**, whose work specializes in the automatic selection and parametrization of machine learning algorithms, as well as AI explainability. <br>
  [Dr. Garouani on Google Scholar](https://scholar.google.fr/citations?user=4nXi7GAAAAAJ&hl=fr).
- **Dr. Julien Aligon**, who specializes in post-hoc methods for prediction explanation. <br>
  [Dr. Aligon on Google Scholar](https://scholar.google.fr/citations?user=SL-IYIQAAAAJ&hl=fr).

The research was carried out by **Meryem Ben Yahia**, M.Sc Machine Learning and Data Mining at Université Jean Monnet.

---
## The Dataset: Autism Brain Imaging Data Exchange I (ABIDE I) Dataset

The dataset used in our study is from the Autism Brain Imaging Data Exchange I (ABIDE I), a 2012 initiative that involves data contributions from 17 international sites. This dataset includes imaging and phenotypic data collected from a total of 1,112 subjects, composed of both individuals with Autism Spectrum Disorder (ASD) and control participants. 

| Category         | ABIDE I                               |
|------------------|---------------------------------------|
| Participants     | 1,112 total (539 ASD, 573 controls)   |
| Age Range        | 7-64 years (median: 14.7)             |
| Number of Sites  | 17 international sites                |
| Imaging Data     | Resting State Functional MRI         
|                  | Structural MRI                      |
| Phenotypic Data  | Composite Phenotypic File           |
|                  | Phenotypic Data Legend              |

#### Access
The dataset and its corresponding data legend are open access and can be found at the following links:
- [Access Link](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html)
- [Data Legend](https://fcon_1000.projects.nitrc.org/indi/abide/ABIDEII_Data_Legend.pdf)

For more detailed information regarding data access, formats, and how to use this dataset, refer to the [official documentation](https://fcon_1000.projects.nitrc.org/indi/abide/).


---
## Repository Structure
```bash
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
```
---
## Results

| Model      | Total Training Time | Mean Accuracy (5 folds) | Standard Deviation (5 folds) |
|------------|---------------------|-------------------------|------------------------------|
| fMRI-input DNN      | 75.19 seconds        | 99.66%                  | 0.42%                        |
| Multimodal-input DNN | 60.19 seconds        | 98.64%                  | 0.86%                        |
| Pheno-input DNN      | 23.82 seconds        | 63.55%                  | 2.04%                        |



---
## License

This project is licensed under the MIT License. 

