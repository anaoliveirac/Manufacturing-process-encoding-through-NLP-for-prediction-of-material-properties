**Manufacturing Process Encoding Through Natural Language Processing for Prediction of Material Properties**

This repository contains the data, code, and supplementary materials related to the research paper:  
[*"Manufacturing Process Encoding Through Natural Language Processing for Prediction of Material Properties"*](https://www.sciencedirect.com/science/article/pii/S0927025624001174).  

## Authors
- **Ana P.O. Costa**<sup>a</sup>  
- **Mariana R.R. Seabra**<sup>c</sup>  
- **José M.A. César de Sá**<sup>a,b</sup>  
- **Abel D. Santos**<sup>a,b</sup>  

<sup>a</sup> Department of Mechanical Engineering, University of Porto  
<sup>b</sup> INEGI, Institute of Science and Innovation in Mechanical and Industrial Engineering  
<sup>c</sup> CFUL - centro de filosofia da universidade de Lisboa  

---

## Abstract

Knowledge of manufacturing processes is crucial to determine the final properties of a material. This work focuses on analyzing the relationship between final properties, chemical composition, and manufacturing process through data analysis. Techniques of natural language processing (NLP) were employed to encode the manufacturing process as input for a neural network.  

### Study Highlights:
1. **Data Preparation**:  
   - Data was gathered, cleaned, and analyzed using statistical methods, K-means clustering, and Principal Component Analysis (PCA).  

2. **Model Development**:  
   - A Fully Connected Neural Network (FCNN) was used to predict elongation, yield strength (YS), and ultimate tensile strength (UTS).  
   - Overfitting was mitigated using dropout functions and K-fold cross-validation.  

3. **Key Results**:  
   - Predictions for elongation, YS, and UTS demonstrated reasonable accuracy.  
   - Case studies included:  
     - Predictions for an existing alloy excluded from the training/test set.  
     - Design of a new stainless steel alloy combining high YS, UTS, and Pitting Resistance Equivalent Number (PREN).  
     - Application to a TRIP (Transformation Induced Plasticity) family steel, showcasing model versatility.

---

## Repository Contents

### Folder Structure
- **`data/`**  
  - Contains raw, processed, and analyzed datasets.  
- **`code/`**  
  - Includes Python scripts and Jupyter notebooks for data preprocessing, model training, and evaluation.  
- **`README.md`**  
  - This file.  

---

## Requirements

### Software
- Python 3.8 or higher

### Libraries
Install dependencies with:  
```bash
pip install -r requirements.txt
