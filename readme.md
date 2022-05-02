## CTL-MTNet: A Novel Mixed Task Net Based on CapsNet and Transfer Learning for Single-Corpus and Cross-Corpus Speech Emotion Recognition


# Introduction
In this paper, a Capsule Network (CapsNet) and Transfer Learning based Mixed Task Net (CTL-MTNet) is proposed to deal with both the single-corpus and cross-corpus SER tasks simultaneously. 

# Prerequisites

Our code is based on Python3 (>= 3.7). There are a few dependencies to run the code. The major libraries are listed as follows:
* TensorFlow (== 2.4.0)
* NumPy (== 1.19.5)
* SciPy (== 1.6.0)
* librosa (==0.8.0)
* Speechpy (==2.4)
* Pandas (== 1.2.1)
* Scikit-learn (== 0.20.4)

# Dataset
In the experiments, in order to compare with the state-of-the-art methods. We use four datasets, including the Institute of Automation of Chinese Academy of Sciences (CASIA), Berlin Emotional Database (EmoDB), Surrey Audio-Visual Expressed Emotion Database (SAVEE), and Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).
# Feature Processing
In this experiment, 39-dimensional MFCCs are extracted from the Librosa toolbox to serve as inputs with a frameshift of 0.0125 s and a frame length of 0.05 s.
* librosa.feature.mfcc
# Model Details
## Training CPAC module
    python ./CPAC_code/main.py 

## Training CAAM module
    
    python ./CAAM_code/main.py 
**Default settings in CPAC**:
* Training configs: 
    * batch_siez = 64, lr = 0.001, epoch = 300, kFold = 10
    * opt ='Adam', $\beta_1$ = 0.975, $\beta_2$ = 0.932, epsilon = 1e-8
    
    
    
**Default settings in CAAM**:
    
* Training configs: 
    * batch_siez = 512, lr = 0.001, epoch = 120
    * opt ='Adam', $\beta_1$ = 0.975, $\beta_2$ = 0.935, epsilon = 1e-8
## Folder structure
```
└────CPAC_code
│    ├──── Common_Model.py
│    ├──── Config.py
│    ├──── main.py
│    ├────Models.py
│    ├──── Utils.py
│    ├────Models/
│    └────Results/
└────CAAM_code/
│    ├────_CAAM/
│    │    ├────__init__.py
│    │    ├────_deep.py
│    │    ├────_mdd.py
│    │    └────utils.py
│    ├────Common_Model.py
│    ├────Config.py
│    ├────CPAC_Models.py
│    ├────main.py
│    ├────Models/
│    └────Utils.py
├── README.md
```