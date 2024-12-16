# XGBoost and CNN model implementation with Ray Framework:

## Description:

This project demonstrates the implementation of distributed machine learning models using the Ray framework. It includes two distinct models:
- **XGBoost**: A classification model trained on the Breast Cancer dataset.
- **CNN**: An image classification model trained on the CIFAR-10 dataset.

The focus of this project is to evaluate the efficiency and scalability of the Ray framework in reducing execution time while maintaining robustness through fault tolerance and hyperparameter optimization.

## Requirements

### Hardware requirements:
- **For XGBoost**:
  - 4 CPUs
  - 16GB RAM
  - 64GB Memory
     
- **For CNN**:
  - 2 CPUs
  - 1 GPUs (minimum)
  - 25GB RAM for CPUs & 16GB RAM for GPU
  - 112GB Memory 

### Software requirements:
- Python 3.10.11
- Libraries:
  - `ray`
  - `tensorflow`
  - `keras`
  - `xgboost`
  - `scikit-learn`


## How to Run: (Approach - 1)

1. Run the batch file that is present in the github repo named as 'Run.bat' file.
   - Navigate to directory where your batch file is located and replace the 'path_to_your_project_directory' to your path.
    ```bash
   cd path_to_your_project_directory
2. Run the batch file by adding the command 'run.bat'
   ```bash
   run.bat
   
## How to Run: (Approach - 2)

You can directly double click on **run.bat** file and execution of python files will start successfully ensuring all the resources are available.



