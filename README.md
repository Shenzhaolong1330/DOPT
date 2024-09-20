# DOPT: D-learning with Off-Policy Target

+ D-learning is a sample-based model-free Learning-based Lyapunov Control (LLC) method proposed by [Quan Quan](https://arxiv.org/abs/2206.03809) as a parallel method to Q-learning. It collects
system data and improves controllers with learned Lyapunov candidates and D-functions

+ DOPT is a LLC method, and is a variant of D-learning method. DOPT is designed to uses current and historical system data to online iteratively optimize the NN controller within the framework of Lyapunov theory. It can obtain a faster converging controller with a stability guarantee, higher sample efficiency and more steady training process than vanilla D-learning and DDPG.

![image](https://github.com/user-attachments/assets/26da8133-a487-4131-9aa8-a10e44c6ec5b)[Overview of the DOPT]

## 1. Files
### 1.1 Files and Folders

+ "_\experiment_results_" contains training data and results of each algorithm.

+ "_\figs_and_animations_" contains files of results presented in the paper and video clip.

+ "_\systems_and_functions_" contains important files of dynamic system classes, algorithm classes, networks and tool kits.

+ "_\_\_4\_\_.ipynb_" is the code for algorithm "\_\_" implemented on system "\_\_".

+ "_Verifying_almost_Lypunov_Conditions.ipynb_" is code for verifying almost Lypunov Conditions in a sample-based way.

+ "_experiment_results.zip_" is zipped file structure, unzip this file before running the code.

+ "_requirements.txt_" is the list of required packages for successfully running these codes.

### 1.2 File Structure

- DOPT-main
  - experiment\_results
    - DDPG
      - data
      - figs
      - model
    - Dlearning
      - data
      - figs
      - model
    - DOPT
      - data
      - figs
      - model
  - figs_and_animations
  - systems_and_functions
    - \_\_pycache\_\_

## 2. Run

+ Run "_\_\_4\_\_.ipynb_" in environments that satisfy "_requirements.txt_".

+ **Unzip the file "_experiment_results.zip_" before running the code** to make sure there is a folder for running data storage.

## 3. Results and Interpretations

### 3.1 DDPG in Inverted Pendulum
![Convergence Speed and Judgment](https://github.com/user-attachments/assets/04679edb-dca0-4fe1-964a-bd58c152f199)

### 3.2 D-learning in Inverted Pendulum
![image](https://github.com/user-attachments/assets/9106200d-9828-41f6-b5f8-7019fd3fe6c7)


### 3.3 DOPT in Inverted Pendulum
Spikes may be spotted in the training process, which is caused by the calculation method for convergence steps. 
This presentation result is somewhat misleading and can actually be viewed as a smooth process, the real convergence performance should be analyzed together with the state trajectory.

![image](https://github.com/user-attachments/assets/35661149-fb00-4796-8aeb-b12382f3c0ed)
[Training processes of DOPT and Trajectory of spike situation in different run]

## 4. Maintaince
For any technical issues, please do not hesitate to contact Shen Zhaolong (shenzhaolong@buaa.edu.cn) or Quan Quan (qq_buaa@buaa.edu.cn).

## 5. <u>Declaration to the Reviewer</u>

The performance of the algorithm DOPT depends on the selection of hyperparameters. After fine-tuning of hyperparameters, the algorithm's performance in this open-source code has exceeded the experimental results presented in the first submission. If this work is accepted, the latest finely tuned results will be added to the final version
