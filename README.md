# DOPT: D-learning with Off-Policy Target

D-learning is a sample-based model-free Learning-based Lyapunov Control (LLC) method proposed by [Quan Quan](https://arxiv.org/abs/2206.03809) as a parallel method to Q-learning. It collects
system data and improves controllers with learned Lyapunov candidates and D-functions

DOPT is a LLC method, and is a variant of D-learning method. DOPT is designed to uses current and historical system data to optimize the NN controller within the framework of Lyapunov theory. It can obtain fast converging controller with a stability guarantee, and higher sample efficiency and more steady training process than vanilla D-learning.

![image](https://github.com/user-attachments/assets/26da8133-a487-4131-9aa8-a10e44c6ec5b)

## 1. Files Structure
"_\systems_and_functions_" contains files of dynamic systems, algorithms and kits.

"_\_\_4\_\_.ipynb_" is code for algorithm \_\_ implemented on system \_\_.

"_Verifying_almost_Lypunov_Conditions.ipynb_" is code for verifying almost Lypunov Conditions in a sample-based way.

"_requirements.txt_" is the list of required packages for successfully running these codes.

## 2. Run

Run "_\_\_4\_\_.ipynb_" in environment that satisfies "_requirements.txt_".

## 3. Maintaince

For any technical issues, please contact Zhaolong Shen (shenzhaolong1330@buaa.edu.cn) or Quan Quan (qq_buaa@buaa.edu.cn).
