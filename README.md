# DOPT: D-learning with Off-Policy Target

D-learning is a sample-based model-free Learning-based Lyapunov Control (LLC) method proposed by [Quan Quan](https://arxiv.org/abs/2206.03809) as a parallel method to Q-learning. It collects
system data and improves controllers with learned Lyapunov candidates and D-functions

DOPT is a LLC method, and is a variant of D-learning method. DOPT is designed to uses current and historical system data to optimize the NN controller within the framework of Lyapunov theory. It can obtain fast converging controller with a stability guarantee, and higher sample efficiency and more steady training process than vanilla D-learning.

![image](https://github.com/user-attachments/assets/26da8133-a487-4131-9aa8-a10e44c6ec5b)

## 1. Files Structure
"\experiment_results" contains experimental results of three algorithms (DDPG, D-learning, DOPT) across two systems (Inverted pendulum and Single-track car).

"\figs_and_animations" contains figures and animations.

"\systems_and_functions" contains files of dynamic systems, algorithms and kits.

"\*\*\_4\_\*\*.ipynb" is code for algorithm \*\* implemented on system \*\*.

"Verifying_almost_Lypunov_Conditions.ipynb" is code for verifying almost Lypunov Conditions in a sample-based way.

"requirements.txt" is the list of required packages for successfully running these codes.

## 2. Run

Run "\*\*\_4\_\*\*.ipynb" in environment that satisfies "requirements.txt".

## 3. Maintaince

For any technical issues, please contact Zhaolong Shen (shenzhaolong1330@buaa.edu.cn) or Quan Quan (qq_buaa@buaa.edu.cn).
