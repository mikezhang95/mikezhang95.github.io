---
layout: post
title: Autonomous Driving Algorithms
date: 2026-01-25
description: 
img: 
tags: [Blog]
author: Yuan Zhang
giscus_comments: true
math: true
---



> Date: 2026-01-31 | Estimated Reading Time: 30 min | Author: Yuan Zhang

Autonomous driving systems combine learning-based prediction with classical estimation and optimization. This post surveys key components commonly found in modern driving stacks. 

[Multipath++, 3DGS, Kalman filter, iLQR]

---

## 1 Overview

state $$x_t$$, control $$u_t$$, observations $$z_t$$

---

## 2 Kalman Filter: State Estimation

The Kalman filter provides an optimal (minimum mean square error) recursive Bayesian estimate $$x_t$$ for the state of a dynamic system under linear Gaussian assumptions. 

### 2.1 Linear Kalman Filter

#### State Model 

The prediction model predicts the next state based on a linear dynamics model: 
$$
x_{t+1} = F_t x_t + B_t u_t + w_t, w_t \sim \mathcal{N}(0, Q_t).
$$
The measurement model generates the observations with a linear transformation: 
$$
z_t = H_t x_t + v_t, v_t \sim \mathcal{N}(0, R_t). 
$$


#### Predict-Update Cycle

The filter maintains a posterior Gaussian estimate of the current state: $$p(x_t):=\mathcal{N}(x_t;\mu_t, \Sigma_t)$$. The posterior comes from consists of 2 steps: the prediction step and update step. 

The prediction step calculates a prior from the prediction model: 


$$
\begin{aligned}
	\mu_{t|t-1} = F_t \mu_{t-1} + B_t u_t \\
	\Sigma_{t|t-1} = F_t \Sigma_{t-1}F_t^T + Q_t \\
\end{aligned}
$$




The update step updates this prior given new observations:


$$
\begin{aligned}
	K_t = \Sigma_{t|t-1} H_t^T(H_y\Sigma_{t|t-1} H_t^T + R_t)^{-1} \\
	\mu_t  = \mu_{t|t-1} + K_t (z_t - H_t \mu_{t|t-1}) \\
	\Sigma_{t} = (I - K_tH_t)\Sigma_{t|t-1} \\
\end{aligned}
$$


Kalman filter is just an application of the combinations of multiple Gaussian distributions. 

---

### 2.2 Extensions

- EKF: nonlinear dynamics
- UKF: sigma-point approximation
- Smoothing for offline estimation

---

## 3 iLQR (Iterative LQR): Trajectory Optimization

iLQR solves discrete-time nonlinear optimal control problems (find optimal $$u^*_t$$) by iterative linearization and quadratic approximation. 

### 3.1 Optimal Control Problem

$$
\begin{aligned}
	\min_{u_0, \dot, u_{T-1}} c_T(x_T) + \sum_{t=0}^{T-1} c_t(x_t, u_u) \\
	s.t. x_{t+1} = f(x_t, u_t), t=0, \dots, T-1
\end{aligned}
$$

### 3.2 iLQR

At each iteration, linearize both transition function $$f$$ and cost function $$c$$ around the current trajectory $$(\bar{x}_t, \bar{u}_t)$$ to achieve an LQR problem:
$$
\begin{aligned}
	\min_{u_0, \dot, u_{T-1}} c_T(x_T)  +  \sum_{t=0}^{T-1} \delta x_t^T Q_{t} \delta x_t + q^T_{t} \delta x_t \\ 
	s.t. \delta x_{t+1} \approx  A_t \delta x_t + B_t \delta u_t, , t=0, \dots, T-1
\end{aligned}
$$


where $$\delta x_t = x_t - \bar{x}_t, \delta u_t = u_t - \bar{u}_t, A_t = \nabla_xf\|_{\bar{x}_t, \bar{u}_t}, B_t = \nabla_uf\|_{\bar{x}_t, \bar{u}_t}, q_t = \nabla_xc_t\|_{\bar{x}_t, \bar{u}_t}, Q_t = \nabla_{xx}c_t\|_{\bar{x}_t, \bar{u}_t}$$, etc. We can extend the cost term on $$u_t$$ as quadratic and linear terms of $$R_t$$ and $$r_t$$ as well.

One can easily solve this local LQR by performing [discrete-time Riccati differential equation](https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator) and calculating gain matrix $$K_t, k_t$$. Then the optimal control $$\delta u_t = k_t + K_t \delta x_t$$. Iterating the whole process until convergence. 

---

## 4 TODO: MultiPath

The core of MultiPath++ is to parameterize a **multimodal probability distribution** of future trajectories directly from historical trajectories and map information via a deep network.

### 4.1 Mathematical Principles

#### Trajectory Distribution Modeling: GMM/MoG

MultiPath++ predicts a distribution of future behavior parameterized as a Gaussian Mixture Model (**GMM**; ), with a mixture of $$M$$ modes: 


$$
p(Y|X) = \sum_{k=1}^K \pi_k \cdot \mathcal{N}(Y; \mu_k(X), \Sigma_k(X))
$$

#### Loss Function

$$
\mathcal{L}_{NLL}= -\log p(Y|X) = -\sum_{m=1}^M \sum_{k=1}^K \mathbb{1}(k = \hat{k}^m) \left[ \log \pi(a^k | x^m; \theta) + \sum_{t=1}^T \log \mathcal{N}(s_t^k | a_t^k + \mu_t^k, \Sigma_t^k; x^m; \theta) \right].
$$



#### Ensemble models: Multipath++

E ensemble models, for each with L modes > M (with redudancy) 

Select M modes from M' = EL predicted heads: 

* select M cluster centroids from in a greedy fashion
* iteratively update the parameters of M centraoids

[zhihu:](https://zhuanlan.zhihu.com/p/1892876738373060311)

TODO: exploding gradients... 



## References

- MultiPath++ (Waymo)
- Kalman, *Linear Filtering and Prediction*
- Li & Todorov, *iLQR*

