---
layout: post
title: Autonomous Driving Algorithms
date: 2026-01-31
description: 
img: 
tags: [Blog]
author: Yuan Zhang
giscus_comments: true
math: true
---



> Date: 2026-01-31 | Estimated Reading Time: 30 min | Author: Yuan Zhang

Autonomous driving systems combine learning-based prediction with classical estimation and optimization. This post surveys key components commonly found in modern driving stacks.

---

## 1. Multimodal Trajectory Prediction

Driving behavior is inherently multimodal:
- Lane changes
- Turns vs. straight
- Yielding vs. asserting

Predictors must output *multiple plausible futures*.

---

## 2. MultiPath++

MultiPath++ is a state-of-the-art trajectory prediction model designed for efficiency and multimodality.

### Key Ideas
- Sparse polyline map encoding
- Agent-centric representation
- Learned latent anchors (instead of fixed anchors)
- Trajectory aggregation and ensembling

---

### 2.1 Loss Design

Typical objective:
- Predict K trajectories \( \{\tau_k\} \)
- Predict confidence scores \( \{p_k\} \)

Loss:
- Min-over-modes regression loss
- Cross-entropy / NLL over mode probabilities

This avoids penalizing correct but low-probability modes.

---

## 3. Kalman Filters

### 3.1 Linear Kalman Filter

System:
\[
x_{t+1} = A x_t + B u_t + w_t
\]

Kalman filter alternates:
- Prediction
- Measurement update

Used for:
- Sensor fusion
- State estimation
- Tracking agents

---

### 3.2 Extensions

- EKF: nonlinear dynamics
- UKF: sigma-point approximation
- Smoothing for offline estimation

---

## 4. iLQR (Iterative LQR)

iLQR solves nonlinear trajectory optimization by:
1. Linearizing dynamics
2. Quadraticizing cost
3. Solving local LQR
4. Rolling out updated trajectory

Efficient and widely used for:
- Motion planning
- MPC
- Low-level control

---

## 5. Learning + Optimization Stack

Typical pipeline:
1. Perception
2. State estimation (Kalman)
3. Prediction (MultiPath++)
4. Planning (iLQR / MPC)
5. Control

---

## 6. Practical Notes

- Learned predictors must be evaluated *in closed loop*
- Classical optimizers offer stability guarantees
- Hybrid systems dominate production autonomy

---

## 7. Key References

- MultiPath++ (Waymo)
- Kalman, *Linear Filtering and Prediction*
- Li & Todorov, *iLQR*

---

## 8. Open Challenges

- Prediction–planning coupling
- Uncertainty-aware planning
- End-to-end differentiable autonomy
