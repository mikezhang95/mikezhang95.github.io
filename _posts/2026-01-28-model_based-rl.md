---
layout: post
title: Model-based Reinforcement Learning
date: 2026-01-28
description: 
tags: [Blog, RL]
author: Yuan Zhang
giscus_comments: true
math: true
---

Model-based reinforcement learning (MBRL) aims to improve sample efficiency by explicitly learning a model of the environment dynamics and using it for planning, policy optimization, or both. Compared to model-free RL, MBRL introduces inductive bias and structure, at the cost of model bias and potential instability. This post surveys modern MBRL methods with an emphasis on *practical algorithmic designs* and *optimization choices* that make them work in real systems.

[Papers: World Models, Muzero, EfficientZero, PETS, MBPO, TDMPC, Dreamer]

---

## 1. Motivation: Why Model-Based RL?

In many real-world domains (robotics, autonomous driving, manipulation), environment interactions are expensive. MBRL attempts to answer:

> Can we learn a sufficiently accurate world model and exploit it to reduce real-world samples?

Benefits:
- Higher data efficiency
- Explicit dynamics modeling
- Natural integration with classical control

Challenges:
- Compounding model errors
- Distribution shift when policies exploit model inaccuracies
- Optimization instability when planning over learned dynamics

---

## 2. Taxonomy of Model-Based RL

### 2.1 Planning with Learned Models

Given Learn a dynamics model: $s_{t+1} = f_\theta(s_t, a_t)$

Then use **online planning** (e.g., Model Predictive Control) to select actions:
- Shooting methods
- Cross-Entropy Method (CEM)
- Random sampling with trajectory scoring

Representative methods:
- PILCO
- PETS (Probabilistic Ensembles with Trajectory Sampling)

Key idea: *do not train a policy directly; re-plan at every step.*

---

### 2.2 Model-Based Policy Optimization

Instead of pure planning, the learned model is used to **generate synthetic data** for training a policy.

Typical loop:
1. Collect real transitions
2. Train a dynamics model
3. Roll out the policy in the model (short horizon)
4. Update the policy using real + imagined data

Representative methods:
- MBPO
- STEVE

Key tradeoff: rollout horizon vs. model bias.

---

### 2.3 Latent World Models

High-dimensional observations (images) are difficult to model directly.

Latent world models:
- Learn an encoder \( z_t = e(s_t) \)
- Learn latent dynamics \( z_{t+1} = g(z_t, a_t) \)
- Train policy entirely in latent space

Representative methods:
- World Models
- Dreamer / DreamerV2 / DreamerV3

Advantages:
- Compact representations
- Improved generalization
- Scales to vision-based RL

---

## 3. Handling Model Uncertainty

### 3.1 Ensemble Models

Train multiple models \( \{f_{\theta_i}\} \) and:
- Sample a model per rollout
- Penalize uncertainty during planning
- Avoid overconfident predictions

Ensembles are one of the most effective and widely used tricks in practical MBRL.

---

### 3.2 Short-Horizon Rollouts

Long rollouts amplify small errors:
\[
\epsilon_{t+k} \approx O(\epsilon^k)
\]

MBPO shows that **many short rollouts** outperform **few long rollouts**.

---

## 4. Optimization and Planning

Common planners:
- Random shooting
- CEM (iteratively refines action distributions)
- Gradient-based optimization (less stable)

Hybrid designs:
- Learned value function + MPC
- Learned policy as proposal distribution for planning

---

## Quick Summary

| Issue                   | Mitigation                    |
| ----------------------- | ----------------------------- |
| Compounding error       | Short rollouts, ensembles     |
| Model exploitation      | Conservative policy updates   |
| High-dimensional inputs | Latent dynamics               |
| Training instability    | Regularization, replay mixing |

---

## References

- Chua et al., *Deep RL in a Handful of Trials using Probabilistic Dynamics Models*
- Janner et al., *When to Trust Your Model: MBPO*
- Hafner et al., *Dreamer: Learning Behaviors from Pixels*

---

## Changelog

2026-01-28: create the page



