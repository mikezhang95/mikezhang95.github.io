---
layout: post
title: Generative Models
date: 2026-01-29
description: 
img: 
tags: [Blog]
author: Yuan Zhang
giscus_comments: true
---



> Date: 2026-01-29 | Estimated Reading Time: 30 min | Author: Yuan Zhang

Generative modeling is a core building block across modern machine learning, from image synthesis to trajectory generation and world modeling. This post reviews two dominant paradigms: Generative Adversarial Networks (GANs) and Diffusion Models.

---

## 1. What Is a Generative Model?

Given data \( x \sim p_{data}(x) \), a generative model learns an approximation \( p_\theta(x) \) such that:
- Samples resemble real data
- The model covers the full data distribution

Evaluation is often qualitative and task-dependent.

---

## 2. Generative Adversarial Networks (GANs)

### 2.1 Core Idea

GANs train two networks:
- Generator \( G(z) \)
- Discriminator \( D(x) \)

Minimax objective:
\[
\min_G \max_D \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))]
\]

---

### 2.2 Strengths and Weaknesses

**Pros**
- Sharp, realistic samples
- Fast inference

**Cons**
- Mode collapse
- Training instability
- Hard to evaluate likelihood

---

### 2.3 Stabilization Techniques

- Wasserstein GAN (WGAN)
- Gradient penalty
- Spectral normalization
- Two-time-scale update rules (TTUR)

---

## 3. Diffusion Models

### 3.1 Forward Process

Gradually add noise:
\[
x_t = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon
\]

---

### 3.2 Reverse Process

Learn a denoiser:
\[
\epsilon_\theta(x_t, t)
\]

Training reduces to a simple MSE loss between true and predicted noise.

---

### 3.3 Advantages

- Stable training
- Strong mode coverage
- Excellent sample diversity

### 3.4 Drawbacks

- Slow sampling (iterative)
- High compute cost

---

## 4. Modern Variants

- Score-based models (SDE formulation)
- DDIM (fast deterministic sampling)
- Latent diffusion (operate in compressed space)
- Classifier-free guidance

---

## 5. GANs vs Diffusion Models

| Aspect             | GAN  | Diffusion        |
| ------------------ | ---- | ---------------- |
| Training stability | Low  | High             |
| Sample quality     | High | Very high        |
| Mode coverage      | Poor | Strong           |
| Sampling speed     | Fast | Slow (improving) |

---

## 6. Applications Beyond Images

- Trajectory generation
- World models
- Policy priors in RL
- Conditional planning

---

## 7. Key References

- Goodfellow et al., *GANs*
- Ho et al., *DDPM*
- Song et al., *Score-Based Generative Modeling*

---

## 8. Open Questions

- Diffusion for online control
- Distillation for real-time systems
- Combining diffusion with planning and RL
