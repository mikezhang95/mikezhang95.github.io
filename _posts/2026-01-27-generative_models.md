---
layout: post
title: Generative Models
date: 2026-01-27
description: 
img: 
tags: [Blog]
author: Yuan
giscus_comments: true
math: true
---

Generative modeling is a core building block across modern machine learning, from image synthesis to trajectory generation and world modeling. 

This post reviews two dominant paradigms: generative adversarial networks (GANs) and diffusion models, and their variants.

---

## 1. What Is a Generative Model?

Given data $x \sim p_\text{data}(x) $, a generative model learns an approximation $p_\theta(x) $ such that: 
- Samples resemble real data
- The model covers the full data distribution

Evaluation is often qualitative and task-dependent.

---

## 2. Variational Auto-Encoder

---

## 3. Generative Adversarial Networks (GANs) [1]

### 3.1 Core Idea

GANs train two networks:
- A generator $G$ generates synthetic samples ($x = G(z)$), given a noise variable input $z$ who usually follows a standard normal distribution and introducespotential output diversity. It is trained to trick the discriminator to offer a high probability. 
- Discriminator $D$ outputs the **probability** of a given sample coming from the real dataset \($D(x)$\). It is trained to distinguish the fake samples from the real ones. 

<img src="https://lilianweng.github.io/posts/2017-08-20-gan/GAN.png" alt="GAN" style="zoom:50%;" />

These two models compete against each other during the training process: the generator $G$ is trying hard to trick the discriminator, while the critic model $D$ is trying hard not to be cheated, which forms a zero-sum game. 

Suppose $p_r,p_g,p_z$ distributions over the real sample $x$, generated sample $x$ and random noise $z$ (usually a standard normal distribution). The minimax objective is like: 
$$
\min_G \max_D L(G, D) = \mathbb{E}_{x \sim p_r}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))] \\
=  \mathbb{E}_{x \sim p_r}[\log D(x)] + \mathbb{E}_{x \sim p_g}[\log(1 - D(x)].
$$

### 3.2 What is the optimal value for $D$? 

Examinte the best value for $D$ by calulating the stationary point: 
$$
D^*(x) = \frac{p_r(x)}{p_r(x) + p_g(x)}.
$$

### 3.3 What does the loss function represent? 

Given the optimal discriminator $D^*$, find the relations between real and synthetic data distributions $p_r$ and $p_g$ 
$$
L(G, D^*) = \mathbb{E}_{x \sim p_r}[\log \frac{p_r(x)}{p_r(x)+p_g(x)}] + \mathbb{E}_{x \sim p_g}[\log \frac{p_g(x)}{p_r(x)+p_g(x)}]  \\ 
				= D_{KL}(p_r\|\frac{p_r+p_g}{2}) - \log2 + D_{KL}(p_g\|\frac{p_r+p_g}{2}) - \log2   \\
				= 2D_{JS}(p_r \| p_g) - \log4.
$$


### 3.4 TODO: Problems in GANs

* Hard to achieve Nash equilibrium
* Low dimensional supports
* Vanishing gradient
* Mode collapse
* Lack of a proper evaluation metric (unknown likelihood of $p_g$)

### 3.5 Wasserstein GAN (WGAN) [2]

[Wasserstein Distance](https://en.wikipedia.org/wiki/Wasserstein_metric) is a measure of the distance between two probability distributions, which can be interpreted as the minimum energy cost of moving and transforming a pile of dirt in the shape of one probability distribution to the shape of the other distribution. When dealing with the continuous probability domain, the distance formula becomes: 
$$
W(p_r, p_g) = \inf_{\gamma \sim \Pi(p_r, p_g)} \mathbb{E}_{(x_1,x_2) \sim \gamma}[ \|x_1 - x_2\| ],
$$
where $\tau(x_1,x_2)$ states the joint distribution that satisfies the boundary conditions of $\int_{x_2} \gamma(x_1,x_2)dx_2 = p_r(x_1)$ and $\int_{x_1} \gamma(x_1,x_2)dx_1 = p_g(x_2)$. 

#### TODO: Why Wasserstein is better than JS or KL divergence?

<img src="https://lilianweng.github.io/posts/2017-08-20-gan/wasserstein_simple_example.png" alt="wasserstein_simple_example" style="zoom:50%;" />

#### Use Wasserstein distance as GAN loss function

It is intractable to exhaust all the possible joint distributions in $\Pi(p_r, p_g)$to compute $\inf_{\gamma\sim \Pi(p_r, p_g)}$ . Thus the authors proposed a smart transformation of the formula based on the **Kantorovich-Rubinstein duality** to:
$$
W(p_r, p_g) = \frac{1}{K} \sup_{\|f\| \le K} \mathbb{E}_{x\sim p_r}[f(x)] - \mathbb{E}_{x \sim p_g} [f(x)],
$$
where the real-valued function $f: \mathbb{R} \to \mathbb{R}$ should be K-Lipschitz continuous. Read this [blog](https://vincentherrmann.github.io/blog/wasserstein/) for more about this duality transformation. In the modified Wasserstein GAN, the “discriminator” model is used to find a **good** function $f_w$ parametrized by $w \in W$ and the loss function is configured as measuring the Wasserstein distance between $p_r$ and $p_g$:
$$
L(p_r,p_g) = W(p_r, p_g) = \max_{w \in W} \mathbb{E}_{x\sim p_r}[f_w(x)] - \mathbb{E}_{x \sim p_g} [f_w(x)].
$$
To maintain the K0Lipschitz continuity of function $f_w$ during training, the paper presents a simple but very practical trick: After every gradient update, clamp the weights $w$ to a small window, such as $[-0.01, 0.01]$. 

### 3.6 Connections to Actor-Critic Methods [3]

#### The Mathematical Bridge: Bilevel Optimization

$$
x^* = \arg \min_x F(x, y^*(x)) \quad\text{(Outer Opt.)} \\
y^*(x) = \arg \min_y f(x, y) \quad\text{(Inner Opt.)}.
$$

| Component               | GANs                                    | Actor-Critic Methods                                         |
| :---------------------- | :-------------------------------------- | :----------------------------------------------------------- |
| **Outer Model (x)**     | Discriminator ($D$)                     | Critic ($Q$-function)                                        |
| **Outer Objective (F)** | $-L(G, D)$                              | $-\mathbb{E}_{s,a}[D_{KL}(\mathbb{E}_{r,s',a'}[r+\gamma Q(s', a')]\|Q (s, a)]$ |
| **Inner Model (y)**     | Generator ($G$)                         | Actor (Policy $\pi$)                                         |
| **Inner Objective (f)** | $-\mathbb{E}_{z\sim p_z}[\log D(G(z))]$ | $-\mathbb{E}_{s \sim \mu,a\sim\pi}[Q(s, a)]$                 |

#### GANs as a kind of Actor-Critic

For an RL environment with: 

* State: stateless
* Action: generates an entire image
* Environment step: randomly decides to show either a real image of a generated image
* Reward: +1 if a real image is shown, 0 if the generated image is shown

reward/state does not depend on action.

---

## 4. Diffusion Models

### 4.1 Forward Process

Gradually add noise:
\[
x_t = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon
\]

---

### 4.2 Reverse Process

Learn a denoiser:
\[
\epsilon_\theta(x_t, t)
\]

Training reduces to a simple MSE loss between true and predicted noise.

---

### 4.3 Advantages

- Stable training
- Strong mode coverage
- Excellent sample diversity

### 4.4 Drawbacks

- Slow sampling (iterative)
- High compute cost

---

## References

[1] Goodfellow, Ian, et al. [“Generative Adversarial Nets.”](https://arxiv.org/pdf/1406.2661.pdf) Neurips, 2014.

[2] Martin Arjovsky, Soumith Chintala, and Léon Bottou. [“Wasserstein GAN.”](https://arxiv.org/pdf/1701.07875.pdf) ICML, 2017. 

[3] David Pfau, Oriol Vinyals. [“Connecting Generative Adversarial Networks and Actor-Critic Methods.”]() arXiv, 2016

 



