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

Given data $$x \sim p_\text{data}(x) $$, a generative model learns an approximation $$p_\theta(x)$$ such that: 
- Samples resemble real data
- The model covers the full data distribution.

<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/generative-overview.png" alt="generative-overview" style="zoom: 50%;" />

---

## 2. Variational Auto-Encoder (VAE)

---

## 3. Generative Adversarial Networks (GANs) [1]

### 3.1 Core Idea

GANs train two networks:
- A generator $$G$$ generates synthetic samples ($$x = G(z)$$), given a noise variable input $$z$$ who usually follows a standard normal distribution and introducespotential output diversity. It is trained to trick the discriminator to offer a high probability. 
- Discriminator $$D$$ outputs the **probability** of a given sample coming from the real dataset \($$D(x)$$\). It is trained to distinguish the fake samples from the real ones. 

<img src="https://lilianweng.github.io/posts/2017-08-20-gan/GAN.png" alt="GAN" style="zoom:50%;" />

These two models compete against each other during the training process: the generator $$G$$ is trying hard to trick the discriminator, while the critic model $$D$$ is trying hard not to be cheated, which forms a zero-sum game. 

Suppose $$p_r,p_g,p_z$$ distributions over the real sample $$x$$, generated sample $$x$$ and random noise $$z$$ (usually a standard normal distribution). The minimax objective is like: 


$$
\begin{aligned}
\min_G \max_D L(G, D) = \mathbb{E}_{x \sim p_r}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))] \\
=  \mathbb{E}_{x \sim p_r}[\log D(x)] + \mathbb{E}_{x \sim p_g}[\log(1 - D(x)].
\end{aligned}
$$



### 3.2 What is the optimal value for $$D$$? 

Examinte the best value for $$D$$ by calulating the stationary point:    


$$
D^*(x) = \frac{p_r(x)}{p_r(x) + p_g(x)}.
$$



### 3.3 What does the loss function represent? 

Given the optimal discriminator $$D^*$$, find the relations between real and synthetic data distributions $$p_r$$ and $$p_g$$: 


$$
\begin{aligned}
L(G, D^*) = \mathbb{E}_{x \sim p_r}[\log \frac{p_r(x)}{p_r(x)+p_g(x)}] + \mathbb{E}_{x \sim p_g}[\log \frac{p_g(x)}{p_r(x)+p_g(x)}]  \\
				= D_{KL}(p_r\|\frac{p_r+p_g}{2}) - \log2 + D_{KL}(p_g\|\frac{p_r+p_g}{2}) - \log2   \\
				= 2D_{JS}(p_r \| p_g) - \log4.
\end{aligned}
$$

### 3.4 TODO: Problems in GANs

* Hard to achieve Nash equilibrium
* Low dimensional supports
* Vanishing gradient
* Mode collapse
* Lack of a proper evaluation metric (unknown likelihood of $$p_g$$)

### 3.5 Wasserstein GAN (WGAN) [2]

[Wasserstein Distance](https://en.wikipedia.org/wiki/Wasserstein_metric) is a measure of the distance between two probability distributions, which can be interpreted as the minimum energy cost of moving and transforming a pile of dirt in the shape of one probability distribution to the shape of the other distribution. When dealing with the continuous probability domain, the distance formula becomes:      


$$
W(p_r, p_g) = \inf_{\gamma \sim \Pi(p_r, p_g)} \mathbb{E}_{(x_1,x_2) \sim \gamma}[ \|x_1 - x_2\| ], \\\\
$$


where $$\tau(x_1,x_2)$$ states the joint distribution that satisfies the boundary conditions of $$\int_{x_2} \gamma(x_1,x_2)dx_2 = p_r(x_1)$$ and $$\int_{x_1} \gamma(x_1,x_2)dx_1 = p_g(x_2)$$.     

#### TODO: Why Wasserstein is better than JS or KL divergence?

<img src="https://lilianweng.github.io/posts/2017-08-20-gan/wasserstein_simple_example.png" alt="wasserstein_simple_example" style="zoom:50%;" />

#### Use Wasserstein distance as GAN loss function

It is intractable to exhaust all the possible joint distributions in $$\Pi(p_r, p_g)$$ to compute $$\inf_{\gamma\sim \Pi(p_r, p_g)}$$ . Thus the authors proposed a smart transformation of the formula based on the **Kantorovich-Rubinstein duality** to:


$$
W(p_r, p_g) = \frac{1}{K} \sup_{\|f\| \le K} \mathbb{E}_{x\sim p_r}[f(x)] - \mathbb{E}_{x \sim p_g} [f(x)],
$$


where the real-valued function $$f: \mathbb{R} \to \mathbb{R}$$ should be K-Lipschitz continuous. Read this [blog](https://vincentherrmann.github.io/blog/wasserstein/) for more about this duality transformation. In the modified Wasserstein GAN, the “discriminator” model is used to find a **good** function $$f_w$$ parametrized by $$w \in W$$ and the loss function is configured as measuring the Wasserstein distance between $$p_r$$ and $$p_g$$:


$$
L(p_r,p_g) = W(p_r, p_g) = \max_{w \in W} \mathbb{E}_{x\sim p_r}[f_w(x)] - \mathbb{E}_{x \sim p_g} [f_w(x)].
$$


To maintain the K0Lipschitz continuity of function $$f_w$$ during training, the paper presents a simple but very practical trick: After every gradient update, clamp the weights $$w$$ to a small window, such as $$[-0.01, 0.01]$$. 

### 3.6 Connections to Actor-Critic Methods [3]

#### The Mathematical Bridge: Bilevel Optimization


$$
\begin{aligned}
x^* = \arg \min_x F(x, y^*(x)) \quad\text{(Outer Opt.)} \\
y^*(x) = \arg \min_y f(x, y) \quad\text{(Inner Opt.)}.
\end{aligned}
$$



| Component               | GANs                                      | Actor-Critic Methods                                         |
| :---------------------- | :---------------------------------------- | :----------------------------------------------------------- |
| **Outer Model (x)**     | Discriminator ($$D$$)                     | Critic ($$Q$$-function)                                      |
| **Outer Objective (F)** | $$-L(G, D)$$                              | $$-\mathbb{E}_{s,a}[D_{KL}(\mathbb{E}_{r,s',a'}[r+\gamma Q(s', a')]\|Q (s, a)]$$ |
| **Inner Model (y)**     | Generator ($$G$$)                         | Actor (Policy $$\pi$$)                                       |
| **Inner Objective (f)** | $$-\mathbb{E}_{z\sim p_z}[\log D(G(z))]$$ | $$-\mathbb{E}_{s \sim \mu,a\sim\pi}[Q(s, a)]$$               |

​          

#### GANs as a kind of Actor-Critic

For an RL environment with: 

* State: stateless
* Action: generates an entire image
* Environment step: randomly decides to show either a real image of a generated image
* Reward: +1 if a real image is shown, 0 if the generated image is shown

reward/state does not depend on action.

---

## 4. Diffusion Models

### 4.1 Forward Diffusion Process: From Data to Noise

Given a data point sampled from a real data distribution $$x_0 \sim p_r(x)$$, a forward diffusion process, we can gradually add small amount of Gaussian noise in $$T$$ steps, producing a sequence os noisy samples $$x_1,\dots,x_T$$. The step sizes are controlled by a variance schedule $${\beta_t \in (0,1)}_{t=1}^T$$:


$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I}), q(x_{0:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1}).
$$


When $$T\to \infty$$, $$x_T$$  is equivalent to an isotropic Gaussian distribution. 

#### Reparametrization tricks: from $$x_0$$ to $$x_t$$ 


$$
\begin{aligned}
x_t = \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t} \epsilon_{t-1} \quad\text{;where $\epsilon_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$} \\
=  \sqrt{(1-\beta_t)(1-\beta_{t-1})}x_{t-2} + \sqrt{1-(1-\beta_t)(1-\beta_{t-1})} \bar{\epsilon}_{t-2} \quad\text{;where $\bar{\epsilon}_{t-2}$ merges two Gaussians} \\
= \dots \\
= \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon \quad\text{;where $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$}\\
q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)\mathbf{I}).
\end{aligned}
$$



#### TODO: Connections with stochasitic gradient Langevin dynamics


$$
x_t = x_{t-1} + \delta/2 \nabla_x \log p(x_{t-1}) + \sqrt{\delta}\epsilon_t, \quad\text{where $\epsilon_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$}.
$$



### 4.2 Reverse Diffusion Process: From Noise to Data

If we can reverse the above process and sample from $$p_{\theta}(x_{t-1}\vert x_t)$$, we will be able to recreate the true sample from a Gaussian noise input, $$x_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$: 


$$
p_{\theta}(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\mu_{\theta}(x_t, t), \Sigma_\theta(x_t, t)), p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1}|x_t).
$$



#### Distribution Representation 1: from $$x_t, x_0$$ to $$x_{t-1}$$ 

It is noteworthy that the reverse conditional probability is tractable when conditioned on real sample $$x_0$$ :


$$
\begin{aligned}
q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1};\tilde{\mu}(x_t, x_0), \tilde{\beta}_t\mathbf{I}), \\
\tilde{\mu}(x_t, x_0) = \frac{\sqrt{1-\beta_t}( 1- \bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}  x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} x_0 \\
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t. 
\end{aligned}
$$

#### Distribution Representation 2: from $$x_t$$ to $$x_{t-1}$$ 

From the reparametrization tricks above, we can replace $$x_0$$ with $$x_t$$:  


$$
\begin{aligned}
\tilde{\mu}(x_t) = \frac{\sqrt{1-\beta_t}( 1- \bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}  x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\frac{1}{\sqrt{\bar{\alpha}_t}} (x_t -  \sqrt{1-\bar{\alpha}_t} \epsilon \\
 = \frac{1}{\sqrt{1-\beta_t}} (x_t  - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon) \quad\text{;where $\bar{\alpha}_{t} = \bar{\alpha}_{t-1}(1-\beta_t)$} 
 \end{aligned}
$$


The above derivations can be achieved by Bayesian rule, see this [blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) for details. We want to use a parametrized function $$\mu_{\theta}(x_t, t)$$ to represent $$\tilde{\mu}(x_t)$$. Since $$x_t$$ is known during training, we only need to learn to predict noises $$\epsilon$$ with $$\epsilon_\theta(x_t, t)$$.

### 4.3 Training and Sampling Process

TODO: dervie DDPM loss:


$$
\begin{aligned}
L_\text{DDPM} = \mathbb{E}_{x_0, t \sim [1, T], \epsilon}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2 ] \\
					= \mathbb{E}_{x_0, t \sim [1, T], \epsilon}[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)\|^2 ] \\
\end{aligned}
$$


<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM-algo.png" alt="DDPM-algo" style="zoom: 33%;" />



## 5. TODO: Flow-Matching Models

If we make the diffusion step $$\Delta t \to 0$$ and forward process from $$x_0 \to x_1$$ this becomes a continuous-time diffusion process.

#### Connections with stochasitic diffential equation (SDE)

The same as a stochasitc differential equation (SDE): 


$$
dx = f(x,t)dt + g(t)dw
$$


It has a corresponding ordinary differential equation (ODE) inducing the same $$p_t(x)$$:


$$
\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)
$$


Discretization this leads to “Denoising Diffusion Implicit Models” (DDIM) 

Flow matching (FM)

This is the continuous version of Langevin dynamics function

---

## References

[1] Goodfellow, Ian, et al. [“Generative Adversarial Nets.”](https://arxiv.org/pdf/1406.2661.pdf) Neurips, 2014.

[2] Martin Arjovsky, Soumith Chintala, and Léon Bottou. [“Wasserstein GAN.”](https://arxiv.org/pdf/1701.07875.pdf) ICML, 2017. 

[3] David Pfau, Oriol Vinyals. [“Connecting Generative Adversarial Networks and Actor-Critic Methods.”]() arXiv, 2016.

[4] Calvin Luo. [“Understanding Diffusion Models: A Unified Perspective.”](https://arxiv.org/pdf/2208.11970) arXiv, 2022.







