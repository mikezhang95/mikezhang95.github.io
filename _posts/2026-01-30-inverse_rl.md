# Imitation Learning and Inverse Reinforcement Learning

> Date: 2026-01-30 | Estimated Reading Time: 30 min | Author: Yuan Zhang

Imitation Learning (IL) and Inverse Reinforcement Learning (IRL) address the problem of learning behavior from expert demonstrations, especially when defining an explicit reward function is difficult.

---

## 1. Problem Setup

Given expert trajectories:
\[
\tau_E = \{(s_t, a_t)\}
\]

Goal:
- Learn a policy \( \pi(a|s) \) (IL)
- Or infer a reward function \( r(s,a) \) (IRL)

---

## 2. Behavioral Cloning (BC)

BC treats imitation as supervised learning:
\[
\min_\theta \mathbb{E}_{(s,a)\sim \tau_E} [\| \pi_\theta(s) - a \|^2]
\]

### Pros
- Simple
- Stable

### Cons
- Covariate shift
- Error accumulation

---

## 3. Inverse Reinforcement Learning

IRL assumes:
> The expert is (near-)optimal under an unknown reward.

Classic IRL:
- Maximum entropy IRL
- Feature matching

Main issue:
- Reward ambiguity (reward shaping equivalence)

---

## 4. Adversarial Imitation Learning

### 4.1 GAIL

GAIL learns a discriminator:
\[
D(s,a)
\]

The policy is trained to fool the discriminator, similar to GANs.

Interpretation:
- Implicit reward learning
- Avoids explicit reward engineering

---

### 4.2 AIRL

AIRL introduces a structured reward:
\[
r(s,a) = f_\theta(s,a) + \gamma h(s') - h(s)
\]

Benefits:
- Reward transferability
- Better interpretability

---

## 5. Practical Training Strategies

- BC warm-start + GAIL fine-tuning
- Off-policy adversarial IL
- Hybrid IL + RL pipelines

---

## 6. When to Use IRL?

Good fit:
- Reward is ambiguous
- Transfer across environments is required

Poor fit:
- Dense, well-defined rewards
- Limited expert data

---

## 7. Key References

- Ho & Ermon, *GAIL*
- Fu et al., *AIRL*
- Ziebart, *Maximum Entropy IRL*

---

## 8. Open Problems

- Sample-efficient IL
- Multi-agent imitation
- Foundation models for IL
