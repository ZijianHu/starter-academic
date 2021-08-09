---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "Deep Q-Learning"
subtitle: ""
summary: "In this blog, DQN and its improvment verions will be introduced. All materials refer ot the RL cource \"Intro to Reinforcement Learning by Bolei Zhou\", https://github.com/zhoubolei"
authors: [Zijian Hu]
tags: [Reinforcement Learning]
categories: [Reinforcement Learning]
date: 2021-08-10T00:20:12+08:00
lastmod: 2021-08-10T00:20:12+08:00
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

## Deep Q networks
+ Two of the issues in Q learning with Vlaue Function Approximation
    + Correlations between samples
    + Non-stationary targets
+ Deep Q-learning (DQN) addresses both of these challenges by
    + Experience replay
    + Fixed Q targets

### Experience Replay
**Objective**: To reduce the correlations among samples, store transition $(s_t, a_t, r_t, s_{t+1})$ in replay memory $D$.
![image info](assets/er.png)

**Solution**:
To perform experience replay, repeat the following:
+ Sample an experience tuple from the dataset: $(s, a, r, s') \sim D$
+ Compute the target value for the sampled tuple: $r + \gamma \max_{a'} Q(s', a', \omega)$
+ Use stochastic gradient descent to update the network weights $$\Delta \omega = \alpha (R_{t+1} + \gamma \max_a \hat{q}(s_{t+1}, a, \omega) - \hat{q}(s_t, a_t, \omega)) \nabla_w \hat{q}(s_t, a_t, \omega)$$

### Fixed Targets
**Objective**: To help improve stability.
![image info](assets/target_1.png)
![image info](assets/target_2.png)
![image info](assets/target_3.png)
fix the target weights used in the target calculation for multiple updates

**Solution**: Fix the target weights used in the target calculation for multiple updates
+ Let a different set of parameter $\omega^-$ be the set of weights used in the target, and $\omega$ be the weights that are being updated.
+ To perfomr experience replay with fixed target, repeat the following
    + Sample an experience tuple from the dataset: $(s, a, r, s') \sim D$
    + Compute the target value for the sampled tuple: $r + \gamma max_{a'} \hat{Q}(s', a', \omega^-)$
    + Use stochastic gradient decent to update the network weights $$\Delta \omega = \alpha (R_{t+1} + \gamma \max_a \hat{q}(s_{t+1}, a, \omega^-) - \hat{q}(s_t, a_t, \omega)) \nabla_w \hat{q}(s_t, a_t, \omega)$$