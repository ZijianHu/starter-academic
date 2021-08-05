---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "RL Overview #01: Introduction"
subtitle: ""
summary: "In this blog, some basic concepts of RL will be introduced. All materials refer to the RL course \"*Intro to Reinforcement Learning by Bolei Zhou*\", https://github.com/zhoubolei/introRL"
authors: []
tags: [Reinforcement Learning]
categories: [Reinforcement Learning]
date: 2021-08-04T14:56:51+08:00
lastmod: 2021-08-04T14:56:51+08:00
featured: false
draft: false
widget: pages
design:
  # Choose how many columns the section has. Valid values: '1' or '2'.
  columns: '1'
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

## Features of RL
+ Sequential data as input (not i.i.d)
+ The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them.
+ Trial-and-error exploration (balance between **exploration and exploitation**)
+ There is no supervisor, only a reward signal, which is also **delayed**.
+ Agent's actions affect the subsequent data it receives (agent's action changes the environment)

## Sequential Decision Making
### Observation 
#### Fully observability
Agent directly observes the environment state, formally as Markov decision process (MDP).

#### Partial observability
Agent indirectly observes the environment, formally as partially observable Markov decision process (POMDP).
+ Black jack or Texas hold'em
+ Atari game with pixel observation

### Action
The interaction between agents and the environment.

### Reward
+ Reward is a scalar feedback signal.
+ Indicate how well agent is doing at step t.
+ Reinforcement Learning is based on the maximization of rewards.

All goals of the agent can be described by the maximization of expected cumulative reward.

## Major components of an RL Agent
### Policy
A policy is the agent's behavior model. It is a map function from state/observation to action.
+ Stochastic policy: $\pi(a \vert s) = P\left[ A_t = a \vert S_t = s \right]$
+ Deterministic policy: $a^* = \arg\max_a \pi\left( a \vert s \right)$

### Value function
Expected discounted sum of future rewards under a particular policy $\pi$. Used to quantify goodness or badness of states and actions.
+ state value function: $v_{\pi} = \mathbb{E}_{\pi} \left[ G_t \vert S_t = s \right] = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\inf} \gamma^k R_{t+k+1} \vert S_t = s \right], \forall s \in \mathcal{S}$
+ action-state value function (quality function, a.k.a Q function): $q_{\pi}(s, a) = \mathbb{E}_{\pi} \left[ G_t \vert S_t = s, A_t = a \right] = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\inf} \gamma^k R_{t+k+1} \vert S_t = s, A_t = a \right]$

### Model
A model predicts what the environment will do next

+ Predict the next state (state transition matrix): $P_{ss'}^a = \mathbb{P} \left[ S_{t+1} = s' \vert S_t = s, A_t = a \right]$
+ Predict the next reward: $R_s^a = \mathbb{E} \left[ R_{t+1} \vert S_t = s, A_t = a \right]$

## Types of RL
### Agent
+ Value-based agent:
    + Explicit: Value function
    + Implicit: Policy (can derive a policy form value function).
+ Policy-based agent:
    + Explicit: policy
    + Novalue function
+ Actor-Critic agent:
    + Explicit: policy and value function

### Model
+ Model-based
    + Explicit: model
    + May or may not have policy and/or value function
+ Model-free
    + Explicit: value function and/or policy function
    + No model


## Exploration and Exploitation
### Exploration
Trying new things that might enable the agent to make better decisions in the future.

### Exploitation
Choosing actions that are expected to yield good reward given the past experience.
