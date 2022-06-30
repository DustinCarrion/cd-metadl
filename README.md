# Cross-Domain MetaDL Competition: Any-way Any-shot Learning 
---
This repository contains the code associated to the [Cross-Domain MetaDL competition](https://codalab.lisn.upsaclay.fr/competitions/3627) organized by:
- Dustin Carrión (LISN/INRIA/CNRS, Université Paris-Saclay, France)
- Ihsan Ullah (LISN/INRIA/CNRS, Université Paris-Saclay, France)
- Sergio Escalera (Universitat de Barcelona and Computer Vision Center, Spain, and ChaLearn, USA)
- Isabelle Guyon (LISN/INRIA/CNRS, Université Paris-Saclay, France, and ChaLearn, USA)
- Felix Mohr (Universidad de La Sabana, Colombia)
- Manh Hung Nguyen (ChaLearn, USA)
- Joaquin Vanschoren (TU Eindhoven, the Netherlands)

## Outline 
[I - Overview](#i---overview)

[II - Tutorial](#ii---tutorial)

[III - References](#iii---references)

---

## I - Overview
This is the official repository of the [Cross-Domain MetaDL: Any-Way Any-Shot Learning Competition with Novel Datasets from Practical Domains presented in NeurIPS 2022 Competition Track](https://neurips.cc/Conferences/2022/CompetitionTrack).

The competition focuses on any-way any-shot learning for image classification. This is an **online competition with code submission**, *i.e.*, you need to provide your submission as raw Python code that will be executed on the CodaLab platform. The code is designed to be flexible and allows participants to explore any type of meta-learning algorithms.

You can find more informations on the [Official website](https://metalearning.chalearn.org).

## II - Tutorial

To follow the tutorial you may either clone or download this repository, or access the material on [Google Colab](https://colab.research.google.com/drive/1ek519iShqp27hW3xtRiIxmrqYgNNImun?usp=sharing). The tutorial notebook is organized as follows:  
* Beginner level (no prerequisite)
* Intermediate level (some knowledge of Python and meta-learning)
* Advanced level (solid knowledge of Python and meta-learning)

Each notebook level includes information of previous levels.

**Note:** The information in this repository is the same as the `starting_kit` that you can download from the [Competition Site](https://codalab.lisn.upsaclay.fr/competitions/3627#participate-get_starting_kit). Therefore, if you already download it from there, it is not necessary to clone or download this repository.

## III - References
- [1] - [J. Snell et al. **Prototypical Networks for Few-shot Learning** -- 2017](https://arxiv.org/pdf/1703.05175)
- [2] - [O. Vinyals et al. **Matching Networks for One Shot Learning** -- 2017](https://arxiv.org/pdf/1606.04080)
- [3] - [C. Finn et al. **Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks** -- 2017](https://arxiv.org/pdf/1703.03400)
- [4] - [E. Triantafillou et al. **Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples** -- 2019](https://arxiv.org/pdf/1903.03096)
- [5] - [Chen et al. **MetaDelta: A Meta-Learning System for Few-shot Image Classification** -- 2021](https://arxiv.org/pdf/2102.10744)
- [6] - [A. El Baz et al. **Lessons learned from the NeurIPS 2021 MetaDL challenge: Backbone fine-tuning without episodic meta-learning dominates for few-shot learning image classification** -- 2022](https://hal.archives-ouvertes.fr/hal-03688638)
- [7] - [I. Ullah et al. **Meta-Album: Multi-domain Meta-Dataset for Few-Shot Image Classification** -- 2022](https://meta-album.github.io/paper.html)


### Disclamer
Some methods in the `tutorial_utils.py` (*e.g.*, plot_task()) are inspired by the introduction notebook of [E. Triantafillou et al. **Meta-Dataset: GitHub repository**](https://github.com/google-research/meta-dataset).
