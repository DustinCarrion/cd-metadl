# Cross-Domain MetaDL Challenge: Few-shot learning series
---
This repository contains the code associated to the Cross-Domain MetaDL challenge organized by:
- Dustin Carrión (U. Paris-Saclay; UPSud, France)
- Ihsan Ullah (U. Paris-Saclay; UPSud, France)
- Isabelle Guyon (U. Paris-Saclay; UPSud/INRIA, France and ChaLearn, USA)
- Sergio Escalera (U. Barcelona, Spain and ChaLearn, USA)
- Felix Mohr (U. de La Sabana, Colombia)
- Manh Hung Nguyen (ChaLearn, USA)

[CodaLab competition link](https://codalab.lisn.upsaclay.fr/competitions/3627?secret_key=2d7c4b66-afa5-4c15-92cb-552f8187245c)

## Outline 
[I - Overview](#i---overview)

[II - Installation](#ii---installation)

[III - References](#iii---references)

---

## I - Overview
This is the official repository of the [Cross-Domain MetaDL: Any-Way Any-Shot Learning Competition with Novel Datasets from Practical Domains presented in NeurIPS 2022 Competition Track](https://neurips.cc/Conferences/2022/CompetitionTrack).

The competition focus on any-way any-shot learning for image classification. This is an **online competition**, *i.e.*, you need to provide your submission as raw Python code that will be ran on the CodaLab platform. The code is designed to be flexible and allows participants to explore any type of meta-learning algorithms.

You can find more informations on the [Official website](https://metalearning.chalearn.org).

## II - Installation

Make sure you first clone the repository. Then you can directly jump to the [Starting kit](starting_kit/README.md) to get started.

We provide 2 different ways of installing the repository.

* Via a conda environment
* Via a Docker image

Follow the `README.md` in either case.

## III - References
- [1] - [E. Triantafillou **Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples** -- 2019](https://arxiv.org/pdf/1903.03096)
- [2] - [J. Snell et al. **Prototypical Networks for Few-shot Learning** -- 2017](https://arxiv.org/pdf/1703.05175)
- [3] - [O. Vinyals et al. **Matching Networks for One Shot Learning** -- 2017](https://arxiv.org/abs/1606.04080)
- [4] - [C. Finn et al. **Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks** -- 2017](https://arxiv.org/pdf/1703.03400)

### Disclamer
Some methods in the `starting_kit/tutorial.ipynb` (*e.g.*, plot_task()) are inspired in the introduction notebook of the recent publication code [E. Triantafillou et al. **Meta-Dataset: GitHub repository**](https://github.com/google-research/meta-dataset).
