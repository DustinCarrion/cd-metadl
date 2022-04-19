# Cross-Domain MetaDL Challenge: Few-shot learning series
---
This repository contains the code associated to the Cross-Domain Meta-Learning challenge organized by:
- Dustin Carrión (U. Paris-Saclay; UPSud, France)
- Ihsan Ullah (U. Paris-Saclay; UPSud, France)
- Isabelle Guyon (U. Paris-Saclay; UPSud/INRIA, France and ChaLearn, USA)
- Sergio Escalera (U. Barcelona, Spain and ChaLearn, USA)
- Felix Mohr (U. de La Sabana, Colombia)
- Manh Hung Nguyen (ChaLearn, USA)

The CodaLab competition link will be available here.

## Outline 
[I - Overview](#i---overview)
[II - Installation](#ii---installation)
[III - References](#iii---references)

---

## I - Overview
The competition focus on any-way any-shot learning for image classification. This is an **online competition**, i.e. you need to provide your submission as raw Python code that will be ran on the CodaLab platform. The code is designed to be a module and to be flexible and allows participants to any type of meta-learning algorithms.

You can find more informations on the [ChaLearn website](https://metalearning.chalearn.org/).

## II - Installation

Make sure you first clone the repository. Then you can directly jump to the [Starting kit](starting_kit/README.md) to get started.

This repository can be installed via a **conda** environment, check the `starting_kit/quick_start.sh` script.

## III - References
- [1] - [E. Triantafillou **Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples** -- 2019](https://arxiv.org/pdf/1903.03096)
- [2] - [J. Snell et al. **Prototypical Networks for Few-shot Learning** -- 2017](https://arxiv.org/pdf/1703.05175)
- [3] - [C. Finn et al. **Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks** -- 2017](https://arxiv.org/pdf/1703.03400)
- [4] - [H-Y Tseng et al. **Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation** -- 2020](https://arxiv.org/abs/2001.08735)
- [5] - [Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. (2015). **Human-level concept learning through probabilistic program induction**.](http://www.sciencemag.org/content/350/6266/1332.short) Science, 350(6266), 1332-1338.

### Disclamer
Some methods in the `starting_kit/tutorial.ipynb` (e.g., plot_episode()) are inspired in the introduction notebook of the recent publication code [E. Triantafillou et al. **Meta-Dataset: GitHub repository**](https://github.com/google-research/meta-dataset).
