# Adversarial Training

## Intuition

Adversarial training is the most straightforward defense against adversarial attacks: include adversarial examples in the training set. During each training step, the defender generates adversarial perturbations of the current batch using an attack algorithm (typically PGD), then trains the model on both the clean and adversarial examples. Over time, the model learns to classify correctly even when inputs are perturbed.

This approach frames robustness as a min-max optimization problem: the attacker tries to maximize the loss, and the defender trains to minimize the worst-case loss.

## The Math

Adversarial training solves the following optimization problem:

$$\min_\theta \mathbb{E}_{(x,y) \sim D} \left[ \max_{\delta \in \mathcal{S}} J(\theta, x + \delta, y) \right]$$

where $\mathcal{S} = \{\\delta : \|\\delta\|_\infty \leq \epsilon\}$ is the allowed perturbation set. In practice, the inner maximization is approximated by running PGD for a fixed number of steps.

## Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `attack_fn` | The attack function used to generate adversarial examples | PGD is recommended |
| `epsilon` | Perturbation budget passed to the attack | 0.1 to 0.3 for MNIST |
| `epochs` | Number of training epochs | 3 to 20 |
| `batch_size` | Training batch size | 32 to 128 |

## Strengths and Limitations

Adversarial training is the most empirically effective defense known for L-infinity robustness. However, it is computationally expensive (each training step requires running the attack), and it typically reduces clean accuracy by a few percentage points. The resulting model is robust primarily against the specific attack and perturbation budget used during training.

## Reference

> Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." *ICLR 2018*. [arXiv:1706.06083](https://arxiv.org/abs/1706.06083)
