# PGD — Projected Gradient Descent

## Intuition

Projected Gradient Descent is an iterative extension of FGSM. Instead of taking one large step in the gradient direction, PGD takes many small steps, projecting the result back onto the allowed perturbation set (the epsilon-ball around the original image) after each step. This makes PGD significantly stronger than FGSM and is widely considered the standard "first-order" adversarial attack.

Madry et al. showed that PGD is the strongest attack using only first-order gradient information, making it the natural benchmark for evaluating adversarial robustness.

## The Math

Starting from $x_0 = x$ (or a random point within the epsilon-ball), PGD iterates:

$$x_{t+1} = \Pi_{x + \mathcal{S}} \left( x_t + \alpha \cdot \text{sign}(\nabla_{x_t} J(\theta, x_t, y)) \right)$$

where $\alpha$ is the step size, $\mathcal{S}$ is the set $\{\\delta : \|\\delta\|_\infty \leq \epsilon\}$, and $\Pi$ denotes projection (clipping) onto that set. The final result is also clipped to $[0, 1]$.

## Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `epsilon` | Maximum perturbation magnitude (L-infinity norm) | 0.01 to 0.3 |
| `alpha` | Step size per iteration | epsilon / 4 to epsilon / 10 |
| `num_iter` | Number of iterations | 7 to 100 |

## Strengths and Limitations

PGD is the gold standard for evaluating adversarial robustness under L-infinity threat models. It is more effective than FGSM but slower due to the iterative nature. The attack strength depends on the number of iterations and step size — more iterations generally produce stronger adversarial examples at the cost of computation time.

## Reference

> Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." *ICLR 2018*. [arXiv:1706.06083](https://arxiv.org/abs/1706.06083)
