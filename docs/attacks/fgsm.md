# FGSM — Fast Gradient Sign Method

## Intuition

The Fast Gradient Sign Method is one of the simplest and most widely used adversarial attacks. The core idea is straightforward: compute the gradient of the loss function with respect to the input image, then perturb the image by a small amount in the direction that increases the loss the most. Because neural networks are trained to minimize loss, moving in the opposite direction (maximizing loss) tends to cause misclassification.

The key insight from Goodfellow et al. is that the linearity of neural networks in high-dimensional spaces makes them inherently vulnerable to small, carefully chosen perturbations.

## The Math

Given a model with parameters $\theta$, an input image $x$, a true label $y$, and a loss function $J(\theta, x, y)$, the adversarial example is computed as:

$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))$$

where $\epsilon$ controls the magnitude of the perturbation. The result is clipped to the valid pixel range $[0, 1]$.

## Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `epsilon` | Maximum perturbation magnitude (L-infinity norm) | 0.01 to 0.3 |

## Strengths and Limitations

FGSM is fast (single forward and backward pass) and effective against undefended models. However, it is a relatively weak attack because it takes only one step. Models defended with adversarial training can often resist FGSM perturbations. For a stronger iterative variant, see [PGD](pgd.md).

## Reference

> Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). "Explaining and Harnessing Adversarial Examples." *ICLR 2015*. [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)
