# sleight

## Overview
Python-based framework designed to simulate adversarial attacks on machine learning models and implement defense mechanisms to improve model robustness.

## Usage Instructions

### 1. Install Requirements
```
pip install -r requirements.txt
```

### 2. Run Example Notebooks
- **FGSM Attack Demo:** See `notebooks/FGSM_Attack.ipynb` for a demonstration of the Fast Gradient Sign Method (FGSM) attack on MNIST.
- **PGD Adversarial Training:** See `notebooks/PGD_Adversarial_Training.ipynb` for adversarial training using the Projected Gradient Descent (PGD) attack.

### 3. Using Attacks and Defenses in Code

#### FGSM Attack Example
```python
from sleight.attacks.fgsm_attack import fgsm_attack
# model: trained tf.keras.Model
# images: input images
# labels: one-hot or categorical labels
adv_images = fgsm_attack(model, images, labels, epsilon=0.1)
```

#### PGD Attack Example
```python
from sleight.attacks.pgd_attack import pgd_attack
adv_images = pgd_attack(model, images, labels, epsilon=0.1, alpha=0.01, num_iter=40)
```

#### Adversarial Training Example
```python
from sleight.defenses.adversarial_training import adversarial_train
# Define a wrapper for the attack function
attack_fn = lambda model, images, labels, epsilon: pgd_attack(model, images, labels, epsilon=epsilon, alpha=0.01, num_iter=40)
robust_model = adversarial_train(model, x_train, y_train, attack_fn, epsilon=0.1, epochs=3, batch_size=64)
```

### 4. Customization
- You can implement and add new attacks in `sleight/attacks/` and new defenses in `sleight/defenses/` following the provided examples.

---
For more details, see the example notebooks in the `notebooks/` directory.