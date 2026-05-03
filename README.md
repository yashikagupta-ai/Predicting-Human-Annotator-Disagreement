# Predicting Human Annotator Disagreement

**3rd Year Deep Neural Networks Course Project**

*Learning what makes images inherently ambiguous*

Based on: Peterson et al., ICCV 2019 — *"Human Uncertainty Makes Classification More Robust"*

---

## Project Overview

This project builds a deep neural network that predicts the **full distribution of human annotator labels** for CIFAR-10 images, rather than just a single hard label. The model outputs a 10-class probability distribution representing how 50+ humans would vote on each image.

A standard classifier asks: *"What is this image?"*  
Our model asks: *"What will 50 different humans think this image is, and how much will they disagree?"*

**Example output:** `[cat: 0.60, dog: 0.25, deer: 0.10, horse: 0.05, others: 0.00]`

---

## Notebooks

| Notebook | Description | Link |
|----------|-------------|------|
| **NB1** | Data Loading & Exploration — downloads datasets, creates splits, computes entropy, generates visualizations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-UrDjfEyu9ErNxMCb40zv_BsUppp1Wy8?usp=sharing) |
| **NB2** | Model Training — ResNet-18 adapted for 32×32 images, pretraining + fine-tuning with KL, JSD, and Custom loss | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jDG3mZt8iR2f975u5ZcZqK30nhACRo5W?usp=sharing) |
| **NB3** | Evaluation & Metrics — inference, KL/JSD/Cosine, entropy correlation, Precision@K, comparison plots | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1biDtTq-6WW0rJ7cVr8yY5HLXXc1QqLqL?usp=sharing) |
| **NB4** | Explainability & Robustness — Grad-CAM, failure cases, manual inspection, corruption robustness | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yxMBgTCjg2YOEFy3jdSvf37CEyXtMB4R?usp=sharing) |
---

## Datasets

| Dataset | Description | Link |
|---------|-------------|------|
| **CIFAR-10H** | Soft labels for 10,000 CIFAR-10 test images (511,400 human judgments, ~51 per image) | [GitHub Repository](https://github.com/jcpeterson/cifar-10h) |
| **CIFAR-10** | Standard 32×32 image dataset (50,000 training, 10,000 test images with hard labels) | [Official Website](https://www.cs.toronto.edu/~kriz/cifar.html) |

---

## How to Run

1. Click the Colab link for the notebook you want to run
2. **Change runtime to GPU:** Runtime → Change runtime type → T4 GPU
3. **Run cells in order** (NB1 → NB2 → NB3 → NB4)
4. The code automatically:
   - Mounts Google Drive
   - Creates all necessary folders
   - Downloads datasets
   - Saves checkpoints and results

> **Note:** Training NB2 takes ~1-2 hours. Ensure at least 2GB free space in Google Drive.

---

## Key Results

| Model | KL ↓ | JSD ↓ | Cosine ↑ | Pearson r ↑ | P@100 ↑ |
|-------|------|-------|----------|-------------|---------|
| **Custom** | **0.1711** | 0.0563 | **0.9597** | **0.4619** | 0.240 |
| **JSD** | 0.2075 | **0.0532** | 0.9552 | 0.4575 | **0.270** |
| **KL** | 0.1787 | 0.0554 | 0.9570 | 0.4512 | 0.190 |

- **Custom model** (KL + entropy penalty) best on primary metric
- **JSD model** best at finding ambiguous images (Precision@100 = 0.270)
- Two-phase training (hard labels → soft labels) is essential

---

## Key Takeaways

- **Disagreement is signal, not noise** — human annotator disagreement reflects real visual similarity
- **Soft-label training improves calibration** — models better represent uncertainty than hard-label training
- **Entropy penalty is effective** — reduces overconfidence on ambiguous inputs
- **Two-phase training is critical** — pretraining followed by fine-tuning yields better performance
- **Failures occur at class boundaries** — especially among visually similar categories (cat, dog, deer, horse)

---

## Repository Structure

- `cifar10h_project/data/` — CIFAR-10 and CIFAR-10H files
- `cifar10h_project/checkpoints/` — Saved model weights (*.pth)
- `cifar10h_project/results/` — Plots, evaluation results
- `cifar10h_project/results/inference_predictions/` — Cached model predictions
- `cifar10h_project/results/explainability/` — Grad-CAM, failure cases
- `cifar10h_project/notebooks/` — NB1, NB2, NB3, NB4
## Team

| Name |
|------|
| Chalasani Manogna |
| Tanvi Borkar |
| Yashika Gupta |
| Ishani Singh |

---

## References

- Peterson, J. C., Battleday, R. M., Griffiths, T. L., & Russakovsky, O. (2019). *Human uncertainty makes classification more robust*. ICCV.
- Krizhevsky, A. (2009). *Learning multiple layers of features from tiny images*. University of Toronto.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. CVPR.

---
