Predicting Human Annotator Disagreement
Project for 3rd Year Deep Neural Networks Course

Learning what makes images inherently ambiguous

Based on: Peterson et al., ICCV 2019 — "Human Uncertainty Makes Classification More Robust"

Project Overview
This project builds a deep neural network that predicts the full distribution of human annotator labels for CIFAR-10 images, rather than just a single hard label. The model outputs a 10-class probability distribution representing how 50+ humans would vote on each image.

Notebooks (Click to Open)
Notebook	Description	Link
NB1	Data Loading & Exploration — downloads CIFAR-10 and CIFAR-10H, creates splits, computes entropy, generates visualizations	Open in Colab
NB2	Model Training — defines ResNet-18 adapted for 32×32 images, pretrains on hard labels, fine-tunes on soft labels with 3 loss functions (KL, JSD, Custom)	Open in Colab
NB3	Evaluation & Metrics — runs inference on test set, computes KL, JSD, Cosine similarity, entropy correlation, Precision@K, generates comparison plots	Open in Colab
NB4	Explainability & Robustness — Grad-CAM visualizations, failure case analysis, manual inspection of ambiguous images, robustness checks (corruptions, subsampling, class-conditional)	Open in Colab
Datasets
Dataset	Description	Link
CIFAR-10H	Soft labels for 10,000 CIFAR-10 test images (511,400 human judgments, ~51 per image)	GitHub Repository
CIFAR-10	Standard 32×32 image dataset (50,000 training, 10,000 test images with hard labels)	Official Website
How to Run
Open any notebook by clicking the Colab link above

Change runtime to GPU: Runtime → Change runtime type → T4 GPU

Run cells in order (NB1 → NB2 → NB3 → NB4)

The code will automatically:

Mount Google Drive

Create necessary folders (cifar10h_project/data/, checkpoints/, results/)

Download CIFAR-10 and CIFAR-10H

Save checkpoints and results to Drive

Note: Training NB2 takes ~1-2 hours. Make sure you have at least 2GB free space in Google Drive.

Key Results
Model	KL ↓	JSD ↓	Cosine ↑	Pearson r ↑	P@100 ↑
Custom	0.1711	0.0563	0.9597	0.4619	0.240
JSD	0.2075	0.0532	0.9552	0.4575	0.270
KL	0.1787	0.0554	0.9570	0.4512	0.190
Custom model (KL + entropy penalty) best on primary metric

JSD model best at finding ambiguous images (Precision@100 = 0.270)

Two-phase training (hard labels → soft labels) is essential

Repository Structure
text
cifar10h_project/
├── data/                 # CIFAR-10 and CIFAR-10H files
├── checkpoints/          # Saved model weights (*.pth)
├── results/              # Plots, evaluation results, inference predictions
│   ├── inference_predictions/
│   └── explainability/   # Grad-CAM, failure cases, manual inspection
└── notebooks/            # NB1, NB2, NB3, NB4 (run in order)
Requirements
All libraries are pre-installed in Google Colab:

Python 3.10+

PyTorch / torchvision

NumPy, Matplotlib, Seaborn

scikit-learn, scipy

OpenCV (cv2)

PIL

Team
Name	Role
Chalasani Manogna	Team Member
Tanvi Borkar	Team Member
Yashika Gupta	Team Member
Ishani Singh	Team Member
References
Peterson, J. C., Battleday, R. M., Griffiths, T. L., & Russakovsky, O. (2019). Human uncertainty makes classification more robust. ICCV.

Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. University of Toronto.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
