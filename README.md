# Comparative Analysis of Lung Cancer Detection using CNN and Meta Learning

## Overview
This project implements an end-to-end machine learning pipeline for lung cancer detection from medical images (e.g., CT scans, X-rays). It compares traditional Convolutional Neural Networks (CNNs) with Meta Learning (Few-Shot Learning) approaches like Prototypical Networks. The notebook evaluates models on metrics such as Accuracy, Precision, Recall, F1 Score, ROC-AUC, and Training Time.

## Features
- **Custom CNN**: Baseline model built from scratch.
- **Transfer Learning**: Pretrained models (ResNet50, DenseNet121, EfficientNetB0).
- **Meta Learning**: Prototypical Networks for few-shot classification.
- Data preprocessing with augmentations.
- Model evaluation and comparison.
- Single image prediction.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/RugvedVaidya/CNN-vs-meta-learning-lung-cancer-detection.git
   cd Comparative-Analysis
   ```

2. Install dependencies:
   ```
   pip install numpy pandas matplotlib seaborn tqdm pillow torch torchvision scikit-learn
   ```

3. Ensure you have a GPU-enabled environment if using CUDA.

## Dataset
- Organize images into class folders (e.g., `dataset/benign/` and `dataset/malignant/`).
- Update `DATASET_PATH` in the notebook to your local path.
- Supported formats: PNG, JPG, JPEG, BMP, TIF, TIFF.

## Usage
1. Open the Jupyter notebook: `Lung_Cancer_Detection_CNN_vs_MetaLearning.ipynb`.
2. Run cells sequentially to load data, train models, and evaluate.
3. For prediction, use the `predict_image` function with a custom image path.

## Models and Training
- Train models with early stopping and validation.
- Compare results in the final section.
- Best model selected based on F1 Score.

## Results
Results are aggregated in a DataFrame and visualized. Transfer learning models typically outperform custom CNNs, while Meta Learning excels in low-data scenarios.

## Contributing
Feel free to fork and submit pull requests for improvements.

## License
This project is for educational purposes. Ensure compliance with data privacy laws for medical images.