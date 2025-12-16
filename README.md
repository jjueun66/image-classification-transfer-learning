# Image Classification with Transfer Learning on CIFAR-10

This project investigates the effects of data augmentation and transfer learning
on image classification performance using the CIFAR-10 dataset.
We compare a baseline CNN trained from scratch with a pretrained ResNet-18
fine-tuned under different preprocessing and augmentation settings.

## Dataset

We use the CIFAR-10 dataset, which consists of 60,000 RGB images
(32×32 resolution) across 10 classes.
The dataset is split into training, validation, and test sets.

## Models

### Baseline CNN
A simple convolutional neural network trained from scratch on CIFAR-10 images
with 32×32 resolution.

### Pretrained ResNet-18
A ResNet-18 model pretrained on ImageNet.
We fine-tune only the last residual block (layer4) and the final fully connected layer.

ResNet-18 was chosen as it is a lightweight and widely used architecture
with readily available ImageNet pretrained weights, making it suitable
for efficient transfer learning experiments.

## Experiments

| Experiment | Model | Input Preprocessing | Augmentation |
|-----------|------|---------------------|--------------|
| Exp1 | Baseline CNN | CIFAR (32×32) | No |
| Exp2 | Baseline CNN | CIFAR (32×32) | Yes |
| Exp3 | ResNet-18 (FT) | CIFAR (32×32) | No |
| Exp4 | ResNet-18 (FT) | CIFAR (32×32) | Yes |
| Exp5 | ResNet-18 (FT) | ImageNet (224×224) | No |
| Exp6 | ResNet-18 (FT) | ImageNet (224×224) | Yes |

## Results

| Experiment | Best Val Acc | Test Acc |
|-----------|--------------|----------|
| Exp1 | 0.7822 | 0.7746 |
| Exp2 | 0.8044 | 0.8106 |
| Exp3 | 0.6792 | 0.6732 |
| Exp4 | 0.6920 | 0.6958 |
| Exp5 | 0.9072 | 0.9046 |
| Exp6 | **0.9192** | **0.9179** |

## Key Findings

1. **Effect of Data Augmentation**
   - Data augmentation consistently improves performance for both baseline
     and transfer learning models.
   - For the baseline CNN, augmentation improves test accuracy from 0.7746 to 0.8106.
   - For the pretrained ResNet-18 with ImageNet preprocessing, augmentation further
     improves test accuracy from 0.9046 to 0.9179.

2. **Effect of Transfer Learning**
   - When using CIFAR-style preprocessing (32×32, CIFAR normalization),
     the pretrained ResNet-18 performs worse than the baseline model,
     despite achieving very high training accuracy.
   - This indicates overfitting and a mismatch between pretrained representations
     and input preprocessing.

3. **Importance of Proper Preprocessing**
   - When ImageNet-style preprocessing (224×224 resizing and ImageNet normalization)
     is applied, the pretrained ResNet-18 significantly outperforms the baseline CNN.
   - This highlights that transfer learning is effective only when the input
     preprocessing is consistent with the pretrained model’s expectations.

## Visualizations

Training/validation accuracy curves and confusion matrices for each experiment
are saved in the `results/` directory.

Example files:
- `exp6_ft_aug_imagenet_preprocess_curves_acc.png`
- `exp6_ft_aug_imagenet_preprocess_confusion.png`

## Conclusion

This project demonstrates that transfer learning can significantly improve
image classification performance on CIFAR-10.
However, the benefit of pretrained models depends critically on using
appropriate input preprocessing and evaluation protocols.
With ImageNet-style preprocessing and data augmentation,
the pretrained ResNet-18 achieves the best performance.

## How to Run

```bash
pip install -r requirements.txt

python -m experiments.exp1_base_no_aug
python -m experiments.exp2_base_aug
python -m experiments.exp3_ft_no_aug
python -m experiments.exp4_ft_aug
python -m experiments.exp5_ft_no_aug_imagenet_preprocess
python -m experiments.exp6_ft_aug_imagenet_preprocess

Note: Model checkpoints and raw training logs are not included in the repository
due to size constraints. All results reported in this README are reproducible
using the provided code.
