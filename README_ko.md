# CIFAR-10을 이용한 전이학습 기반 이미지 분류

본 프로젝트는 CIFAR-10 데이터셋을 사용하여
데이터 증강(data augmentation)과 전이학습(transfer learning)이
이미지 분류 성능에 미치는 영향을 분석한다.
from-scratch로 학습한 Baseline CNN과
ImageNet으로 사전학습된 ResNet-18을
서로 다른 전처리 및 증강 설정 하에서 fine-tuning하여 성능을 비교한다.

## Dataset

본 프로젝트에서는 CIFAR-10 데이터셋을 사용한다.
CIFAR-10은 10개의 클래스에 대해 총 60,000장의 RGB 이미지로 구성되어 있으며,
각 이미지의 해상도는 32×32이다.
데이터는 학습(training), 검증(validation), 테스트(test) 세트로 분할하여 사용한다.

## Models

### Baseline CNN
CIFAR-10 이미지(32×32 해상도)를 입력으로 받는
간단한 합성곱 신경망(CNN)을 from-scratch 방식으로 학습하였다.

### Pretrained ResNet-18
ImageNet 데이터셋으로 사전학습된 ResNet-18 모델을 사용하였다.
본 프로젝트에서는 마지막 residual block(layer4)과
최종 fully connected layer만을 fine-tuning하였다.

ResNet-18은 비교적 가볍고 널리 사용되는 구조이며,
torchvision에서 ImageNet 사전학습 가중치를 쉽게 사용할 수 있어
효율적인 전이학습 실험에 적합한 모델로 선택하였다.

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

1. **데이터 증강의 효과**
   - 데이터 증강은 baseline 모델과 전이학습 모델 모두에서
     분류 성능을 일관되게 향상시켰다.
   - Baseline CNN의 경우, 데이터 증강을 적용하면
     test 정확도가 0.7746에서 0.8106으로 증가하였다.
   - ImageNet 전처리를 적용한 ResNet-18에서는
     데이터 증강을 통해 test 정확도가 0.9046에서 0.9179까지 추가로 향상되었다.

2. **전이학습의 효과**
   - CIFAR 스타일 전처리(32×32, CIFAR 정규화)를 사용한 경우,
     사전학습된 ResNet-18은 학습 정확도는 매우 높았으나
     baseline CNN보다 검증 및 테스트 성능이 낮았다.
   - 이는 과적합(overfitting)과 함께,
     사전학습된 표현과 입력 전처리 간의 불일치로 인한 성능 저하를 의미한다.

3. **적절한 전처리의 중요성**
   - ImageNet 스타일 전처리(224×224 resize 및 ImageNet 정규화)를 적용한 경우,
     사전학습된 ResNet-18은 baseline CNN을 크게 상회하는 성능을 보였다.
   - 이는 전이학습이 효과적으로 작동하기 위해서는
     사전학습 시 사용된 입력 분포와 일관된 전처리가 필수적임을 보여준다.

## Visualizations

각 실험에 대한 학습/검증 정확도 곡선과 confusion matrix는
`results/` 디렉토리에 저장되어 있다.

Example files:
- `exp6_ft_aug_imagenet_preprocess_curves_acc.png`
- `exp6_ft_aug_imagenet_preprocess_confusion.png`

## Conclusion

본 프로젝트를 통해 전이학습은 CIFAR-10 이미지 분류 성능을
유의미하게 향상시킬 수 있음을 확인하였다.
다만, 사전학습 모델의 성능 향상 효과는
입력 전처리와 평가 방식이 적절히 설정되었을 때에만 극대화된다.
ImageNet 스타일 전처리와 데이터 증강을 함께 적용한 경우,
사전학습된 ResNet-18이 가장 높은 성능을 달성하였다.

## How to Run

```bash
pip install -r requirements.txt

python -m experiments.exp1_base_no_aug
python -m experiments.exp2_base_aug
python -m experiments.exp3_ft_no_aug
python -m experiments.exp4_ft_aug
python -m experiments.exp5_ft_no_aug_imagenet_preprocess
python -m experiments.exp6_ft_aug_imagenet_preprocess

