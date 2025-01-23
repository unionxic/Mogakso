# YOLO를 활용한 Violence Detection 가이드

## 1. YOLO란 무엇인가?

**YOLO(You Only Look Once)**는 실시간 Object Detection 분야에서 많이 사용되는 딥러닝 모델입니다.

**“You Only Look Once”**라는 이름처럼 이미지를 한 번 보고(단 한 번의 신경망 Forward Pass) 그 안에 있는 Object들의 **위치(Bounding Box)**와 어떤 Object인지(Class)를 동시에 예측합니다.

초기 R-CNN 계열(Region Proposal + Classification) 모델들은 이미지를 여러 번 나누어 처리하거나 별도의 영역 추출 과정을 거쳤기에 속도가 비교적 느렸지만, YOLO는 하나의 CNN(Convolutional Neural Network)으로 한 번에 모든 예측을 수행하기 때문에 빠른 속도가 강점입니다.

### YOLO의 기본 흐름

1. 입력 이미지가 일정 크기로 Resize되어 신경망으로 들어갑니다.
2. CNN이 이미지 특징을 추출하고, 마지막에 S x S Grid(예: 13 x 13 혹은 19 x 19 등) 형태의 출력이 만들어집니다.
3. 각 Grid Cell마다 여러 개의 Bounding Box(Anchor Box) 좌표와 그 박스 안에 Object가 존재할 확률(Objectness), 그리고 Object Class 확률(예: Person, Car, Dog 등)을 예측합니다.
4. 각 예측을 NMS(Non-Maximum Suppression) 등의 Post-Processing 기법을 사용해 겹치는 박스 중 가장 확실도가 높은 예측을 남기고 나머지는 제거합니다.

---

## 2. Violence Detection에서 YOLO가 어떻게 쓰일까?

일반적으로 **Violence Detection**은 ‘폭력 행동을 하고 있는지’를 판별하는 문제이므로, 보통 Action Recognition의 영역에 가깝습니다. 하지만 다음과 같은 방식으로 YOLO를 활용할 수도 있습니다:

### 1) 폭력을 Object(Class)로 간주해 학습

- 이미지 속에서 "Violence" 동작이 일어나고 있는 구간을 하나의 "Object"라고 가정하고, Bounding Box로 감싸도록 Dataset을 구성할 수 있습니다.
- Violence 행동이 벌어지고 있는 사람들의 신체 일부나 서로 엉켜 있는 부분을 "Violence"라는 라벨로 묶어서 Bounding Box로 표시하고, 비폭력 장면에서는 Bounding Box가 없이 "No Object"로 처리합니다.

### 2) Multi-Class Classification + Violence Class 추가

- Person, Car, Bag 등의 기존 Object Label에 “Violence” 또는 “Fight” 같은 Label을 추가해서 학습할 수 있습니다.
- 특정 장면이 ‘폭력이 일어나고 있다’고 판별할 단서가 분명하다면 YOLO Object Detection을 통해 어느 정도 구분이 가능합니다.

### 3) 추가적인 Temporal Information 활용

- Violence는 ‘연속된 동작’이므로, 단순히 이미지 한 장만 보고 판단하기보다는 여러 Frame을 종합적으로 분석해야 합니다.
- 3D-CNN, LSTM 등의 추가 모델과 YOLO를 혼합해 사용할 수 있습니다.

---

## 3. YOLO가 내부적으로 사용하는 이론 (간단 버전)

### Bounding Box(Anchors) Prediction

- YOLO는 이미지를 Grid로 나눈 뒤, 각 Grid Cell이 미리 정의된 k개의 “Anchor Box”에 대해 위치 보정을 예측합니다.
- 좌표 변환 및 Scale 조정 과정을 거쳐 최종 박스를 도출합니다.

### Objectness Score

- 각 박스별로 “이 박스 안에 진짜 Object가 있느냐?”를 나타내는 점수를 예측합니다.
- 0과 1 사이 값으로 나타내며, 1에 가까울수록 Object가 존재할 확률이 높습니다.

### Class Probability

- Object가 있다면, 해당 Object의 Class(예: Violence, Non-Violence)를 예측합니다.

### Loss Function

- 위치 오차, Objectness Score 오차, Class Classification 오차 등을 종합하여 Loss를 최소화하도록 학습합니다.

### NMS (Non-Maximum Suppression)

- 겹치는 박스 중 가장 높은 확률의 박스를 남기고 나머지를 제거합니다.

---

## 4. 실제 프로젝트 적용 방법

### 1) Dataset 준비

- Video(또는 Image)에서 Violence가 발생하는 부분을 Bounding Box로 라벨링해야 합니다.
- Class는 "Violence" 또는 "Fight", "Normal" 등으로 나눌 수 있습니다.

### 2) Training Pipeline 구성

- PyTorch나 TensorFlow를 사용하여 YOLO 구현체를 활용합니다.
- Dataset은 YOLO 포맷(txt 형식)으로 정리합니다.

### 3) Training 설정

- `epochs`, `batch size`, `learning rate` 등 Hyperparameter를 설정합니다.
- Class 수(`num_classes`)를 Violence 및 Non-Violence 등으로 설정해야 합니다.

### 4) Inference(검증) 단계

- 학습된 모델을 사용하여 새로운 Video의 Frame을 추출하고 "Violence" 박스를 얼마나 정확히 찾아내는지 확인합니다.
- Precision, Recall, mAP 등의 Metric으로 성능을 평가합니다.

### 5) Real-Time 적용

- YOLO는 빠른 속도를 제공하므로 GPU를 활용하면 Real-Time(초당 30프레임 전후)으로 처리가 가능합니다.
- Violence Detection 시 경고 시스템 등을 추가할 수 있습니다.

---

## 5. Conclusion

YOLO를 활용한 Violence Detection은 실시간으로 영상에서 특정 행동을 탐지할 수 있는 효과적인 방법입니다. 하지만 Violence는 연속된 동작을 포함하기 때문에 추가적인 분석 기법을 함께 적용하면 더 높은 정확도를 얻을 수 있습니다.



