
## A classification method based on optical flow for violence detection

- Firstly, input frames are converted to the grayscale, and the optical flow is obtained between two consequence frames. 
- Next, optical flow magnitude and orientation changes should be calculated between pairs of sequential frames. Secondly, different threshold values which are set adaptively are applied to the optical flow magnitude and orientation changes for obtaining six binary indicators which are then processed to calculate the HOMO descriptor. 
- Finally, the descriptor is considered as a feature vector of the SVM classifier.
	- 먼저 입력 프레임을 회색조(그레이스케일)로 변환하고, 두 연속 프레임 간의 광학 흐름(optical flow)을 계산합니다. 이후, 연속된 프레임 쌍 사이에서 광학 흐름의 크기(magnitude)와 방향(orientation)의 변화를 계산합니다. 다음으로, 광학 흐름의 크기와 방향 변화에 대해 적응적으로 설정된 여러 임계값을 적용하여 6개의 이진 지표(binary indicator)를 얻습니다. 이 지표들은 HOMO 기술자를 계산하는 데 사용됩니다. 마지막으로, HOMO 기술자는 SVM 분류기의 특징 벡터로 활용됩니다.
	- Histogram of Optical Flow Magnitude and Orientation(HOMO)는 결국 광학 흐름(Optimal Flow)의 크기(Magnitude)와 방향(Orientation)의 변화를 분석하여 폭력 행위를 탐지하기 위한 특징을 추출하는 기술자.
- https://www.sciencedirect.com/science/article/pii/S0957417419301460?casa_token=e0vM7BqyAhIAAAAA:dBpW7svanmmGX8Y8PLyN6WyGNse-DsEEKv6TrxgGsGQs_kz00UcPO9NfOjCrWRHNfriouzc4_so#bib0011


---
##  Efficient Violence Detection in Surveillance

### CNN (Convolutional Neural Network)
CNN은 주로 **이미지 및 비디오 처리**에 사용되는 딥러닝 알고리즘으로, 공간적 구조를 학습하는 데 특화되어 있습니다. CNN은 이미지의 **공간적 계층적 특징(Spatial Hierarchical Features)** 을 추출하는 데 뛰어난 성능을 보입니다.

### RNN (Recurrent Neural Network)
RNN은 **시간적(Sequential) 데이터**를 처리하는 데 특화된 딥러닝 알고리즘으로, 이전의 정보가 다음 계산에 영향을 미칠 수 있도록 설계되었습니다.

- Proposed Model
	- MobileNet V2를 사용. 각 프레임에 대해 공간적 특징을 추출하고, 이를 시간 분배방식으로 처리.
		- MobileNet V2는 CNN의 경량화 모델.
	- LSTM을 활용하여 연속적인 프레임 간의 시간적 상관관계를 학습.
		- LSTM은 RNN의 일종.
		- LSTM 구조는 30프레임(1초)의 배치를 입력받아 연속적인 프레임 간 시간적 상관관계를 학습.
		- 움직임의 패턴과 연속적인 행동 데이터를 기반으로 폭력 여부를 판단.
	- Dense 레이어로 구성된 단순한 이진 분류기를 통해 폭력 여부를 예측.
	- 손실 함수로 Binary Cross Entropy를 사용.
1. MobileNet V2를 encoder로 사용하여 각 비디오 프레임의 공간적 특징을 추출.
2. MobileNet V2에서 추출된 Feature Map을 U-Net 디코더에 결합하여. 더 풍부한 공간적 정보를 학습.
	- MobileNet을 채택함으로써 기존 최신 분류기들보다 계산량과 메모리 요구량을 줄일 수 있음. 기존 CNN보다 훨씬 적은 계산량으로도 유사한 정확도를 제공.
### U-Net의 개념
- **U-Net**은 원래 **의료 영상 분할**을 위해 개발된 CNN 기반의 구조로, **인코더-디코더** 구조를 가집니다.
- 주요 특징:
    - **인코더**: 입력 데이터에서 특징을 추출.
    - **디코더**: 추출된 특징을 기반으로 원래 데이터 형태를 복원.
    - **스킵 연결(Skip Connection)**: 인코더 단계에서 추출된 저수준 정보를 디코더에 전달하여 더 세밀한 특징 복원.


- https://www.mdpi.com/1424-8220/22/6/2216
---

## 컨볼루션 뉴럴 네트워크를 이용한 군중 행동 감지
### 1. VGG-16 기반의 CNN
- **VGG-16**은 16개의 계층(13개의 합성곱 계층, 3개의 완전 연결 계층)으로 구성된 심층 신경망입니다.
- 이 모델은 고정된 크기(64x64)로 입력된 이미지에서 **공간적 특징**을 추출하며, 논문에서 다음과 같은 방식으로 사용됩니다:
    1. **프레임 별 특징 추출**:
        - 각 비디오를 프레임으로 나눈 후, VGG-16 모델을 통해 프레임별 특징 맵을 생성.
    2. **활성화 함수(Softmax)**:
        - 추출된 특징을 기반으로 **정상(Normal)**, **도주(Escape)**, **폭력(Fight)**의 세 가지 클래스로 분류.
    3. **미세 조정(Fine-tuning)**:
        - 사전 학습된 VGG-16을 사용하고, 학습 데이터셋에 맞게 마지막 층(완전 연결 층)과 출력 층을 재학습.
### 2. 데이터 전처리 및 클래스 분류
#### **데이터 전처리**
- **비디오 데이터 → 프레임 변환**:
    - 각 비디오를 프레임 단위로 변환하여 **고정된 크기(64x64)**로 조정.
    - 데이터셋 불균형 문제 해결을 위해 클래스별로 균형 있게 샘플링.
    - 최종적으로 **정상(Normal)**, **도주(Escape)**, **폭력(Fight)** 클래스를 구성.
#### **클래스 분류**
- VGG-16 모델의 **출력층(Softmax)**에서 각 프레임을 다음 세 가지 행동으로 분류:
    1. **Normal**: 군중의 정상적인 움직임.
    2. **Escape**: 비정상적인 도주 행동.
    3. **Fight**: 폭력적인 움직임.
### 3. 특징 추출 및 분류 과정

#### **특징 추출 과정**
1. **합성곱 계층 (Convolutional Layers)**:
    - 필터를 사용해 입력 이미지의 국소적 특징(엣지, 텍스처 등)을 학습.
    - 각 계층이 저수준(로우 레벨) → 고수준(하이 레벨) 특징을 점진적으로 학습.
2. **풀링 계층 (Pooling Layers)**:
    - 특징 맵의 크기를 축소하며 계산 효율성 향상.
    - 중요한 특징만 남기기 위해 **Max Pooling(최대 풀링)** 사용.
3. **완전 연결 계층 (Fully Connected Layers)**:
    - CNN이 추출한 특징 맵을 1차원 벡터로 변환.
    - 전체 정보를 통합하여 최종 분류 수행.

