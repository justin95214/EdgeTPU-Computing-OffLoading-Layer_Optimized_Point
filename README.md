# 엣지컴퓨터 기반 오프로딩기법(OffLoading)을 위한 CNN모델 최적화 레이어 최적점 연구

Notion : https://www.notion.so/OffLoading-CNN-58f58536b35e4f5e99b2621388e74723

임베디드 기기에 효율적인 전력소모를 위해 모바일 클라우드 컴퓨팅 적용을 위해 연산 오프로딩(Computing OffLoading)을 통해 CNN기반 모델에 대한 ON-Device로 엣지컴퓨터(EdgeTPU)추론의 레이어 포인트 지점 연구

## 1. 연산 오프로딩 기법

수행 중인 연산 중 일부를 다른 기기로 옮겨 실행하여, 실시간 인코딩이나 모델 추론과 같이 복잡한 연산 능력을 요구하는 일부를, 제한적인 환경의 모바일 기기에서 고성능 서버 혹은 클라우드로 옮겨 현장의 모바일 기기의 전력 효율을 높이는 기법

## 2. 실험 기기 세부사항

![%E1%84%8B%E1%85%A6%E1%86%BA%E1%84%8C%E1%85%B5%E1%84%8F%E1%85%A5%E1%86%B7%E1%84%91%E1%85%B2%E1%84%90%E1%85%A5%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A1%E1%86%AB%20%E1%84%8B%E1%85%A9%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%83%E1%85%B5%E1%86%BC%E1%84%80%E1%85%B5%E1%84%87%E1%85%A5%E1%86%B8(OffLoading)%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%8B%20251355e78b724f40a90bacb4244b8bd2/Untitled.png](https://github.com/justin95214/EdgeTPU-Computing-OffLoading-Layer_Optimized_Point/blob/main/Resource/chart.png)

!

### 2-1. Modules Requirement

라즈베리파이 Stretch Ubuntu 16.04
- Python 3.5.2
- Tensorflow 1.15
- keras 2.3.1
- EdgeTPU Compiler 2.0
- EdgeTPU Runtime 12

## 3. 실행시간 및 소모 전력 측정-EdgeTPU Engine & Complier (2차 최종본)
(Notion 표링크와 연결됨)
[3-1. VGG16모델 기준 Raspi3+ 레이어별 실행시간 및 소모 전력 측정](https://www.notion.so/2a87febb01cb4daa9021a660d1c9c267)
![%E1%84%8B%E1%85%A6%E1%86%BA%E1%84%8C%E1%85%B5%E1%84%8F%E1%85%A5%E1%86%B7%E1%84%91%E1%85%B2%E1%84%90%E1%85%A5%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A1%E1%86%AB%20%E1%84%8B%E1%85%A9%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%83%E1%85%B5%E1%86%BC%E1%84%80%E1%85%B5%E1%84%87%E1%85%A5%E1%86%B8(OffLoading)%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%8B%20251355e78b724f40a90bacb4244b8bd2/Untitled.png](https://github.com/justin95214/EdgeTPU-Computing-OffLoading-Layer_Optimized_Point/blob/main/Resource/raspi.png)
- type: float32
!
!
!
[%E1%84%8B%E1%85%A6%E1%86%BA%E1%84%8C%E1%85%B5%E1%84%8F%E1%85%A5%E1%86%B7%E1%84%91%E1%85%B2%E1%84%90%E1%85%A5%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A1%E1%86%AB%20%E1%84%8B%E1%85%A9%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%83%E1%85%B5%E1%86%BC%E1%84%80%E1%85%B5%E1%84%87%E1%85%A5%E1%86%B8(OffLoading)%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%8B%20251355e78b724f40a90bacb4244b8bd2/Untitled.png](https://github.com/justin95214/EdgeTPU-Computing-OffLoading-Layer_Optimized_Point/blob/main/Resource/coral.png)
- type : uint8
- DataSet Link :

[실험모델 h5 tflite-20210702T131309Z-001.zip](https://drive.google.com/file/d/1ALiP4MigxddU9ljWFnOEK__WbI-zgsvn/view?usp=drivesdk)

- 그래프 결과( 아래: coral+ raspi / 위 : raspi)

![%E1%84%8B%E1%85%A6%E1%86%BA%E1%84%8C%E1%85%B5%E1%84%8F%E1%85%A5%E1%86%B7%E1%84%91%E1%85%B2%E1%84%90%E1%85%A5%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A1%E1%86%AB%20%E1%84%8B%E1%85%A9%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%83%E1%85%B5%E1%86%BC%E1%84%80%E1%85%B5%E1%84%87%E1%85%A5%E1%86%B8(OffLoading)%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%8B%20251355e78b724f40a90bacb4244b8bd2/Untitled%204.png](%E1%84%8B%E1%85%A6%E1%86%BA%E1%84%8C%E1%85%B5%E1%84%8F%E1%85%A5%E1%86%B7%E1%84%91%E1%85%B2%E1%84%90%E1%85%A5%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A1%E1%86%AB%20%E1%84%8B%E1%85%A9%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%83%E1%85%B5%E1%86%BC%E1%84%80%E1%85%B5%E1%84%87%E1%85%A5%E1%86%B8(OffLoading)%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%8B%20251355e78b724f40a90bacb4244b8bd2/Untitled%204.png)

![%E1%84%8B%E1%85%A6%E1%86%BA%E1%84%8C%E1%85%B5%E1%84%8F%E1%85%A5%E1%86%B7%E1%84%91%E1%85%B2%E1%84%90%E1%85%A5%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A1%E1%86%AB%20%E1%84%8B%E1%85%A9%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%83%E1%85%B5%E1%86%BC%E1%84%80%E1%85%B5%E1%84%87%E1%85%A5%E1%86%B8(OffLoading)%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%8B%20251355e78b724f40a90bacb4244b8bd2/Untitled%205.png](%E1%84%8B%E1%85%A6%E1%86%BA%E1%84%8C%E1%85%B5%E1%84%8F%E1%85%A5%E1%86%B7%E1%84%91%E1%85%B2%E1%84%90%E1%85%A5%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A1%E1%86%AB%20%E1%84%8B%E1%85%A9%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%83%E1%85%B5%E1%86%BC%E1%84%80%E1%85%B5%E1%84%87%E1%85%A5%E1%86%B8(OffLoading)%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%8B%20251355e78b724f40a90bacb4244b8bd2/Untitled%205.png)

## 4. 모델 변환 과정

1. Tensorflow-keras를 통해 VGG모델 생성
2. Keras모델 layer별 .h5 저장
3. keras 2 tflite 으로 .tftlie 변환
4. Coral Accelerator의 EdgeTPU Engine 활용하여 .h5 & .tflite모델 추론 비교
5.  um25c, Monsoon 전력 측정기를 통해 전력소모 측정

![%E1%84%8B%E1%85%A6%E1%86%BA%E1%84%8C%E1%85%B5%E1%84%8F%E1%85%A5%E1%86%B7%E1%84%91%E1%85%B2%E1%84%90%E1%85%A5%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A1%E1%86%AB%20%E1%84%8B%E1%85%A9%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%83%E1%85%B5%E1%86%BC%E1%84%80%E1%85%B5%E1%84%87%E1%85%A5%E1%86%B8(OffLoading)%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%8B%20251355e78b724f40a90bacb4244b8bd2/Untitled%206.png](%E1%84%8B%E1%85%A6%E1%86%BA%E1%84%8C%E1%85%B5%E1%84%8F%E1%85%A5%E1%86%B7%E1%84%91%E1%85%B2%E1%84%90%E1%85%A5%20%E1%84%80%E1%85%B5%E1%84%87%E1%85%A1%E1%86%AB%20%E1%84%8B%E1%85%A9%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%83%E1%85%B5%E1%86%BC%E1%84%80%E1%85%B5%E1%84%87%E1%85%A5%E1%86%B8(OffLoading)%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%8B%20251355e78b724f40a90bacb4244b8bd2/Untitled%206.png)

## 5. 문제점

1차 측정 연구시

- 문제점 :
    1. EdgeTPU Engine에 EdgeTPU Compiler을 적용 시에 Coral의 연산 Ram용량이 초과 되면, CPU의 연산을 사용하므로, Only CPU연산과 EdgeTPU의 연산에 의한 비교가 어려움이 있다고 판단하여 EdgeTPU Engine에 CPU의 Complier를 사용함
    2.  1번의 문제점으로 Engine를 사용하여, usb포트에 연결하여 Coral에 전력소모는 되지만, complier가 CPU이므로 CPU연산 처리 >> 컴퓨터 관점) 그래픽카드 전원은 들어오지만, tensorflow-gpu를 사용하지 못한 현상 발생

        ⇒  임베디드 기기에서 소모하는 총 전력소모을 줄이는 목적이므로

## 6. 2차 재측정 개선 점

2차 재측정으로 개선된 성능 결과 도출

- EdgeTPU Engine에 EdgeTPU Compiler을 적용 시에 Coral의 연산 Ram용량이 초과 되면, CPU의 연산을 사용하므로, Only CPU연산과 EdgeTPU의 연산에 의한 비교가 어려움이 있다고 판단하여 EdgeTPU Engine에 CPU의 Complier를 사용함

⇒> **EdgeTPU Engine & Complier 모두 사용**

⇒> **평균 1회 실행시간과 누적 전력량 부문에서 이전 결과와 큰 차이를 보임**
