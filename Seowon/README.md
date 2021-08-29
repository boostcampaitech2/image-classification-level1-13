> <i> 새롭게 시도한 점은 🔅 이걸로 표시했습니다. </i><br/>
> <i> 추가로 시도해야 할 점은 ✅ 이걸로 표시했습니다. </i>

<b> xception.ipynb </b>
  - 모델 : Xception
  - 성능 : acc 76.603, f1 0.687 💥
  - train, validation 비율 : 9:1
  - optimizer : Adam
  - scheduler : StepLR
  - error : test inference 하는 과정에서 이 에러 났는데, https://stackoverflow.com/questions/57079219/img-should-be-pil-image-got-class-torch-tensor
            transforms에서 toTensor 다음에 Normalize 
  - 🔅 batch size 64로 늘려서 해봤는데 정확도 75프로로 조금 낮아짐. Xception은 batch 32가 적당해보임
  - 🔅 train : validation 비율 95:5보다 9:1이 더 좋음. learning_rate 1e-4보다 1e-3이 더 좋음
  - dropout 0.7보다 dropout 0.5가 더 좋음. dropout 0.7 acc = 75.5873

<b> xception2.ipynb </b>
  - 모델 : Xception
  - 성능 : acc 71.46
  - train, val 비율 : 9:1
  - 🔅 optimizer : optim.RAdam(model.parameters(), lr=0.0015, betas=(0.9, 0.999), weight_decay=1e-4)
  - 🔅 scheduler : torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
  - 🔅 dropout : 0.5
  - ✅ optimizer와 scheduler의 lr과 T_max 값을 바꿔야할 것 같음. 특히 T_max 값을 batch size 크기와 비례해서 잡아야하는데 너무 크게 잡은 것 같음

 
   
<b> efficientnet.ipynb </b>
  - 모델 : efficientnet b0
  - 성능 : acc 73.5873, f1 0.6548
  - train, validation 비율 : 95:5
  - optimizer : Adam
  - scheduler : StepLR
  - 🔅 OOM 문제로 이미지의 가로, 세로 길이를 반으로 줄임. 근데 서버 재시작하니까 OOM 문제 없어졌으니까 원본 크기로 다시 시도해보기
  - 🔅 epoch이 3이 넘어가면서 accuracy가 계속 100으로 측정되길래, overfitting이 발생한 것 같아서 epoch 2 까지만 돌리고 제출해봤는데 acc 71%가 나옴. dropout 등의 다른 방식으로 overfitting을 잡고, early stop 되는 경우를 더 타이트하게 잡아야 할 것 같음
  - 🔅 efficientnet.ipynb에서 Resize를 작게 하지 않고 원본 크기로 학습하고 batchsize도 64로 늘렸더니 <b>acc 75.6032, f1 0.6770</b> 로 상승
  - 🔅 dropout 0.5 추가했는데 💥 76.9841, 0.6925 💥로 성능 향상 - [efficientnet_dropout.ipynb](https://github.com/boostcampaitech2/image-classification-level1-13/blob/main/Seowon/efficientnet_dropout.ipynb) 
