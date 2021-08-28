<b> xception.ipynb </b>
  - 모델 : Xception
  - 성능 : acc 76.603, f1 0.687 💥
  - train, validation 비율 : 9:1
  - optimizer : Adam
  - scheduler : StepLR
  - error : test inference 하는 과정에서 이 에러 났는데, https://stackoverflow.com/questions/57079219/img-should-be-pil-image-got-class-torch-tensor
            transforms에서 toTensor 다음에 Normalize 
   
<b> efficientnet.ipynb </b>
  - 모델 : efficientnet b0
  - 성능 : acc 73.5873, f1 0.6548
  - train, validation 비율 : 95:5
  - optimizer : Adam
  - scheduler : StepLR
  - 🔅 OOM 문제로 이미지의 가로, 세로 길이를 반으로 줄임. 근데 서버 재시작하니까 OOM 문제 없어졌으니까 원본 크기로 다시 시도해보기
  - 🔅 epoch이 3이 넘어가면서 accuracy가 계속 100으로 측정되길래, overfitting이 발생한 것 같아서 epoch 2 까지만 돌리고 제출해봤는데 acc 71%가 나옴. dropout 등의 다른 방식으로 overfitting을 잡고, early stop 되는 경우를 더 타이트하게 잡아야 할 것 같음
