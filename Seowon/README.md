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

<b> xception_multi_output directory </b>
  - 모델 : xception
  - 성능 : acc 70.5238%, f1 0.6019
  - train, validation 비율 : 9:1
  - optimizer : Adam
  - scheduler : StepLR
  - ✅ Xception timm으로 가져오니까 에러떠서 전체 코드를 가져와서 사용했는데, pretrained 모델을 사용 못해서 정확도가 떨어지는 것 같기도 함 -> pretrained 해결 후 정확도 74로 상승
  - 🔅 train set에 대한 accuracy는 99, 100까지 나오는데 validation set에 대한 accuracy는 77이 최대인 걸 보니 overfitting 문제가 있어서, dropout 0.5를 추가함
  - 🔅 weight_decay에 l2 normalization을 적용
  - 참고 사이트 
  : [multioutput관련1](https://medium.com/jdsc-tech-blog/multioutput-cnn-in-pytorch-c5f702d4915f) 
    [multioutput관련2](https://learnopencv.com/multi-label-image-classification-with-pytorch/)
    [feature/classifier](https://rwightman.github.io/pytorch-image-models/feature_extraction/#multi-scale-feature-maps-feature-pyramid)
    [xception code](https://github.com/tstandley/Xception-PyTorch/blob/master/xception.py)
  
  - 🔅 age 예측 분포 중에 30~59가 적기도 하고, loss도 낮게 나와서 age의 loss에 1.2를 곱해줌 성능 향상 -> 76.5238, 0.6814
  

<b> basecode </b>
  - 모델 : xception
  - 성능 : (<i>f1 loss</i>) acc 72.5556% f1 0.6304
  - val_ratio : 0.2
  - optimizer : Adam
  - scheduler : CosineAnnealingLR
  - 🔅 loss function을 f1, label_smoothing, cross_entropy, focal loss 를 사용해봤는데, cross_entropy > label_smoothing > f1 순으로 좋음
  - 🔅 Dataset이 같은 얼굴 사람 사진이 train과 valid에 들어가면 valid에서 학습한 사람의 얼굴로 test하게 되는 문제가 발생하므로 사람으로 train/val 나눔
  
   
