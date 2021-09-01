import os
import pandas as pd
import dataset
import torch
import numpy as np
import model
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from PIL import Image
import argparse
from importlib import import_module

def main(args, test_dir, weight_path):
    if not os.path.exists("{}/{}".format(weight_path, args.name)):
        raise RuntimeError('실험이 존재하지 않음')
    ############################################# 데이터 생성 ############################################################
    # meta 데이터와 이미지 경로를 불러옵니다.
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    transform = transforms.Compose([
        Resize((args.resize[0], args.resize[1])),
        ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 데이터셋 불러오기
    dataset_module = getattr(import_module("dataset"), args.dataset)
    dataset_ = dataset_module(image_paths, transform, train=False) ##

    loader = DataLoader(
        dataset_,
        shuffle=False
    )
    
    oof_pred = None
    
    for k, epoch in enumerate(args.kfold):
    ############################################# 모델 생성 ############################################################
        if not os.path.exists("{}/{}/{}_Fold_{}{}.pt".format(weight_path, args.name, k, args.weight_file, epoch)):
            raise RuntimeError('저장된 모델이 없음')
        print("{} Fold Inference Start".format(k))
        # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
        Weight_PATH = "{}/{}/{}_Fold_{}{}.pt".format(weight_path, args.name, k, args.weight_file, epoch) # epoch 저장하고 가져오기 만들어야함
        device = torch.device('cuda')
        model_module = getattr(import_module("model"), args.model)
        model_ = model_module(num_classes=18).to(device)
        model_.load_state_dict(torch.load(Weight_PATH))
        model_.eval()

    ############################################# 추론 과정 ############################################################
        # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
        all_predictions = []
        for images in loader:
            with torch.no_grad():
                images = images.to(device)
                pred = model_(images)
                pred = pred.argmax(dim=-1)
                all_predictions.extend(pred.cpu().numpy())
                
        fold_pred = np.array(all_predictions)
        if oof_pred is None:
            oof_pred = fold_pred / len(args.kfold)
        else:
            oof_pred += fold_pred / len(args.kfold)
            
        print("{} Fold Inference Done".format(k))
        
    # 제출할 파일을 저장합니다.
    submission['ans'] = np.argmax(oof_pred, axis=1)
    submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
    print('test inference is done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    
    # Data and Model checkpoints directories
    parser.add_argument("--dataset", type=str, default="Dataset", help='dataset augmentation type (default: Dataset)')
    parser.add_argument("--resize", nargs="+", type=int, default=[380, 380], help='resize size for image when training')
    parser.add_argument("--kfold", nargs="+", type=int, default=[0, 0, 0, 0, 0], help='high acc epoch num from each fold')
    parser.add_argument('--model', type=str, default='efficient_b4', help='model type (default: efficient_b4)') # 내 전담 모델
    parser.add_argument('--name', default=None, help='model load at {WEIGHT_PATH}/{name}')
    parser.add_argument('--weight_file', default='model_weight')

    # Container environment
    parser.add_argument('--test_dir', type=str, default=os.environ.get('SM_CHANNEL_TEST', '/opt/ml/input/data/eval'), help="test directory path (default=/opt/ml/input/data/eval)")
    parser.add_argument('--weight_path', type=str, default=os.environ.get('WEIGHT_PATH', './save_weights'), help="directory path having weights files")

    args = parser.parse_args()

    main(args, args.test_dir, args.weight_path)