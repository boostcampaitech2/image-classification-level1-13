# import argparse
# import collections
import torch
import numpy as np
import dataset
import model
import loss
import pandas as pd
import torch.nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
import torch.optim as optim

# from parse_config import ConfigParser

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
# def main(config):
def main():
#     logger = config.get_logger('train')

    # 파일가져오기 # 위치 code/Model
    csv_path = "../../input/data/train/train_class_aug.csv" # Arg_parser로 입력받기
    data = pd.read_csv(csv_path)
    TRAIN_PATH = data['path'].values
    TRAIN_CLASS = data['total_class_18'].values
    TRAIN_TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#                                           AddGaussianNoise(0., 1.)
                                         ])
    
    csv_path_valid = "../../input/data/train/val_class_aug.csv" # Arg_parser로 입력받기
    data_valid = pd.read_csv(csv_path_valid)
    VALID_PATH = data_valid['path'].values
    VALID_CLASS = data_valid['total_class_18'].values
    VALID_TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # 데이터셋 생성 순서 -> img_paths, transform, class_=None, train=False
    # Train
    dataset_train = dataset.Dataset(img_paths=TRAIN_PATH, transform= TRAIN_TRANSFORM, class_ = TRAIN_CLASS, train=True)
    dataset_train = DataLoader(dataset=dataset_train, batch_size=32, shuffle=True, num_workers=4) # GPU * 4
    
    dataset_valid = dataset.Dataset(img_paths=VALID_PATH, transform= VALID_TRANSFORM, class_ = VALID_CLASS, train=True) # train은 False가 맞지만, valid 데이터가 맞는지 확인을 위해서는 라벨을 return 받아야함 따라서 train = True로 줌
    dataset_valid = DataLoader(dataset=dataset_valid, batch_size=32, shuffle=True, num_workers=4) # GPU * 4
    
    # validation 분할
#     valid_data_loader = data_loader.split_validation()

    # 모델 생성
    model_ = model.efficient_b0(num_classes=18)
#     logger.info(model)

    # GPU 준비 - 어차피 여기선 v100 1개 할당만 받음
    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    if n_gpu > 0:
        print("Use GPU")
    else:
        print("Use CPU")
    device_ids = list(range(n_gpu))
    
    model_ = model_.to(device)
    if len(device_ids) > 1:
        model_ = torch.nn.DataParallel(model_, device_ids=device_ids)

    # loss함수
    loss_ = loss.CrossEntropyLoss()

    # optimizer
    trainable_params = filter(lambda p: p.requires_grad, model_.parameters())
    lr = 1e-3
    optm = optim.Adam(trainable_params, lr)
    
    # 학습과정 중 평가
    def func_eval(model_,data_iter,device): # 평가하기
        with torch.no_grad():
            n_total,n_correct = 0,0
            model_.eval() # evaluate (affects DropOut and BN)
            for batch_in,batch_out in data_iter:
                y_trgt = batch_out.to(device)
                model_pred = model_(batch_in.to(device))
                _,y_pred = torch.max(model_pred.data,1)
                n_correct += (y_pred==y_trgt).sum().item()
                n_total += batch_in.size(0)
            val_accr = (n_correct/n_total)
            model_.train() # back to train mode 
        return val_accr

    # train
    print ("Start training.") # 트레이닝 코드 이전 코드와 같음
#     model_.init_param() # initialize parameters # pretrain일 경우할 필요 없음
#     model_.load_weights("./darknet53.conv.74") # darknet만
    model_.train() # to train mode 
    EPOCHS,print_every = 300, 1
    prior_acc = 0
    for epoch in range(EPOCHS):
        loss_val_sum = 0
        for batch_in,batch_out in dataset_train:
            # Forward path
            y_pred = model_.forward(batch_in.to(device))
            loss_out = loss_(y_pred,batch_out.to(device))
            # Update
            optm.zero_grad()
            loss_out.backward()
            optm.step()
            loss_val_sum += loss_out
        loss_val_avg = loss_val_sum/len(dataset_train)
        # Print
        if ((epoch%print_every)==0) or (epoch==(EPOCHS-1)):
            train_accr = func_eval(model_,dataset_train,device)
            test_accr = func_eval(model_,dataset_valid,device)
            print ("epoch:[%d] loss:[%.3f] train_accr:[%.3f] test_accr:[%.3f]."%
                   (epoch,loss_val_avg,train_accr,test_accr))
            with open("./log/TrainLog.txt", "a") as file:
                file.write("epoch:[%d] loss:[%.3f] train_accr:[%.3f] test_accr:[%.3f].\n"%
                   (epoch,loss_val_avg,train_accr,test_accr))
#             if prior_acc < test_accr: # 성능이 개선된 경우에만 저장
            torch.save(model_.state_dict(), "./save_weights/model_weight{}.pt".format(epoch))
#             prior_acc = test_accr
    print ("Done")


if __name__ == '__main__':
#     args = argparse.ArgumentParser(description='PyTorch Template')
#     args.add_argument('-c', '--config', default=None, type=str,
#                       help='config file path (default: None)')
#     args.add_argument('-r', '--resume', default=None, type=str,
#                       help='path to latest checkpoint (default: None)')
#     args.add_argument('-d', '--device', default=None, type=str,
#                       help='indices of GPUs to enable (default: all)')

#     # custom cli options to modify configuration from default values given in json file.
#     CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
#     options = [
#         CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
#         CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
#     ]
#     config = ConfigParser.from_args(args, options)
#     main(config)
    main()