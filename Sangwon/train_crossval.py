import argparse
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
import wandb
from sklearn.model_selection import StratifiedKFold

from importlib import import_module
import os
import json
from torch.optim.lr_scheduler import StepLR
    
def main(args, data_, save_path):
    ############################################ 폴더 확인 ##############################################################
    print("args_name", args.name)
    # 모델 저장폴더, 로그폴더 생성
    if args.name == None:
        raise RuntimeError('실험 이름 필수 지정(중복안되게 지을 것)')
    if not os.path.exists("{}".format(save_path)):
        os.mkdir("{}".format(save_path))
    if not os.path.exists("{}/{}".format(save_path, args.name)):
        os.mkdir("{}/{}".format(save_path, args.name))
    else:
        raise RuntimeError('실험 이름 중복')
    if not os.path.exists("./log"):
        os.mkdir("./log")
    
    ############################################# 평가 함수 ############################################################
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
        
    ############################################# k-fold 데이터 생성 ############################################################
    
    k_fold = 5 # k_fold 5로 고정 # 근데 기본이 5임
    
    # 파일가져오기 # 위치 code/Model
    data = pd.read_csv(data_)
    
    X, y = data['path'].values, data['total_class_18'].values
    skf = StratifiedKFold(n_splits=k_fold, random_state=123, shuffle=True)
    EPOCHS, print_every = args.epochs, args.log_interval
    EachFoldHighAccEpoch = []
    EachFoldHighAcc = []
    
    for k, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        # train data
        TRAIN_PATH = X_train
        TRAIN_CLASS = y_train
        TRAIN_TRANSFORM = transforms.Compose([transforms.Resize((args.resize[0], args.resize[1])), transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                             ])
        # valida data
        VALID_PATH = X_test
        VALID_CLASS = y_test
        VALID_TRANSFORM = transforms.Compose([transforms.Resize((args.resize[0], args.resize[1])), transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        dataset_module = getattr(import_module("dataset"), args.dataset)
        # train_load
        dataset_train = dataset_module(img_paths=TRAIN_PATH, transform= TRAIN_TRANSFORM, class_ = TRAIN_CLASS, train=True)
        dataset_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4) # GPU * 4

        # valid_load
        dataset_valid = dataset_module(img_paths=VALID_PATH, transform= VALID_TRANSFORM, class_ = VALID_CLASS, train=True) # train은 False가 맞지만, valid 데이터가 맞는지 확인을 위해서는 라벨을 return 받아야함 따라서 train = True로 줌
        dataset_valid = DataLoader(dataset=dataset_valid, batch_size=args.valid_batch_size, shuffle=True, num_workers=4) # GPU * 4

    ############################################# 모델 생성 (초기화 - pretrained 가져옴) ########################################
        model_module = getattr(import_module("model"), args.model)
        model_ = model_module(num_classes=18)

        # wandb init
        wandb.init(project='sangwon', entity='13ai')
        config = wandb.config.update(args)
        
        # wandb 이름
        wandb.run.name = "{}_Fold_{}".format(args.name, k)

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
        loss_module = getattr(import_module("loss"), args.criterion)
        loss_ = loss_module()

        # optimizer
        trainable_params = filter(lambda p: p.requires_grad, model_.parameters())
        opt_module = getattr(import_module("torch.optim"), args.optimizer)
        optm = opt_module(trainable_params, lr=args.lr, weight_decay=5e-4)

        scheduler = StepLR(optm, args.lr_decay_step, gamma=0.5)
    
        ############################################# 학습 과정 ############################################################
        # train
        print("Start training {}-Fold.".format(k)) # 트레이닝 코드 이전 코드와 같음
#         model_.init_param() # initialize parameters # pretrain일 경우할 필요 없음
#         model_.load_weights("./darknet53.conv.74") # darknet만
        model_.train() # to train mode 
        
        prior_acc = 0
        high_acc_epoch = 0
        wandb.watch(model_)
        
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

                wandb.log({"loss":loss_val_avg,
                          "train_acc":train_accr*100,
                          "val_acc":test_accr*100})

                with open("./log/TrainLog_{}.txt".format(args.name), "a") as file: # 혹시 모를 백업
                    file.write("fold:[%d] epoch:[%d] loss:[%.3f] train_accr:[%.3f] test_accr:[%.3f].\n"%
                       (k, epoch,loss_val_avg,train_accr,test_accr))
                if prior_acc < test_accr: # 성능이 개선된 경우에만 저장
                    prior_acc = test_accr
                    high_acc_epoch = epoch
                torch.save(model_.state_dict(), "{}/{}/{}_Fold_model_weight{}.pt".format(save_path, args.name, k, epoch))

            scheduler.step()
        print ("{} Fold Done".format(k))
        EachFoldHighAcc.append(prior_acc * 100)
        EachFoldHighAccEpoch.append(high_acc_epoch)
    print("Average of FoldHighAcc : ", sum(EachFoldHighAcc)/len(EachFoldHighAcc)) # 각 fold별 에포크 마지막 값의 valid_acc 평균
    print("EachFoldHighAccEpoch : ", EachFoldHighAccEpoch) # 각 폴드별 가장 높은 에포크


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    
    # Data and Model checkpoints directories
    parser.add_argument("--epochs", type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument("--dataset", type=str, default="Dataset", help='dataset augmentation type (default: Dataset)')
    parser.add_argument("--resize", nargs="+", type=int, default=[380, 380], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--valid_batch_size', type=int, default=32, help='input batch size for validing (default: 32)')
    parser.add_argument('--model', type=str, default='efficient_b4', help='model type (default: efficient_b4)') # 내 전담 모델
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio for validaton (default: 0.1)')
    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss', help='criterion type (default: CrossEntropyLoss)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=1, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default=None, help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/total_class_aug.csv'), help="data csv file (default=/opt/ml/input/data/train/total_class_aug.csv)")
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './save_weights'), help="directory path having model files")

    args = parser.parse_args()
    
    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    main(args, args.data, args.model_dir)