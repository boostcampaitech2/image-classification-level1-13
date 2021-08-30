import argparse
import random
import glob
import os
import multiprocessing
import re
import json
from importlib import import_module
from pathlib import Path
from copy import copy

import matplotlib as mplyWriter

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import wandb

import dataset
from dataset import MaskBaseDataset
from loss import create_criterion


def seed_everything(seed:int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN only use deterministic convolution algorithms.
    #  ??? cuDNN : The NVIDIA® CUDA® Deep Neural Network library™ (cuDNN) is a GPU-accelerated library of primitives for deep neural networks
    #  - ref_1(torch.are_deterministic_algorithms_enabled()) : https://pytorch.org/docs/stable/generated/torch.are_deterministic_algorithms_enabled.html#torch.are_deterministic_algorithms_enabled
    #  - ref_2(torch.use_deterministic_algorithms()) : https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms2
    torch.backends.cudnn.deterministic = True 

    # If True,
    # cuDNN benchmark multiple convolution algorithms and select the fastest.
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer:torch.optim.Optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def increment_path(path, exist_ok:bool = False):
    """ Automatically increment path (i.e.) runs/exp --> runs/exp0, runs/exp1 ... etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}"
        exist_ok (bool) : whether increment path (increment if False)
    """
    path = Path(path)

    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    """
    np_images 에 저장된 이미지 중 n 개를 뽑아서 figure 에 그린다.
    (정답과 예측 레이블을 표기해준다.)
    
    Args:
        np_images
        gts
        preds
        n
        shuffle
    """
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.sample(range(batch_size), k=n)
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T

    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for task, gt_label, pred_label
            in zip(tasks, gt_decoded_labels, pred_decoded_labels)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def train(data_dir:str, model_dir:str, args:argparse.Namespace):
    # Fix seed
    seed_everything(args.seed)

    # Set save directiory
    #  -> ./model/test1, ./model/test2, ./model/test3, ... 
    save_dir = increment_path(os.path.join(model_dir, args.model))

    # Set wandb run name as model_ver
    wandb.run.name = save_dir.split("/")[-1]

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    # default : BaseAugmentation
    dataset_module = getattr(import_module("dataset"), args.dataset)
    dataset = dataset_module(
        data_dir=data_dir
    )
    train_set, val_set = dataset.split_dataset()
    train_set.dataset = copy(dataset)
    num_classes = dataset.num_classes # 18 

    # -- augmentation
    # default : BaseAugmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std
    )
    base_transform_module = getattr(import_module("dataset"), "BaseAugmentation")
    base_transform = base_transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std
    )
    train_set.dataset.set_transform(transform)
    val_set.dataset.set_transform(base_transform)
    # dataset.set_transform(transform)

    # -- data_loader
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )


    # -- model
    # default : BaseModel
    model_module = getattr(import_module("model"), args.model)
    model = model_module(
        num_classes=num_classes
    ).to(device)

    # torch.nn.DataParallel : https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html?highlight=torch%20nn%20dataparallel#torch.nn.DataParallel
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion) # default : cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    # StepLR : https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html?highlight=steplr#torch.optim.lr_scheduler.StepLR
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # --logging
    # with tensorboard
    logger = SummaryWriter(log_dir=save_dir) # 모델 정보 로깅
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    
    best_val_acc = 0
    best_val_loss = np.inf
    figure_train_input = None
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0.0
        matches = 0
        
        for idx, train_batch in enumerate(train_loader, start=1):
            inputs, labels = train_batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            # print(outs)
            # torch.argmax : https://pytorch.org/docs/stable/generated/torch.argmax.html?highlight=argmax#torch.argmax
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if idx % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx}/{len(train_loader)}) || " 
                    f"training loss {train_loss:4.4f} || "
                    f"training accuracy {train_acc:4.2%} || "
                    f"lr {current_lr:4.5f}"
                )

                if figure_train_input is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure_train_input = grid_image(
                        inputs_np, labels, preds, n=16, 
                        shuffle=args.dataset not in ["MaskSplitByProfileDataset"]
                    )

                ## tensorborad logging (training)
                # logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx,)
                # logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                # wandb logging (training)
                wandb.log({"Train/loss" : train_loss,
                            "Train/accuracy" : train_acc * 100,
                            "Train/learning rate": current_lr})

                loss_value = 0.0
                matches = 0

        wandb.log({"Train/input": figure_train_input})
        scheduler.step() # lr 조정
        
        # val loop
        with torch.no_grad():
            print("Calculating validation results ...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs, labels = inputs.to(device), labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (preds == labels).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, 
                        shuffle=args.dataset not in ["MaskSplitByProfileDataset"]
                    )
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            # best_val_acc = min(best_val_acc, val_acc)
            
            if val_acc > best_val_acc:
                print(
                    f"\n  > New Best Model for val accuracy : {val_acc:4.2%}! "
                    f"saving the best model ... "
                )
                torch.save(model.module.state_dict(), os.path.join(save_dir, "best.pth"))
                best_val_acc = val_acc
            # model backup 
            torch.save(model.module.state_dict(), os.path.join(save_dir, "last.pth"))
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )

            ## tensorboard logging (validation)
            # logger.add_scalar("Val/loss", val_loss, epoch)
            # logger.add_scalar("Val/accuracy", val_acc, epoch)
            # logger.add_figure("results", figure, epoch)

            # wandb logging (validation)
            wandb.log({"Val/loss" : val_loss,
                        "Val/accuracy" : val_acc * 100,
                        "results" : figure})
            print()

    print("END of train.")


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()

    # from dotenv import load_dotenv
    import os

    # Data and Model checkpoints directories
    parser.add_argument("--seed", type=int, default=42, help='random seed (default: 42)')
    parser.add_argument("--epochs", type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument("--dataset", type=str, default="MaskBaseDataset", help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument("--augmentation", type=str, default="BaseAugmentation", help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'), help="directory path having datas (default=/opt/ml/input/data/train/images)")
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'), help="directory path having model files")

    
    args = parser.parse_args()
    

    data_dir = args.data_dir
    model_dir = args.model_dir

    # wandb init 
    ###################################################
    # project 이름은 따로 설정해야합니다!
    wandb.init(project='seunghun', entity='13ai')
    ###################################################
    wandb.config.update(args)
    
    # start train
    train(data_dir, model_dir, args)
