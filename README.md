# image-classification-level1-13
Code for solution in AI Stage Image Classification Challenge.

## Hardware
The following specs were used to create the original solution.
- Nvidia V100

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
pip install -r requirements.txt
```

## Dataset Preparation
All required files include images are already in data directory.

In the beginning, the data is structured as:
```
/
  +- input
  |  +- data
  |  |  +- train
  |  |  |  +- images (including many ID folders)
  |  |  |  +- train.csv
  |  |  +- eval
  |  |  |  +- images
  |  |  |  +- info.csv
  +- code
  |  +- sample_submission.ipynb
  |  +- requirements.txt
```

### Generate CSV files & Augmentation Images
Run DataAug_Labeling.ipynb

## Training
Default setting is already done.
```console
$ python3 train_crossval.py --name "Exp name"
```

You can change many options.
Following content is default setting
```
parser.add_argument("--epochs", type=int, default=10, help='number of epochs to train (default: 10)')
parser.add_argument("--dataset", type=str, default="Dataset", help='dataset augmentation type (default: Dataset)')
parser.add_argument("--resize", nargs="+", type=int, default=[380, 380], help='resize size for image when training')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
parser.add_argument('--valid_batch_size', type=int, default=32, help='input batch size for validing (default: 32)')
parser.add_argument('--model', type=str, default='efficient_b4', help='model type (default: efficient_b4)')
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
parser.add_argument('--criterion', type=str, default='CrossEntropyLoss', help='criterion type (default: CrossEntropyLoss)')
parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
parser.add_argument('--log_interval', type=int, default=1, help='how many batches to wait before logging training status')
parser.add_argument('--name', default=None, help='model save at {SM_MODEL_DIR}/{name}')
parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/total_class_aug.csv'), help="data csv file (default=/opt/ml/input/data/train/total_class_aug.csv)")
parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './save_weights'), help="directory path having model files")
```

You can use wandb which is developer tools for machine learning.

Change code in train_crossval.py
```
run = wandb.init(project='Your Project Name', entity='Your Entity', reinit=True)
```

## Inference
If trained weights are prepared, you can create files that contains class probabilities of images.
Default setting is already done.
```console
$ python inference_crossval.py --name "Exp name"
```

You can change many options.
Following content is default setting
```
parser.add_argument("--dataset", type=str, default="Dataset", help='dataset augmentation type (default: Dataset)')
parser.add_argument("--resize", nargs="+", type=int, default=[380, 380], help='resize size for image when training')
parser.add_argument("--kfold", nargs="+", type=int, default=[0, 0, 0, 0, 0], help='high acc epoch num from each fold')
parser.add_argument('--model', type=str, default='efficient_b4', help='model type (default: efficient_b4)')
parser.add_argument('--name', default=None, help='model load at {WEIGHT_PATH}/{name}')
parser.add_argument('--weight_file', default='model_weight')
parser.add_argument('--test_dir', type=str, default=os.environ.get('SM_CHANNEL_TEST', '/opt/ml/input/data/eval'), help="test directory path (default=/opt/ml/input/data/eval)")
parser.add_argument('--weight_path', type=str, default=os.environ.get('WEIGHT_PATH', './save_weights'), help="directory path having weights files")
```

submission file saved at '/opt/ml/input/data/eval'(default)
