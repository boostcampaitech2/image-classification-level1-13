# image-classification-level1-13
Code for solution in AI Stage Image Classification Challenge.

## Hardware
The following specs were used to create the original solution.
- Nvidia V100

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n hpa python=3.6
source activate hpa
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

### Generate CSV files
*You can skip this step. All CSV files are prepared in data directory.*

#### Duplicated Image List
There are duplicated images. To search them, run following commands. *duplicates.ahash.csv* and *duplicates.phash.csv* will be generated.
```
$ python tools/find_duplicate_images.py
```

#### Split Dataset
Create 5 folds CV set. One for training, the other for searching augmentation. *split.stratified.[0-4].csv* and *split.stratified.small.[0-4].csv* will be generated.
```
$ python stratified_split.py
$ python stratified_split.py --use_external=0
```

#### Search Data Leak
To learn more about data leak, please, refer to [this post](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/72534). Following comand will create *data_leak.ahash.csv* and *data_leak.phash.csv*. [The other leak](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/73395y) is already in *data* directory.
```
$ python find_data_leak.py
```

## Training
In configs directory, you can find configurations I used train my final models. My final submission is ensemble of resnet34 x 5, inception-v3 and se-resnext50, but ensemble of inception-v3 and se-resnext50's performance is better.

### Search augmentation
To find suitable augmentation, 256x256 image and resnet18 are used.
It takes about 2 days on TitanX. The result(best_policy.data) will be located in *results/search* directory.
The policy that I used is located in *data* directory.
```
$ python train.py --config=configs/search.yml
```

### Train models
To train models, run following commands.
```
$ python train.py --config={config_path}
```
To train all models, run `sh train.sh`

The expected training times are:

Model | GPUs | Image size | Training Epochs | Training Time
------------ | ------------- | ------------- | ------------- | -------------
resnet34 | 1x TitanX | 512 | 40 | 16 hours
inception-v3 | 3x TitanX | 1024 | 27 | 1day 15 hours
se-resnext50 | 2x TitanX | 1024 | 22 | 2days 15 hours

### Average weights
To average weights, run following commands.
```
$ python swa.py --config={config_path}
```
To average weights of all models, simply run `sh swa.sh`
The averages weights will be located in *results/{train_dir}/checkpoint*.

### Pretrained models
You can download pretrained model that used for my submission from [link](https://www.dropbox.com/s/qo65gw8kml5hgag/results.tar.gz?dl=0). Or run following command.
```
$ wget https://www.dropbox.com/s/qo65gw8kml5hgag/results.tar.gz
$ tar xzvf results.tar.gz
```
Unzip them into results then you can see following structure:
```
results
  +- resnet34.0.policy
  |  +- checkpoint
  +- resnet34.1.policy
  |  +- checkpoint
  +- resnet34.2.policy
  |  +- checkpoint
  +- resnet34.3.policy
  |  +- checkpoint
  +- resnet34.4.policy
  |  +- checkpoint
  +- inceptionv3.attention.policy.per_image_norm.1024
  |  +- checkpoint
  +- se_resnext50.attention.policy.per_image_norm.1024
  |  +- checkpoint
```

## Inference
If trained weights are prepared, you can create files that contains class probabilities of images.
```
$ python inference.py \
  --config={config_filepath} \
  --num_tta={number_of_tta_images, 4 or 8} \
  --output={output_filepath} \
  --split={test or test_val}
```
To make submission, you must inference test and test_val splits. For example:
```
$ python inference.py --config=configs/resnet34.0.policy.yml --num_tta=8 --output=inferences/resnet34.0.test_val.csv --split=test_val
$ python inference.py --config=configs/resnet34.0.policy.yml --num_tta=8 --output=inferences/resnet34.0.test.csv --split=test
```
To inference all models, simply run `sh inference.sh`

## Make Submission
Following command will ensemble of all models and make submissions.
```
$ python make_submission.py
```
If you don't want to use, modify *make_submission.py*.
For example, if you want to use inception-v3 and se-resnext50 then modify *test_val_filenames, test_filenames and weights* in *make_submission.py*.
```
test_val_filenames = ['inferences/inceptionv3.0.test_val.csv',
                      'inferences/se_resnext50.0.test_val.csv']
                      
test_filenames = ['inferences/inceptionv3.0.test.csv',
                  'inferences/se_resnext50.0.test.csv']
                  
weights = [1.0, 1.0]
```
The command generate two files. One for original submission and the other is modified using data leak.
- submissions/submission.csv
- submissions/submission.csv.leak.csv

