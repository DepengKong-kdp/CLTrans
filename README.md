# (CLTrans) Self-Supervised Knowledge Mining from Unlabeled Data for Bearing Fault Diagnosis under Limited Annotations.
This is a PyTorch implementation of the paper entitled "Self-Supervised Knowledge Mining from Unlabeled Data for Bearing Fault Diagnosis under Limited Annotations".\
![CLTrans Framework](https://github.com/DepengKong-kdp/CLTrans/blob/main/results/fig1.png)

## Preparation
The following dependencies have been used in our implementation and validated as feasible.
````
python==3.6.8
torch==1.6.0
numpy==1.18.3
matplotlib==3.3.4
tqdm==4.45.0
argparse==1.1
````

## Self-Supervised Pre-Training
To do the self-supervised pre-training using CLTrans, run:
````
python pre_train.py --dataset_name [the dataset you want to use; Default CWRU]
                    --batch_size_pretrain [Default 256]
                    --lr [Default None]
                    --momentum [Default 0.9]
                    --weight_decay [Default 1e-4]
                    --num_epoch [Default 200]
                    --device [Default cuda]
````
You need to change the hyper-parameters according to your requirements, otherwise, use the default parameters.\
The pre-trained model parameters will be saved under the file folder `.\params\`.

## Knowlege Transfer
To fine-tune the pre-trained model and transfer the learned knowledge from one dataset to another, run:
````
python transfer_train.py --dataset_name_pretrain [the source-domain dataset]
                    --dataset_name [the target-domain dataset]
                    --batch_size_pretrain [Default 256]
                    --lr [Default 0.01]
                    --num_epoch [Default 100]
                    --device [Default cuda]
                    --load_params [Default True]
                    --percent [Default 1.0]
                    --num_labeled [Default 2400]
````
You need to change the hyper-parameters according to your requirements, otherwise, use the default parameters.\
The fine-tuned model parameters will be saved under the file folder `.\params\downstream\`.

## Visualization
To check the performance of the pre-training or the fine-tuning process, you can reduce the dimensionality of data from different datasets and visualize them by running:
````
python Dim_reduce.py --dataset_name_pretrain [the source-domain dataset]
                    --dataset_name [the target-domain dataset]
                    --load_params [Default False]
                    --finetune [Default True]
                    --device [Default cuda]
````
The results will be saved as `.png` under the file folder `./results`.

## Data
There are five datasets used in this work. Among them, the CWRU, MFPT, XJTU-SY, and the FEMTO-ST datasets are publicly available. The dataset used in Case 4 is made ourselves and it is availble by contacting us.
