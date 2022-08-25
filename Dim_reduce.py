
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.utils.data import DataLoader
import argparse
import tqdm
from MulticoreTSNE import MulticoreTSNE as TSNE

from Utils.Contrastive_framework import transfer_model
from Utils.Datasets import Downstream_set

def get_options():
    args = argparse.ArgumentParser()

    args.add_argument('--dataset_name_pretrain', default='CWRU', type=str, help='Name of the dataset for pre-training.')
    args.add_argument('--dataset_name', default='CWRU', type=str, help='Name of the dataset for downstream task.')
    args.add_argument('--load_params', default=False, type=bool, help='Load the pre-trained parameters.')
    args.add_argument('--finetune', default=True, type=bool, help='Load the fine-tuned model parameters. If false, then load the pre-train model parameters only')
    args.add_argument('--device',default='cuda', type=str, help='Device where the model is located.')

    args = args.parse_args()
    return args
options = get_options()

class_num = {'CWRU': 10, 'MFPT': 3}
save_flag = ''
if options.finetune:
    save_flag = 'finetune'
else:
    save_flag = 'pretrain'

test_set    = Downstream_set(options.dataset_name, train=False)
test_loader = DataLoader(test_set, batch_size=512)

model = transfer_model(options.dataset_name_pretrain, load_params=~options.finetune)
# loader transferred model parameters
if options.finetune:
    model.load_state_dict(torch.load('./params/downstream/{}-transfer-{}.pkl'.format(options.dataset_name_pretrain, options.dataset_name)))
model.to(options.device)

model.eval()
data_arr    = []
labels_arr  = []
feature_arr = []
for data, labels in tqdm.tqdm(test_loader):
    data, labels = data.to(options.device).float(), labels.to(options.device).long()

    # forward
    output, encoded = model(data)
    data_arr.append(data.squeeze().detach().cpu().numpy())
    feature_arr.append(encoded.squeeze().detach().cpu().numpy())
    labels_arr.append(labels.detach().cpu().numpy())

data_arr    = np.vstack(data_arr)
feature_arr = np.vstack(feature_arr)
labels_arr  = np.hstack(labels_arr)

# dimension reduction
reduced_data = TSNE(n_jobs=16).fit_transform(feature_arr)

Oranges = cm.get_cmap('Paired', 50)
color_list = np.linspace(0.1, 0.9, class_num[options.dataset_name])
plt.figure(figsize=(8,6))
for i, item in enumerate(tqdm.tqdm(reduced_data)):
    plt.scatter(item[0], item[1], c=Oranges(color_list[class_num[options.dataset_name]-1-labels_arr[i]]), s=10)
plt.xticks([])
plt.yticks([])
plt.savefig('./results/{}_{}-{}.png'.format(options.dataset_name_pretrain, options.dataset_name, save_flag), dpi=600)
plt.show()