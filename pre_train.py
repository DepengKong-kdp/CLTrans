
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tqdm
from Utils.Datasets import Pretrain_set
from Utils.Contrastive_framework import CLTrans
from Utils.Augmentations import transforms


def train(opt):
    # define train loader
    train_set = Pretrain_set(opt.dataset_name, transforms)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size_pretrain, shuffle=True)
    # define model
    model = CLTrans()
    model.to(opt.device)
    criterion = nn.CosineSimilarity(dim=1).to(opt.device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay
                                )
    min_loss = 1e5
    xx =[]
    count_x = 0
    loss_list = []
    one_list = []
    z_std_list = []
    std_list = []
    plt.ion()
    for epoch in range(opt.num_epoch):
        epoch_loss = 0.0
        for i, (x1, x2) in enumerate(tqdm.tqdm(train_loader)):
            x1, x2 = x1.to(opt.device).float(), x2.to(opt.device).float()

            # compute output and loss
            p1, p2, z1, z2 = model(x1, x2)
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

            epoch_loss += loss.item()

            # compute gradient and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                # compute the std of the learned representation z1, ideally: 1/sqrt(50) ≈ 0.141
                z_norm2 = torch.sqrt(torch.sum(z1 * z1, dim=1))
                z_std_ = z1 / z_norm2.unsqueeze(1)
                z_std = np.std(z_std_.cpu().numpy(),1)
                z_std_list.append(z_std.mean())
                std_list.append(0.125)
                loss_list.append(loss.item())
                one_list.append(-1)
                # plot std of z1 and batch loss
                count_x += 1
                xx.append(count_x/49)
                plt.clf()
                plt.subplot(2,1,1)
                plt.plot(xx, loss_list,label='epoch loss')
                plt.plot(xx, one_list,label='Theoretical minimum',c='grey',linewidth=1)
                plt.legend()
                plt.subplot(2,1,2)
                plt.plot(xx, z_std_list,label='std of z1')
                plt.plot(xx, std_list,label='Theoretical std',c='grey',linewidth=1)
                plt.legend()
                plt.pause(0.1)
        epoch_loss = epoch_loss/(i+1)
        if epoch_loss <= min_loss:
            min_loss = epoch_loss
            torch.save(model.state_dict(),'./params/CLTrans_{}-best-BN.pkl'.format(opt.dataset_name))
        print('Epoch: {}/{}  epoch loss: {:.3f}'.format(epoch+1, opt.num_epoch, epoch_loss))
    torch.save(model.state_dict(), './params/SimSiam_{}-last-BN.pkl'.format(opt.dataset_name))
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(xx, loss_list, label='epoch loss')
    plt.plot(xx, one_list, label='Theoretical minimum', c='grey',linewidth=1)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(xx, z_std_list, label='std of z1')
    plt.plot(xx, std_list, label='Theoretical std', c='grey',linewidth=1)
    plt.legend()
    plt.savefig('./results/SimSiam_{}-{}.png'.format(opt.dataset_name, opt.batch_size_support), dpi=600)

import argparse
def get_options():
    args = argparse.ArgumentParser()

    args.add_argument('--dataset_name', default='CWRU', type=str, help='Name of the dataset.')
    args.add_argument('--batch_size_pretrain', default=256, type=int, help='Batch size of the pre-training process.')
    args.add_argument('--lr', default=None, type=float, help='Learning rate of the pre-training process')
    args.add_argument('--momentum', default=0.9, type=float, help='Momentum of the optimizer.')
    args.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay of the optimizer.')
    args.add_argument('--num_epoch', default=200, type=int, help='Number of epochs.')
    args.add_argument('--device', default='cuda', type=str, help='Device where the model is located.')

    args = args.parse_args()
    if args.lr is None:
        args.lr = 0.05 * args.batch_size_pretrain * 0.1 / 256
    return args

if __name__ == '__main__':
    options = get_options()
    print('\n' + '=' * 50)
    print('Hyper parameters settings:')
    print('=' * 50)
    for key in vars(options).keys():
        print('{:<20}: {}'.format(key, eval('options.{}'.format(key))))
        print('-'*50)
    print('\n')

    train(options)

