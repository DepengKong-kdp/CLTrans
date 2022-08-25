import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

from Utils.Contrastive_framework import CLTrans, transfer_model
from Utils.Datasets import Downstream_set

def transfer_train(opt):
    # define model
    model = transfer_model(opt.dataset_name_pretrain, opt.load_params).to(opt.device)

    # define data sets & loader
    train_set    = Downstream_set(opt.dataset_name, train=True, percent=opt.percent, num_labeled=opt.num_labeled)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
    test_set     = Downstream_set(opt.dataset_name, train=False)
    test_loader  = DataLoader(test_set, batch_size=512, shuffle=True)
    print('Train_set: {} \nTest_set: {}'.format(len(train_set), len(test_set)))

    optimizer    = torch.optim.Adam([
        {'params': model.backbone.parameters(), 'lr':opt.lr * 0.001},
        {'params': model.fc.parameters(), 'lr': opt.lr},
    ])
    criterion    = nn.CrossEntropyLoss()

    # main loop
    best_acc = 0.0
    for epoch in range(opt.num_epoch):
        # train loop
        num_corrects = 0
        epoch_loss   = 0.0
        model.train(True)
        model.backbone.eval()
        for data, labels in tqdm.tqdm(train_loader):
            data, labels = data.to(opt.device).float(), labels.to(opt.device).long()

            # forward
            output,_ = model(data)
            loss     = criterion(output, labels)
            epoch_loss   += loss.item()
            _, preds      = torch.max(output, dim=1)
            num_corrects += torch.sum(preds == labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_acc = float(num_corrects) / len(train_set) * 100

        num_corrects = 0
        model.eval()
        for data, labels in test_loader:
            data, labels = data.to(opt.device).float(), labels.to(opt.device).long()

            # forward
            output,_ = model(data)
            _, preds = torch.max(output, dim=1)
            num_corrects += torch.sum(preds == labels)
        test_acc = float(num_corrects) / len(test_set) * 100
        if best_acc < test_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), './params/downstream/{}-transfer_to-{}.pkl'.format(opt.dataset_name_pretrain, opt.dataset_name))
        print('Epoch: {}/{}, loss: {:.3f}, train Acc: {:.3f}%, test Acc: {:.3f}%, best Acc: {:.3f}%.'.format(epoch+1, opt.num_epoch, epoch_loss, train_acc, test_acc, best_acc))

        if (epoch + 1)%10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
    return train_acc, best_acc

import argparse
def get_options():
    args = argparse.ArgumentParser()

    args.add_argument('--dataset_name_pretrain', default='CWRU', type=str, help='Name of the dataset for pre-training.')
    args.add_argument('--dataset_name', default='CWRU', type=str, help='Name of the dataset for downstream task.')
    args.add_argument('--batch_size', default=256, type=int, help='Batch size.')
    args.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    args.add_argument('--num_epoch', default=100, type=int, help='Number of epochs.')
    args.add_argument('--load_params', default=True, type=bool, help='If load the pre-trained model parameters.')
    args.add_argument('--device', default='cuda', type=str, help='Device where the model is located.')
    args.add_argument('--percent', default=1.0, type=float, help='Percent of used labeled training data.')
    args.add_argument('--num_labeled', default=2400, type=int, help='Amount of used labeled training data.')

    args = args.parse_args()
    return args

if __name__ == '__main__':
    options = get_options()
    train_acc, best_acc = transfer_train(options)
    print('Training process completed: \n'
          '   the training accuracy = {:.3f}%\n'
          '   the best test accuracy = {:.3f}%.'.format(train_acc, best_acc))