
import numpy as np
from torch.utils.data import Dataset

class Pretrain_set(Dataset):
    def __init__(self, dataset_name, transforms=None):
        super(Pretrain_set, self).__init__()

        print('Loading dataset {} for pre-training ...'.format(dataset_name))
        data_path = "./datasets/{}/{}_dataset.npy".format(dataset_name, dataset_name)
        data      = np.load(data_path, allow_pickle=True)
        data      = data.item()['unlabeled_data']

        if transforms:
            print('     Augmenting data ...')
            self.data_trans1 = transforms(data).copy()
            self.data_trans2 = transforms(data).copy()
            print('     Augmentations Done.')
        else:
            self.data_trans1 = data
            self.data_trans2 = data
        print('Data loaded.')

    def __getitem__(self, item):
        trans1, trans2 = self.data_trans1[item], self.data_trans2[item]
        return trans1, trans2

    def __len__(self):
        return self.data_trans1.shape[0]


class Downstream_set(Dataset):
    def __init__(self, dataset_name, train=True, percent=1.0, num_labeled=2400):
        super(Downstream_set, self).__init__()

        self.train = train

        print('Loading dataset {} for downstream tasks ...'.format(dataset_name))
        data_path = "./datasets/{}/{}_dataset.npy".format(dataset_name, dataset_name)
        data0     = np.load(data_path, allow_pickle=True)
        data      = data0.item()['labeled_data']
        labels    = data0.item()['label']

        if train:
            self.train_data  = []
            self.train_label = []
        else:
            self.test_data   = []
            self.test_label  = []

        if dataset_name == 'CWRU':
            num_class = 10
        elif dataset_name == 'MFPT':
            num_class = 3
        else:
            raise AssertionError('Please ensure that the dataset name is correct!')
        for i in range(num_class):
            class_data  = data[i*3000 : (i+1)*3000]
            class_label = labels[i*3000 : (i+1)*3000]

            if train:
                self.train_data.append(class_data[:min(num_labeled, int(2400 * percent))])
                self.train_label.append(class_label[:min(num_labeled, int(2400 * percent))])
            else:
                self.test_data.append(class_data[2400:])
                self.test_label.append(class_label[2400:])

        if train:
            self.train_data  = np.vstack(self.train_data)
            self.train_label = np.hstack(self.train_label)
        else:
            self.test_data   = np.vstack(self.test_data)
            self.test_label  = np.hstack(self.test_label)
        print('Data loaded.')

    def __getitem__(self, item):
        if self.train:
            data, label = self.train_data[item], self.train_label[item]
        else:
            data, label = self.test_data[item], self.test_label[item]
        return data, label

    def __len__(self):
        if self.train:
            return self.train_data.shape[0]
        else:
            return self.test_data.shape[0]

if __name__ == '__main__':
    dataset = Downstream_set('CWRU')
    print(dataset.__len__())