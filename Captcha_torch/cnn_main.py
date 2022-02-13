import math
import os
import torch
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils import data
from torch import nn, optim
from CNN_NET import CNNNet, DNNNet
from tqdm import tqdm


class DataSet(data.Dataset):

    def __init__(self, data_, label_):
        self.data_ = torch.from_numpy(np.expand_dims(np.array(data_), axis=1)) / 255.
        self.label_ = torch.from_numpy(label_).long()

    def __getitem__(self, index_):
        Data_ = self.data_[index_]
        Lable_ = self.label_[index_]

        return [Data_, Lable_]

    def __len__(self):
        return len(self.data_)


def get_data(path_):
    h5 = h5py.File(path_, 'r')
    images_data = h5['images_data'][:]
    images_label = h5['images_label'][:]
    h5.close()

    np.random.seed(1)
    np.random.shuffle(images_data)
    np.random.seed(1)
    np.random.shuffle(images_label)

    data_x_train = images_data[:450]
    data_y_train = images_label[:450]
    data_x_test = images_data[450:]
    data_y_test = images_label[450:]

    train_dataset = DataSet(data_x_train, data_y_train)
    test_dataset = DataSet(data_x_test, data_y_test)

    return train_dataset, test_dataset


def train(epoch):
    model.train()
    global best_loss_train, best_acc_train
    loss_train = 0.0
    acc_train = 0.0
    for step, (input_, label_) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input_, label_ = input_.to(device), label_.to(device)
        output_ = model(input_)
        loss = loss_fn(output_.reshape(-1, num_class), label_.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_train += loss.item()
        _, label_pred = output_.max(-1)

        acc_ = torch.where((label_pred == label_).sum(-1) < 4, 0, 1).sum().item()
        acc_train += acc_

    # Save checkpoint. best_train_loss
    if save_weights_loss_train and (loss_train / len(train_loader) < best_loss_train):
        print('Saving best train loss...')
        best_loss_train = loss_train / len(train_loader)
        a, b = math.modf(best_loss_train)
        model_path = os.path.join(save_weights_path,
                                  '{}train_ckpt_loss{:0>4d}{:0>4d}.pth'.format(epoch, int(b), int(a * 10000)))
        torch.save(model.state_dict(), model_path)

    # Save checkpoint. best_train_acc
    if save_weights_acc_train and (acc_train / train_set.__len__() > best_acc_train):
        print('Saving best train acc...')
        best_acc_train = acc_train / train_set.__len__()
        a, b = math.modf(best_acc_train * 100)
        model_path = os.path.join(save_weights_path,
                                  '{}train_ckpt_acc{:0>4d}{:0>4d}.pth'.format(epoch, int(b), int(a * 10000)))
        torch.save(model.state_dict(), model_path)

    # print('Train Loss: %.3f | Acc-1: %.3f |' % (loss_train / len(train_loader), acc_train / len(train_loader)))
    return loss_train / len(train_loader), acc_train / train_set.__len__()


def test(epoch):
    model.eval()
    global best_loss_test, best_acc_test
    loss_test = 0.0
    acc_test = 0.0
    for step, (input_, label_) in tqdm(enumerate(test_loader), total=len(test_loader)):
        input_, label_ = input_.to(device), label_.to(device)
        with torch.no_grad():
            output_ = model(input_)
        loss = loss_fn(output_.reshape(-1, num_class), label_.reshape(-1))

        loss_test += loss.item()
        _, label_pred = output_.max(-1)
        acc_ = torch.where((label_pred == label_).sum(-1) < 4, 0, 1).sum().item()
        acc_test += acc_

        # Save checkpoint. best_test_loss
    if save_weights_loss_test and (loss_test / len(test_loader) < best_loss_test):
        print('Saving best test loss...')
        best_loss_test = loss_test / len(test_loader)
        a, b = math.modf(best_loss_test)
        model_path = os.path.join(save_weights_path,
                                  '{}test_ckpt_loss{:0>4d}{:0>4d}.pth'.format(epoch, int(b), int(a * 10000)))
        torch.save(model.state_dict(), model_path)

    # Save checkpoint. best_test_acc
    if save_weights_acc_test and (acc_test / test_set.__len__() > best_acc_test):
        print('Saving best test acc...')
        best_acc_test = acc_test / test_set.__len__()
        a, b = math.modf(best_acc_test * 100)
        model_path = os.path.join(save_weights_path,
                                  '{}test_ckpt_acc{:0>4d}{:0>4d}.pth'.format(epoch, int(b), int(a * 10000)))
        torch.save(model.state_dict(), model_path)

    # print('Test Loss: %.3f | Acc-1: %.3f |' % (loss_test / len(test_loader), acc_test / len(test_loader)))
    return loss_test / len(test_loader), acc_test / test_set.__len__()


def save_plot_result(dict_result):

    fig1 = plt.figure(figsize=(20, 8), dpi=80)
    plt.plot(range(len(dict_result["loss_train"])), dict_result["loss_train"], label='train_loss')
    plt.plot(range(len(dict_result["loss_test"])), dict_result["loss_test"], color='red', label='test_loss')
    plt.legend()
    plt.title('loss')
    fig1.savefig('loss.pdf')
    plt.show()

    fig2 = plt.figure(figsize=(20, 8), dpi=80)
    plt.plot(range(len(dict_result["acc_train"])), dict_result["acc_train"], label='train_acc')
    plt.plot(range(len(dict_result["acc_test"])), dict_result["acc_test"], color='red', label='test_acc')
    plt.legend()
    plt.title('acc')
    fig2.savefig('acc.pdf')
    plt.show()


if __name__ == '__main__':

    Epoch = 600
    batch_size = 16
    save_weights_loss_train = False
    save_weights_acc_train = False
    save_weights_acc_test = False
    save_weights_loss_test = False
    load_weights = False
    best_loss_train = 10.0
    best_loss_test = 10.0
    best_acc_train = 0.0
    best_acc_test = 0.0
    num_class = 10
    save_weights_path = './checkpoint_CNN'
    weight_path = './checkpoint/ckpt_loss00000010.pth'
    path_data = r'train_images_pro.h5'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_set, test_set = get_data(path_data)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # model = CNNNet()
    model = DNNNet(22 * 62, 512, 256, 40)
    criterion = nn.CrossEntropyLoss()
    if load_weights:
        model.load_state_dict(torch.load(weight_path))
        print("loading weights success")
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, eta_min=0.001)

    if save_weights_loss_train or save_weights_acc_train or save_weights_acc_test or save_weights_loss_test:
        if not os.path.isdir(save_weights_path):
            os.mkdir(save_weights_path)

    dict_save = {"Epoch": [], "loss_train": [], "acc_train": [], "loss_test": [], "acc_test": []}
    for epoch in range(Epoch):
        loss_train, acc_train = train(epoch)
        loss_test, acc_test = test(epoch)
        scheduler.step()  # update lr
        print('epoch: {}, train_loss: {:.4f}, train_acc: {:.4f}, test_loss: {:.4f}, test_acc: {:.4f}'.format(
            epoch, loss_train, acc_train, loss_test, acc_test))
        print('lr: {:.8f}'.format(scheduler.get_last_lr()[0]))

        dict_save["Epoch"].append(epoch)
        dict_save["loss_train"].append(loss_train)
        dict_save["acc_train"].append(acc_train)
        dict_save["loss_test"].append(loss_test)
        dict_save["acc_test"].append(acc_test)

        # if acc_train > 0.9 and acc_test > 0.9:
        #     a1, b1 = math.modf(acc_train * 100)
        #     a2, b2 = math.modf(acc_test * 100)
        #     if not os.path.isdir(save_weights_path):
        #         os.mkdir(save_weights_path)
        #     model_path = os.path.join(save_weights_path,
        #                               '{}train_acc{:0>4d}{:0>4d}test_acc{:0>4d}{:0>4d}.pth'.format(
        #                                   epoch, int(b1), int(a1 * 10000), int(b2), int(a2 * 10000)))
        #     torch.save(model.state_dict(), model_path)
        #     print('save weights success')

    save_plot_result(dict_save)
    df_history = pd.DataFrame(dict_save)
    df_history.to_csv("./history_cnn.csv", index=False)
