import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys
from icecream import ic
import torch
import torch.nn as nn
from torch.autograd import Variable
from itertools import cycle
from tqdm import tqdm
torch.manual_seed(24)

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix


from Model import DenseNet3
from Dataset import Fashion_MNIST, MNIST, Cifar_10, SVHN, TinyImagenet_r, \
    TinyImagenet_c, Fashion_MNIST_17, Fashion_MNIST_89, SVHN_07, SVHN_89, MNIST_IND, MNIST_OOD
from WOOD_Loss import NLLWOOD_Loss_v2, sink_dist_test_v2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Let's use", torch.cuda.device_count(), "GPUs!")

##parameters, Fashion MNIST is in distribution dataset, and MNIST is out of distribution dataset


beta = torch.Tensor([float(sys.argv[1])]).to(device)
num_epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])
InD_batch_size = int(sys.argv[4])
InD_Dataset = str(sys.argv[5])
OOD_Dataset = str(sys.argv[6])
C = int(sys.argv[7])

# Additional cmd line arguments processing
EXP_NAME = str(sys.argv[8])
n_ood = int(sys.argv[9])
n_cls = int(sys.argv[10])

OOD_batch_size = batch_size - InD_batch_size
OOD_batch_size = min(OOD_batch_size, n_ood) # Fix this bug later
ic(f"OOD batch size: {OOD_batch_size}.")

test_batch_size = 100
learning_rate = 0.001
##parameters in loss
num_class = torch.LongTensor([n_cls]).to(device)

data_dic = {
    'MNIST': MNIST,
    'FashionMNIST': Fashion_MNIST, 
    'Cifar10': Cifar_10,
    'SVHN': SVHN, 
    'Imagenet_r': TinyImagenet_r,
    'Imagenet_c': TinyImagenet_c,
    'FashionMNIST-17': Fashion_MNIST_17,
    'FashionMNIST-89': Fashion_MNIST_89,
    'SVHN-07': SVHN_07,
    'SVHN-89': SVHN_89,
    'MNIST-IND': MNIST_IND,
    'MNIST-OOD': MNIST_OOD
}

InD_train_loader, InD_test_loader = data_dic[InD_Dataset](InD_batch_size, test_batch_size)
OOD_train_loader, OOD_test_loader = data_dic[OOD_Dataset](OOD_batch_size, test_batch_size)


# Configure Path
ood_path = os.path.join('..', 'Out-of-Distribution-GANs', 'checkpoint', 'OOD-Sample', EXP_NAME, f"OOD-Balanced-{n_ood}.pt")
ood_img_batch, ood_img_label = torch.load(ood_path)
ood_data = list(zip(ood_img_batch, ood_img_label))

OOD_train_loader = torch.utils.data.DataLoader(ood_data, batch_size=OOD_batch_size, shuffle=True)

file_root = './runs/' + f"{EXP_NAME}" + f'/{n_ood}/'
os.makedirs(file_root, exist_ok=True)
file_name = file_root + 'log.txt'


f = open(file_name, 'w')
f.write('DenseNet 100 f ' + InD_Dataset + ' InD ' + OOD_Dataset + ' OOD experiment epoch = ' + str(num_epochs) + ' beta = ' + str(beta[0]) + ' OOD Size = ' + str(OOD_batch_size) + '\n')

tpr95_lst = []
tpr99_lst = []
mc = 3

for mc_num in range(mc):
    ##load model
    model = DenseNet3(depth=100, num_classes=n_cls, input_channel = C)
    model.to(device)
    model = nn.DataParallel(model)

    ##load loss function
    NLLWOOD_l = NLLWOOD_Loss_v2.apply
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # print(model)

    # Lists for knowing classwise accuracy
    predictions_list = []
    labels_list = []
    best_test_acc = 0
    best_tpr95 = 0
    best_tpr99 = 0
    best_then01_dist = 1

    for epoch in tqdm(range(num_epochs)):
        count = 0
        for (InD_images, InD_labels), (OOD_images, OOD_labels) in zip(InD_train_loader, cycle(OOD_train_loader)):
        #for InD_images, InD_labels in InD_train_loader:
            model.train()
            
            ##load a batch of ood data
            ##change the label of ood data
            OOD_labels[:] = num_class[0]

            images = torch.cat([InD_images, OOD_images], dim=0)
            labels = torch.cat([InD_labels, OOD_labels], dim=0)

            ##shuffle the order of InD and OOD samples
            idx = torch.randperm(images.shape[0])
            images = images[idx].view(images.size())
            labels = labels[idx].view(labels.size())

            # Transfering images and labels to GPU if available
            images, labels = images.to(device), labels.to(device)

            train = Variable(images)
            #print(train.shape)
            labels = Variable(labels)

            # Forward pass 
            outputs = model(train)
            
            loss = NLLWOOD_l(outputs, labels, num_class, beta, device)

            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            optimizer.zero_grad()

            # Propagating the error backward
            loss.backward()

            # Optimizing the parameters
            optimizer.step()

            count += 1
            
            
            # Testing the model
            if not (count % 800):    # It's same as "if count % 100 == 0"
                total = 0
                correct = 0
                InD_test_sink_dist_list = []
                model.eval()
                ##InD samples test
                for images, labels in InD_test_loader:
                    images, labels = images.to(device), labels.to(device)
                    labels_list.append(labels)

                    test = Variable(images.view(images.size()))

                    outputs = model(test)


                    InD_sink_dist = sink_dist_test_v2(outputs, labels, num_class[0], device).cpu().detach().numpy()
                    InD_test_sink_dist_list.append(InD_sink_dist)


                    predictions = torch.max(outputs, 1)[1].to(device)
                    predictions_list.append(predictions)

                    correct += (predictions == labels).sum()

                    total += len(labels)
                
                InD_test_mean_sink_dist = np.concatenate(InD_test_sink_dist_list, axis=0)
                InD_sink_mean = np.mean(InD_test_mean_sink_dist)
                
                accuracy = correct * 100 / float(total)

                if accuracy >= best_test_acc:
                    best_test_acc = accuracy
                    torch.save(model.state_dict(), f'%s/acc_{mc_num}_model.t7' % file_root)

                ##OOD samples test
                OOD_test_sink_dist_list = []
                total = 0
                for images, labels in OOD_test_loader:
                    labels[:] = 0
                    images, labels = images.to(device), labels.to(device)
                    labels_list.append(labels)

                    test = Variable(images.view(images.size()))

                    outputs = model(test)
                    total += len(labels)
                    OOD_sink_dist = sink_dist_test_v2(outputs, labels, num_class[0], device).cpu().detach().numpy()
                    OOD_test_sink_dist_list.append(OOD_sink_dist)
                
                OOD_test_mean_sink_dist = np.concatenate(OOD_test_sink_dist_list, axis=0)
                OOD_sink_mean = np.mean(OOD_test_mean_sink_dist)

                # 0.95 TNR
                thresh95 = np.quantile(InD_test_mean_sink_dist, 0.95)
                tpr95 = 1 - OOD_test_mean_sink_dist[OOD_test_mean_sink_dist<=thresh95].shape[0] / float(OOD_test_mean_sink_dist.shape[0])
                thresh95 = float(format(thresh95, '.4g'))
                tpr95 = float(format(tpr95, '.4g'))

                # 0.99 TNR
                thresh99 = np.quantile(InD_test_mean_sink_dist, 0.99)
                tpr99 = 1 - OOD_test_mean_sink_dist[OOD_test_mean_sink_dist<=thresh99].shape[0] / float(OOD_test_mean_sink_dist.shape[0])
                thresh99 = float(format(thresh99, '.4g'))
                tpr99 = float(format(tpr99, '.4g'))

                accuracy = float(format(accuracy, '.4g'))
                
                log = "Epoch: {}, Iteration: {}, Loss: {}, Accuracy: {}%, InD_sink: {}, OOD_sink: {}, OOD_95_TPR: {}, OOD_99_TPR: {}".format(epoch, count, loss[0], accuracy, InD_sink_mean, OOD_sink_mean, tpr95, tpr99)
                print(log)
                ic(log)

                f.write(log+'\n')
                
                if tpr99 >= best_tpr99:
                    best_tpr99 = tpr99

                if tpr95 >= best_tpr95:
                    best_tpr95 = tpr95
                    torch.save(model.state_dict(), f'%s/OOD_model_{mc_num}.t7' % file_root)
    
    f.write(f"\n MC Iteration {mc_num} ends.\n")
    tpr95_lst.append(best_tpr95)
    tpr99_lst.append(best_tpr99)

f.write(f"\n TPR at 95 TNR: {tpr95_lst}")
f.write(f"\n TPR at 95 TNR MEAN: {np.mean(tpr95_lst)}")
mad95 = np.mean(np.abs(np.mean(tpr95_lst) - np.array(tpr95_lst)))
f.write(f"\n TPR at 95 TNR MAD: {mad95}")

f.write(f"\n TPR at 99 TNR: {tpr99_lst}")
f.write(f"\n TPR at 99 TNR MEAN: {np.mean(tpr99_lst)}")
mad99 = np.mean(np.abs(np.mean(tpr99_lst) - np.array(tpr99_lst)))
f.write(f"\n TPR at 99 TNR MAD: {mad99}")
f.close()








