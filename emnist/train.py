# Imports
import pickle
import re
import zipfile
from collections import defaultdict
from math import sqrt
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tqdm
from ridgeplot import ridge_plot
from torchvision.datasets import EMNIST
from vgg import VGG_net
from xai import *

torch.multiprocessing.set_sharing_strategy("file_system")
import os


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    batch_size = 128
    ghost_samples = 40
    trainset = WrappedDataset(
        EMNIST(
            root="mnist/data",
            train=True,
            download=True,
            transform=transform,
            split="byclass",
        )
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_sampler=ClassSampler(trainset, batch_size, True, ghost_samples),
        num_workers=28,
    )
    testset = WrappedDataset(
        EMNIST(
            root="mnist/data",
            train=False,
            download=True,
            transform=transform,
            split="byclass",
        )
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=8
    )
    # problem mit shuffle true
    # brauchen eigenen sampler eventuell?
    # so? https://discuss.pytorch.org/t/index-concept-in-torch-utils-data-dataloader/72449/6
    # https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.__getitem__
    print("loaded data...")
    CLASS_NAMES = trainset.ds.classes
    net = VGG_net(in_channels=1,vgg_type="VGG9").to_sequential()
    # net.load_state_dict(torch.load(os.environ["MODEL_FILE"]))
    net = net.to(device)

    # optimizer = torch.optim.SGD(net.parameters(), lr=0.00666, weight_decay=0.0340)
    optimizer = WrappedOptimizer(
        torch.optim.SGD, history_file=os.environ["HISTORY_FILE"]
    )(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    loss = nn.CrossEntropyLoss(reduction="none")
    # loss = nn.BCEWithLogitsLoss()

    for epoch_num in range(int(os.environ["EPOCHS"])):
        print("starting epoch: " + str(epoch_num))
        correct = 0
        total = 0

        net.train()  # train mode
        pbar = tqdm.tqdm(trainloader)  # show progress bar
        for data in pbar:
            # get the inputs; data is a list of [inputs (1 image 4 dimension WxHxCxB), labels (tensor 1 dimension array of length 128, value range 0-9), index of batch (tensor 1 dimension array of length 128?, value large)]
            inputs, labels, ind = data
            inputs, labels = inputs.to(device), labels.to(device)
            if len(labels) < 128:
                continue
            # print(CLASS_NAMES[labels[:1].item()])
            # contributions, preactivations, cosines, dot_products, norms, l = explain(net, inputs[:1])
            # classes, weights = class_statistics(contributions, preactivations, cosines, norms, l)
            # for layer in classes:
            #     dirs, sample_weights = classes[layer], weights[layer]
            #     dirs = {CLASS_NAMES[y]:d for y,d in dirs.items()}
            #     sample_weights = {CLASS_NAMES[y]:d for y,d in sample_weights.items()}
            #     plot = ridge_plot(dirs, sample_weights=sample_weights)
            #     plot.write_html(f"{layer}.html")
            # print(list([c.shape for c in contributions]))
            # print(list([c.shape for c in preactivations]))
            # print(list([c.shape for c in cosines]))
            # print(l.shape)
            # for c in cosines:
            #     print(torch.histc(c,21,c.min(),c.max()))
            # asdfs
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            # print(torch.nn.functional.one_hot(labels, num_classes=10))
            # l = loss(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())
            l = loss(outputs, labels)[:-ghost_samples].mean()

            _, predicted = torch.max(outputs.data, 1)
            total += labels[:-ghost_samples].size(0)
            correct += (predicted == labels)[:-ghost_samples].sum().item()
            pbar.set_description(
                f"Loss: {l.item():.4f} Accuracy: {1.0 * correct / total:.4f}"
            )
            l.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()

            # ind:
            optimizer.archive(ids=ind, labels=labels)

        torch.save(net.state_dict(), os.environ["MODEL_FILE"])
        # torch.save(net.state_dict(), "network.pt")
        # torch.save(optimizer, "optimizer.pt")
        print("Training loss is: " + str(l.item()))
        train_acc = correct / total
        print("Training accuracy is: " + str(train_acc))

        correct = 0
        total = 0
        net.eval()  # eval mode
        with torch.no_grad():
            for data in testloader:
                inputs, labels, ind = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_acc = correct / total
        print("Test accuracy is: " + str(test_acc))
        scheduler.step()
        net = net.to(device)
    torch.save(net.state_dict(), os.environ["MODEL_FILE"])
    # torch.save(net.cpu().state_dict(), "cifar_log.pt")
    print("done with everything")


if __name__ == "__main__":
    main()
