import torch
import torch.nn.functional as F
import torchvision.transforms as T
import argparse
import numpy as np
import torchvision
from models.lmser_classifier import SimpleLmser_Classifier
from matplotlib import pyplot as plt
import os
import random
import math


def set_seed_pytorch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args, model, trainloader, optimizer):
    MSE_Loss = 0
    CE_Loss = 0
    Accuracy = 0
    for i, data in enumerate(trainloader):
        img, lbl = data
        img = img.to(args.device)
        lbl = lbl.to(args.device)
        bs = img.size(0)
        img = img.view(bs, -1)
        y, pre = model(img)
        loss = F.mse_loss(img, y)
        cls_loss = F.cross_entropy(pre, lbl)
        total_loss = loss + cls_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        model.set_DCW()
        MSE_Loss += loss.detach().cpu().item()
        CE_Loss += cls_loss.detach().cpu().item()
        Accuracy += (pre.argmax(dim=1).detach().cpu() == lbl.cpu()).sum().item() / bs
        print(f"{i} train mse loss:{MSE_Loss / (i + 1)}, cls loss:{CE_Loss / (i + 1)}, accuracy:{Accuracy / (i + 1)}")
    return MSE_Loss / (i + 1), CE_Loss / (i + 1), Accuracy / (i + 1)


def test(args, model, testloader):
    MSE_Loss = 0
    CE_Loss = 0
    Accuracy = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            img, lbl = data
            img = img.to(args.device)
            lbl = lbl.to(args.device)
            bs = img.size(0)
            img = img.view(bs, -1)
            y, pre = model(img)
            loss = F.mse_loss(img, y)
            cls_loss = F.cross_entropy(pre, lbl)
            MSE_Loss += loss.cpu().item()
            CE_Loss += cls_loss.cpu().item()
            Accuracy += (pre.argmax(dim=1).detach().cpu() == lbl.cpu()).sum().item() / bs
            print(f"{i} test loss:{MSE_Loss / (i + 1)}, cls loss:{CE_Loss / (i + 1)}, accuracy:{Accuracy / (i + 1)}")
    return MSE_Loss / (i + 1), CE_Loss / (i + 1), Accuracy / (i + 1)


def draw(args, train_mse_list, train_ce_list, train_acc_list, test_mse_list, test_ce_list, test_acc_list, path):
    x = np.arange(len(train_mse_list))
    plt.figure()
    plt.subplot(131)
    plt.plot(x, np.array(train_mse_list), color="blue")
    plt.plot(x, np.array(test_mse_list), color="red")
    plt.ylabel("Reconstruction loss")

    plt.subplot(132)
    plt.plot(x, np.array(train_ce_list), color="blue")
    plt.plot(x, np.array(test_ce_list), color="red")
    plt.ylabel("Classification loss")

    plt.subplot(133)
    plt.plot(x, np.array(train_acc_list), color="blue")
    plt.plot(x, np.array(test_acc_list), color="red")
    plt.ylabel("Classification accuracy")
    plt.savefig(f"{path}curves_layer-{args.layer_num}_reflect-{args.reflect_num}_channel-{args.channel}_lr-{args.lr}.png")
    plt.close()


def main(args):
    mean, std = (0.1307,), (0.3081,)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    train_trans = T.Compose((T.RandomHorizontalFlip(0.5), T.ToTensor(), T.Normalize(mean=mean, std=std)))
    test_trans = T.Compose((T.ToTensor(), T.Normalize(mean=mean, std=std)))
    set_seed_pytorch(args.epoch + args.layer_num * 100)
    model = SimpleLmser_Classifier(class_num=args.class_num, layer_num=args.layer_num, reflect_num=args.reflect_num, channel=args.channel)
    if args.dataset == "MNIST":
        trainset = torchvision.datasets.MNIST(root="./data/MNIST/", train=True, download=True, transform=train_trans)
        testset = torchvision.datasets.MNIST(root="./data/MNIST/", train=False, download=True, transform=test_trans)
    elif args.dataset == "F-MNIST":
        trainset = torchvision.datasets.FashionMNIST(root="./data/F-MNIST/", train=True, download=True, transform=train_trans)
        testset = torchvision.datasets.FashionMNIST(root="./data/F-MNIST/", train=False, download=True, transform=test_trans)
    else:
        raise RuntimeError("Invalid dataset name!")
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False)
    # params = []
    model.to(args.device)
    # for i in range(args.layer_num):
    #     model.fc[i].to(args.device)
    #     model.dec_fc[i].to(args.device)
    #     params.append({"params": model.fc[i].parameters()})
    #     params.append({"params": model.dec_fc[i].parameters()})

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_mse_list = []
    train_ce_list = []
    train_acc_list = []
    test_mse_list = []
    test_ce_list = []
    test_acc_list = []

    log_lr_st = math.log10(args.lr)
    lr_epoch = torch.logspace(log_lr_st, log_lr_st - 1, steps=args.epoch)

    for epoch in range(args.epoch):
        set_seed_pytorch(args.epoch + args.layer_num * 10 + epoch * 100 + args.reflect_num * 1000)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_epoch[epoch]
        train_mse_loss, train_ce_loss, train_acc = train(args, model, train_loader, optimizer)
        model.eval()
        set_seed_pytorch(args.epoch + args.layer_num * 20 + epoch * 200 + args.reflect_num * 2000)
        test_mse_loss, test_ce_loss, test_acc = test(args, model, test_loader)
        model.train()
        train_mse_list.append(train_mse_loss)
        train_ce_list.append(train_ce_loss)
        train_acc_list.append(train_acc)
        test_mse_list.append(test_mse_loss)
        test_ce_list.append(test_ce_loss)
        test_acc_list.append(test_acc)
        draw(args, train_mse_list, train_ce_list, train_acc_list, test_mse_list, test_ce_list, test_acc_list, args.save_path)
        if epoch % 10 == 0 or epoch == args.epoch - 1:
            torch.save(model, args.save_path + f"model_layer-{args.layer_num}_reflect-{args.reflect_num}_channel-{args.channel}_lr-{args.lr}_epoch-{epoch}.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker.')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--save_path", type=str, default="./result/lmser_classifier/")
    parser.add_argument("--dataset", type=str, default="F-MNIST", choices=["MNIST", "F-MNIST"])

    parser.add_argument("--class_num", type=int, default=10)
    parser.add_argument("--layer_num", type=int, default=3)
    parser.add_argument("--reflect_num", type=int, default=1)
    parser.add_argument("--channel", type=int, default=128)
    args = parser.parse_args()
    if args.device != "cpu":
        args.device = int(args.device)
    args.save_path = args.save_path + f"{args.dataset}/"
    main(args)
