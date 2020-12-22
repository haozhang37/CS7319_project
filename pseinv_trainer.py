import torch
import torch.nn.functional as F
import torchvision.transforms as T
import argparse
import numpy as np
import torchvision
from models.simple_lmser import Pse_Inv_Lmser
from matplotlib import pyplot as plt
import os
import random
import math
from util import weight_analysis
from torch.utils.data import DataLoader
from samplers import CategoriesSampler


def set_seed_pytorch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args, model, trainloader, optimizer):
    Loss = 0
    for i, data in enumerate(trainloader):
        img, _ = data
        img = img.to(args.device)
        bs = img.size(0)
        img = img.view(bs, -1)
        y = model(img)
        loss = F.mse_loss(img, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.set_DPN()
        Loss += loss.detach().cpu().item()
        # print(f"{i} train loss:{Loss / (i + 1)}")
        if i % 10 == 0:
            print(f"{i} train loss:{Loss / (i + 1)}")
    return Loss / (i + 1)


def test(args, model, testloader):
    Loss = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            img, _ = data
            img = img.to(args.device)
            bs = img.size(0)
            img = img.view(bs, -1)
            y = model(img)
            loss = F.mse_loss(img, y)
            Loss += loss.cpu().item()
            if i % 50 == 0:
                print(f"{i} test loss:{Loss / (i + 1)}")
    return Loss / (i + 1)


def draw(args,train_list, test_list, path):
    x = np.arange(len(train_list))
    plt.figure()
    plt.plot(x, np.array(train_list), color="blue")
    plt.plot(x, np.array(test_list), color="red")
    plt.savefig(f"{path}curves_layer-{args.layer_num}_reflect-{args.reflect_num}_channel-{args.channel}_lr-{args.lr}.png")
    plt.close()


def main(args):
    mean, std = (0.1307,), (0.3081,)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    train_trans = T.Compose((T.RandomHorizontalFlip(0.5), T.ToTensor(), T.Normalize(mean=mean, std=std)))
    test_trans = T.Compose((T.ToTensor(), T.Normalize(mean=mean, std=std)))
    set_seed_pytorch(args.epoch + args.layer_num * 100)
    model = Pse_Inv_Lmser(class_num=args.class_num, layer_num=args.layer_num, reflect_num=args.reflect_num, channel=args.channel)
    if args.dataset == "MNIST":
        trainset = torchvision.datasets.MNIST(root="./data/MNIST/", train=True, download=True, transform=train_trans)
        testset = torchvision.datasets.MNIST(root="./data/MNIST/", train=False, download=True, transform=test_trans)
    elif args.dataset == "F-MNIST":
        trainset = torchvision.datasets.FashionMNIST(root="./data/F-MNIST/", train=True, download=True, transform=train_trans)
        testset = torchvision.datasets.FashionMNIST(root="./data/F-MNIST/", train=False, download=True, transform=test_trans)
    else:
        raise RuntimeError("Invalid dataset name!")

    if args.use_small_samples:
        sampler = CategoriesSampler(trainset.targets, args.num_batch, args.way, args.n_per_batch)
        train_loader = DataLoader(trainset, batch_sampler=sampler, pin_memory=True)
    else:
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
    train_list = []
    test_list = []

    log_lr_st = math.log10(args.lr)
    lr_epoch = torch.logspace(log_lr_st, log_lr_st - 1, steps=args.epoch)

    for epoch in range(args.epoch):
        set_seed_pytorch(args.epoch + args.layer_num * 10 + epoch * 100 + args.reflect_num * 1000)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_epoch[epoch]
        train_loss = train(args, model, train_loader, optimizer)
        model.eval()
        set_seed_pytorch(args.epoch + args.layer_num * 20 + epoch * 200 + args.reflect_num * 2000)
        test_loss = test(args, model, test_loader)
        model.train()
        train_list.append(train_loss)
        test_list.append(test_loss)
        draw(args, train_list, test_list, args.save_path)
        if epoch % 10 == 0 or epoch == args.epoch - 1:
            torch.save(model, args.save_path + f"model_layer-{args.layer_num}_reflect-{args.reflect_num}_channel-{args.channel}_lr-{args.lr}_epoch-{epoch}.pkl")

    print("finished training")
    # weight analysis
    with torch.no_grad():
        print("weight analysis start:")
        for i in range(args.layer_num):
            w1 = model.fc[i].weight.to('cpu').numpy()
            w2 = model.dec_fc[i].weight.to('cpu').numpy()
            print(f"weight analysis: fc[{i}].weight[] and dec_fc[{i}].weight:")
            weight_analysis(w1, w2)

# =====================Lmser without DPN========================
# weight analysis start:
# weight analysis: fc[0].weight and dec_fc[0].weight:
# transpose distance:  10.27124
# inverse distance:  6.3389487
# weight analysis: fc[1].weight and dec_fc[1].weight:
# transpose distance:  2831.982
# inverse distance:  4351.746
# weight analysis: fc[2].weight and dec_fc[2].weight:
# transpose distance:  1.0128667
# inverse distance:  0.35673222

# =====================Lmser with DPN ==========================
# weight analysis start:
# weight analysis: fc[0].weight and dec_fc[0].weight:
# transpose distance:  0.0007783578
# inverse distance:  0.0
# weight analysis: fc[1].weight and dec_fc[1].weight:
# transpose distance:  0.0009829665
# inverse distance:  0.0
# weight analysis: fc[2].weight and dec_fc[2].weight:
# transpose distance:  4.3793516e-06
# inverse distance:  0.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker.')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--save_path", type=str, default="./result/pse_inv_lmser/")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "F-MNIST"])

    parser.add_argument("--class_num", type=int, default=10)
    parser.add_argument("--layer_num", type=int, default=3)
    parser.add_argument("--reflect_num", type=int, default=1)
    parser.add_argument("--channel", type=int, default=128)

    # small sample learning
    parser.add_argument("--use_small_samples", type=bool, default=True)
    parser.add_argument("--num_batch", type=int, default=100)
    parser.add_argument("--n_per_batch", type=int, default=20)
    parser.add_argument("--way", type=int, default=5)

    args = parser.parse_args()
    if args.device != "cpu":
        args.device = int(args.device)
    args.save_path = args.save_path + f"{args.dataset}/"
    main(args)
