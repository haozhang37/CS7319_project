import torch
import torch.nn.functional as F
import torchvision.transforms as T
import argparse
import numpy as np
import torchvision
from models.simple_lmser import SimpleLmser
from matplotlib import pyplot as plt
import os


def visualization(args, model, testloader):
    path = args.save_path + f"visualization/layer-{args.layer_num}_reflect-{args.reflect_num}_channel-{args.channel}_lr-{args.lr}_epoch-{args.epoch - 1}/"
    if not os.path.exists(path):
        os.makedirs(path)

    with torch.no_grad():
        for i, data in enumerate(testloader):
            print(f"Visualizing {i}-th image")
            img, _ = data
            img = img.to(args.device)
            size = img.size()
            bs = img.size(0)
            img = img.view(bs, -1)
            y = model(img)
            torchvision.utils.save_image(y[0].cpu().view(size), path + f"reconstruct_img-{i}.jpg", normalize=True)
            torchvision.utils.save_image(img[0].cpu().view(size), path + f"original_img-{i}.jpg", normalize=True)

            if i > args.visualize_num:
                break


def main(args):
    mean, std = (0.1307,), (0.3081,)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    test_trans = T.Compose((T.ToTensor(), T.Normalize(mean=mean, std=std)))
    testset = torchvision.datasets.MNIST(root="./data/MNIST/", train=False, download=True, transform=test_trans)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False)
    model = torch.load(args.save_path + f"model_layer-{args.layer_num}_reflect-{args.reflect_num}_channel-{args.channel}_lr-{args.lr}_epoch-{args.epoch - 1}.pkl", map_location=lambda storage, loc: storage.cuda(args.device))

    visualization(args, model, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker.')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--save_path", type=str, default="./result/simple_lmser/")
    parser.add_argument("--visualize_num", type=int, default=10)

    parser.add_argument("--class_num", type=int, default=10)
    parser.add_argument("--layer_num", type=int, default=3)
    parser.add_argument("--reflect_num", type=int, default=2)
    parser.add_argument("--channel", type=int, default=128)
    args = parser.parse_args()
    if args.device != "cpu":
        args.device = int(args.device)
    main(args)
