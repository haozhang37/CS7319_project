import torch
import torch.nn.functional as F
import torchvision.transforms as T
import argparse
import numpy as np
import torchvision
from models.simple_lmser import SimpleLmser
from matplotlib import pyplot as plt
import os

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import offsetbox
from sklearn import (manifold, decomposition, discriminant_analysis)
import cv2
import sklearn.cluster
import sklearn.metrics

import PIL


def tensor2np(t):
    """
    tensor(n,....) to np(n,-1)
    """
    return t.detach().numpy().reshape(t.shape[0], -1)


def count_label(y):
    """
    计算总的label个数
    """
    labels = []
    for label in y:
        if label not in labels:
            labels.append(label)
    return len(labels)


def cluster_score(X, y):
    """获取聚类结果

    Kmeans with ARI+AMI

    Adjusted Rand index score 调整兰德系数
    Adjusted Mutual Information score 调整互信息
    两者值越大相似度越高聚类效果越好
    https://blog.csdn.net/u010159842/article/details/78624135
    """

    n_label = count_label(y)
    myKmeans = sklearn.cluster.KMeans(n_clusters=n_label)
    x_cluster = myKmeans.fit_predict(X)

    ARI = sklearn.metrics.adjusted_rand_score(y, x_cluster)
    AMI = sklearn.metrics.adjusted_mutual_info_score(
        y, x_cluster, average_method='arithmetic')

    return ARI, AMI


def not_outliers(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score <= thresh


def plot_embedding_3D(X, y=None, title=None,savepath=None,sub="111",fig=None):
    """
    # Scale and visualize the embedding vectors

    # X is the projection location in 3D

    # y is index of labels #标签的下标

    """
    assert X.shape[1] == 3, "error! require 3D points"
    assert len(X) == len(y), "error! require X has same length to y"

    n_label = count_label(y)

    if fig is not None:
        ax = fig.add_subplot(sub, projection='3d')
    else:
        ax = plt.subplot(sub, projection='3d')

    for i in range(X.shape[0]):
        ax.scatter(X[i, 0], X[i, 1], X[i, 2],
                   color=plt.cm.jet(y[i] / n_label),alpha = 0.5)

    if title is not None:
        plt.title(title)
    if savepath is not None:
        plt.tight_layout()
        plt.savefig(savepath+"-3D.png", dpi=200)


def plot_embedding_2D(X, y=None, title=None, savepath=None, sub="111", fig=None):
    """
    # Scale and visualize the embedding vectors

    # X is the projection location in 2D

    # y is index of labels #标签的下标

    """
    assert X.shape[1] == 2, "error! require 2D points"
    assert len(X) == len(y), "error! require X has same length to y"

    n_label = count_label(y)

    # 归一化画布坐标为scale坐标即比例坐标
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) * 1.0 / (x_max - x_min)

    if fig is not None:
        ax = fig.add_subplot(sub)
    else:
        ax = plt.subplot(sub)

    # 标出数据的分布
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], str(y[i]),
                color=plt.cm.jet(y[i] / n_label),
                fontdict={'weight': 'bold', 'size': 7},
                alpha=0.5)

    if title is not None:
        plt.title(title)
    if savepath is not None:
        plt.tight_layout()
        plt.savefig(savepath + "-2D.png", dpi=200)


def draw_TSNE(args, model, testloader):
    # 超参数
    # hyper parameters
    # SAVEDIR = "vis13/"
    path = args.save_path + f"visualization/layer-{args.layer_num}_reflect-{args.reflect_num}_channel-{args.channel}_class-{args.class_num}_lr-{args.lr}_epoch-{args.epoch - 1}/"

    all_score = []
    all_features = []

    def get_feature(module, input, output):
        all_features.append(output)
    for i in range(model.layer_num):
        model.conv[i].register_forward_hook(get_feature)
    # 提取部分数据集
    for batch, (X_input, y) in enumerate(testloader):
        batch_score = []

        y = y.detach().numpy()

        # 预测
        if "cnn" not in args.save_path:
            X_input = X_input.view(X_input.size(0), -1)
        X_input = X_input.to(args.device)
        all_features = []
        Xs = model(X_input)
        if isinstance(Xs, tuple):
            Xs = Xs[-1]

        # 分类正确率
        # X_res = Xs.detach().cpu().numpy()
        # print("acc:", np.mean(np.argmax(X_res, axis=1) == y))

        # 分层画出结果
        for layer, X in enumerate(all_features):
            print("===========layer:{}============".format(layer))
            X = tensor2np(X.cpu())
            # print(X.shape)

            # 我们有了每层的数据 X 和 Y
            layer = "output"
            vmodel = manifold.TSNE(n_components=3)
            X_ = vmodel.fit_transform(X)
            ARI, AMI = cluster_score(X_, y)
            score = [ARI, AMI, vmodel.kl_divergence_]
            batch_score.append(score)
            title = "TSNE-{}".format(layer)
            print(title, "ARI, AMI , vars :{}".format(score))

            if batch == 0:
                y_ = y[not_outliers(X_)]
                X_draw = X_[not_outliers(X_)]

                fig = plt.figure()
                plot_embedding_3D(X_draw, y_, savepath=path + title, fig=fig, sub="111")
                fig = plt.figure()
                plot_embedding_2D(X_draw[:, :2], y_, savepath=path + title, fig=fig, sub="111")

        all_score.append(batch_score)
        if batch >= 2:
            break
    print("==============final================")
    print("ARI, AMI, kl \n", np.array(all_score))
    print("ARI, AMI, kl means\n", np.mean(all_score, axis=0))


def visualization(args, model, testloader):
    path = args.save_path + f"visualization/layer-{args.layer_num}_reflect-{args.reflect_num}_channel-{args.channel}_class-{args.class_num}_lr-{args.lr}_epoch-{args.epoch - 1}/"
    if not os.path.exists(path):
        os.makedirs(path)

    with torch.no_grad():
        for i, data in enumerate(testloader):
            print(f"Visualizing {i}-th image")
            img, _ = data
            img = img.to(args.device)
            size = img.size()
            bs = img.size(0)
            if "cnn" not in args.save_path:
                img = img.view(bs, -1)
            y = model(img)
            if isinstance(y, tuple):
                y = y[0]
            norm_img = img[0].cpu().view(size)
            norm_img = (norm_img - torch.min(norm_img)) / (torch.max(norm_img) - torch.min(norm_img))
            norm_y = y[0].cpu().view(size)
            norm_y = (norm_y - torch.min(norm_y)) / (torch.max(norm_y) - torch.min(norm_y))
            torchvision.utils.save_image(norm_y, path + f"reconstruct_img-{i}.jpg", normalize=True)
            torchvision.utils.save_image(norm_img, path + f"original_img-{i}.jpg", normalize=True)

            if i > args.visualize_num:
                break


def interpolate_forward(model, x, m=10):

    def inter(output):
        ori_output = output
        new_output = []
        for i in range(m + 1):
            new_output.append((ori_output[0] * float(m - i) / m + ori_output[1] * float(i) / m).unsqueeze(0))
        new_output = torch.cat(new_output[:], 0).to(output.device)
        return new_output

    for i in range(model.reflect):
        short_cut = []
        for j in range(model.block_num):
            for k in range(model.layer_num):
                l = j * model.layer_num + k
                if k != model.layer_num - 1 and j != model.block_num - 1:
                    x = F.sigmoid(model.conv[l](x))
                else:
                    x = model.conv[l](x)
                short_cut.append(inter(x))
        y = x
        x = inter(x)
        for j in range(model.block_num):
            for k in range(model.layer_num):
                l = model.layer_num * model.block_num - j * model.layer_num - k - 1
                if j != model.block_num - 1 and k != model.layer_num:
                    x = F.sigmoid(model.dec_conv[l](x + short_cut[l]))
                else:
                    x = model.dec_conv[l](x + short_cut[l])
    return x, y.view(y.size(0), -1)


def interpolate(args, model, testset, k=10):
    testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False)
    path = args.save_path + f"visualization/layer-{args.layer_num}_reflect-{args.reflect_num}_channel-{args.channel}_class-{args.class_num}_lr-{args.lr}_epoch-{args.epoch - 1}/"
    if not os.path.exists(path):
        os.makedirs(path)

    # def get_feature(module, input, output):
    #     ori_output = output
    #     new_output = []
    #     for i in range(k + 1):
    #         new_output.append((ori_output[0] * float(k - i) / k + ori_output[1] * float(i) / k).unsqueeze(0))
    #     new_output = torch.cat(new_output[:], 0).to(output.device)
    #     return new_output
    # for layer in model.conv:
    #     layer.register_forward_hook(get_feature)

    with torch.no_grad():
        for i, data in enumerate(testloader):
            print(f"Visualizing {2 * i}-th and {2 * i + 1}-th image")
            img, _ = data
            img = img.to(args.device)
            size = img.size()
            bs = img.size(0)
            if "cnn" not in args.save_path:
                img = img.view(bs, -1)
            y = interpolate_forward(model, img)
            if isinstance(y, tuple):
                y = y[0]
            norm_img = img.cpu()
            norm_img = (norm_img - torch.min(norm_img)) / (torch.max(norm_img) - torch.min(norm_img))
            norm_y = y.cpu()
            norm_y = (norm_y - torch.min(norm_y)) / (torch.max(norm_y) - torch.min(norm_y))
            # torchvision.utils.save_image(norm_y[0], path + f"reconstruct_img-{2 * i}.jpg", normalize=True)
            # torchvision.utils.save_image(norm_y[1], path + f"reconstruct_img-{2 * i + 1}.jpg", normalize=True)
            torchvision.utils.save_image(norm_y, path + f"inter_reconstruct_img-{2 * i}-{2 * i + 1}.jpg", normalize=True, nrow=12)
            torchvision.utils.save_image(norm_img[0], path + f"inter_original_img-{2 * i}.jpg", normalize=True)
            torchvision.utils.save_image(norm_img[1], path + f"inter_original_img-{2 * i + 1}.jpg", normalize=True)

            if i > args.visualize_num:
                break


def main(args):
    if args.dataset == "MNIST":
        mean, std = (0.1307,), (0.3081,)
        train_trans = T.Compose((T.RandomHorizontalFlip(0.5), T.ToTensor(), T.Normalize(mean=mean, std=std)))
        test_trans = T.Compose((T.ToTensor(), T.Normalize(mean=mean, std=std)))
        trainset = torchvision.datasets.MNIST(root="./data/MNIST/", train=True, download=True, transform=train_trans)
        testset = torchvision.datasets.MNIST(root="./data/MNIST/", train=False, download=True, transform=test_trans)
    elif args.dataset == "F-MNIST":
        mean, std = (0.1307,), (0.3081,)
        train_trans = T.Compose((T.RandomHorizontalFlip(0.5), T.ToTensor(), T.Normalize(mean=mean, std=std)))
        test_trans = T.Compose((T.ToTensor(), T.Normalize(mean=mean, std=std)))
        trainset = torchvision.datasets.FashionMNIST(root="./data/F-MNIST/", train=True, download=True,
                                                     transform=train_trans)
        testset = torchvision.datasets.FashionMNIST(root="./data/F-MNIST/", train=False, download=True,
                                                    transform=test_trans)
    elif args.dataset == "STL10":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        train_trans = T.Compose((T.RandomHorizontalFlip(0.5), T.ToTensor(), T.Normalize(mean=mean, std=std)))
        test_trans = T.Compose((T.ToTensor(), T.Normalize(mean=mean, std=std)))
        trainset = torchvision.datasets.STL10(root="./data/STL10/", split="train", download=True,
                                              transform=train_trans)
        testset = torchvision.datasets.STL10(root="./data/STL10/", split="test", download=True,
                                             transform=test_trans)
    else:
        raise RuntimeError("Invalid dataset name!")
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False)
    # model = torch.load(args.save_path + f"model_layer-{args.layer_num}_reflect-{args.reflect_num}_channel-{args.channel}_class-{args.class_num}_lr-{args.lr}_epoch-{args.epoch - 1}.pkl", map_location=lambda storage, loc: storage.cuda(args.device))
    model = torch.load(args.save_path + f"model_layer-{args.layer_num}_reflect-{args.reflect_num}_channel-{args.channel}_lr-{args.lr}_epoch-{args.epoch - 1}.pkl", map_location=lambda storage, loc: storage.cuda(args.device))
    model.to(args.device)
    # visualization(args, model, test_loader)
    # draw_TSNE(args, model, test_loader)
    interpolate(args, model, testset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker.')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--save_path", type=str, default="./result/cnnlmser_classifier/")
    parser.add_argument("--visualize_num", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="STL10", choices=["MNIST", "F-MNIST", "STL10"])

    parser.add_argument("--class_num", type=int, default=10)
    parser.add_argument("--layer_num", type=int, default=2)
    parser.add_argument("--reflect_num", type=int, default=1)
    parser.add_argument("--channel", type=int, default=128)
    args = parser.parse_args()
    if args.device != "cpu":
        args.device = int(args.device)
    args.save_path = args.save_path + f"{args.dataset}/"
    main(args)
