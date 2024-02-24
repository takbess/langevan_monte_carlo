# ライブラリのインポート
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import MixtureSameFamily, Categorical

def plot_dist(dist,filename,ranges):
    ((xmin,xmax),(ymin,ymax)) = ranges
    # 二次元の格子点の座標を作成
    ls_x = np.linspace(xmin, xmax, 1000)
    ls_y = np.linspace(ymin, ymax, 1000)
    x, y = np.meshgrid(ls_x, ls_y)
    point = torch.tensor(np.vstack([x.flatten(), y.flatten()]).T)
    # 格子点の座標における尤度を算出
    p = torch.exp(dist.log_prob(point))

    # 二次元正規分布を可視化
    plt.title('dist')
    plt.pcolormesh(x, y, p.reshape(x.shape), cmap='viridis')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.colorbar()
    # plt.show()
    plt.savefig(filename)
    print("saved:",filename)
    plt.clf()
    return plt

def plot_samples(samples,filename,ranges):
    # サンプリング結果の可視化
    plt.title('langevin monte carlo sampling')
    plt.hist2d(
        samples[:,0], 
        samples[:,1], 
        range=ranges, 
        cmap='viridis', 
        bins=50, 
    )
    plt.gca().set_aspect('equal', adjustable='box')
    ((xmin,xmax),(ymin,ymax)) = ranges
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.colorbar()
    # plt.show()
    plt.savefig(filename)
    print("saved:",filename)
    plt.clf()
    return plt
