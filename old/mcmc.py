# ライブラリのインポート
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import MixtureSameFamily, Categorical

# 乱数シードを固定
# torch.manual_seed(1234)

# 次元を定義
dim = 2
# 平均0、共分散行列が二次元の単位行列の二次元正規分布を作成
dist = MultivariateNormal(torch.zeros((dim)), torch.eye(dim))
# usage: class: loc, covariance_matrix
# usage: tensor([0,0]), tensor([[1,0],[0,1]])

# 二次元の格子点の座標を作成
ls = np.linspace(-2, 2, 100) # (1000,)
x, y = np.meshgrid(ls, ls)
# usage: x1,x2,...,xn: 1d array_like. each len is N1,N2,...,Nn.
# return: X1,X2,...,Xn: (N1,N2,...,Nn) array_like
# total grid number is N1*N2*...*Nn.

point = torch.tensor(np.vstack([x.flatten(), y.flatten()]).T) # (1000000,2)
# 格子点の座標における尤度を算出
p = torch.exp(dist.log_prob(point)) # (N,2) shape tensor

# 二次元正規分布を可視化
plt.title('2-dim normal distribution')
plt.pcolormesh(x, y, p.reshape(x.shape), cmap='viridis')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.colorbar()
plt.show()
plt.savefig("gt.png")


# ランジュバン・モンテカルロ法の実装
def langevin_monte_carlo(dist, num_samples, num_steps, step_size):
    # 初期サンプルを乱数から生成
    x = torch.randn(num_samples, dim) # ~N(0,1)のはず。今の場合はずるい気がする、、。
    for i in range(num_steps):
        x.requires_grad_()
        log_p = dist.log_prob(x)
        score = torch.autograd.grad(log_p.sum(), x)[0]
        with torch.no_grad():
            noise = torch.randn(num_samples, dim) # ~N(0,1)
            x = x + step_size * score + np.sqrt(2 * step_size) * noise
    return x


# ランジュバン・モンテカルロ法のパラメータ
num_samples = 10000
num_steps = 100
step_size = 0.001

# サンプリングの実行
samples = langevin_monte_carlo(dist, num_samples, num_steps, step_size)

# サンプリング結果の可視化
plt.title('langevin monte carlo sampling')
plt.hist2d(
    samples[:,0], 
    samples[:,1], 
    range=((-2, 2), (-2, 2)), 
    cmap='viridis', 
    bins=50, 
)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.colorbar()
plt.show()
plt.savefig("lmc.png")


#### 1 sampleでも途中経過(trajectory)をすべて追ったら分布が再現されるんじゃね？
# ランジュバン・モンテカルロ法の実装
def langevin_monte_carlo(dist, num_samples, num_steps, step_size):
    # 初期サンプルを乱数から生成
    x = torch.randn(num_samples, dim) # ~N(0,1)のはず。今の場合はずるい気がする、、。
    x_all = []
    for i in range(num_steps):
        x.requires_grad_()
        log_p = dist.log_prob(x)
        score = torch.autograd.grad(log_p.sum(), x)[0]
        with torch.no_grad():
            noise = torch.randn(num_samples, dim) # ~N(0,1)
            x = x + step_size * score + np.sqrt(2 * step_size) * noise
        x_all.append(x)

    x_all = torch.cat(x_all)
    return x_all.detach()


# ランジュバン・モンテカルロ法のパラメータ
num_samples = 100
num_steps = 10000
step_size = 0.001

# サンプリングの実行
samples = langevin_monte_carlo(dist, num_samples, num_steps, step_size)

# サンプリング結果の可視化
plt.title('langevin monte carlo sampling')
plt.hist2d(
    samples[:,0], 
    samples[:,1], 
    range=((-2, 2), (-2, 2)), 
    cmap='viridis', 
    bins=50, 
)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.colorbar()
plt.show()
plt.savefig("lmc_trajectory.png")

