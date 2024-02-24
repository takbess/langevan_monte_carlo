# ライブラリのインポート
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import MixtureSameFamily, Categorical
from tqdm import tqdm
dim = 2

# ランジュバン・モンテカルロ法の実装
def langevin_monte_carlo(dist, num_samples, num_steps, step_size):
    # 初期サンプルを乱数から生成
    x = torch.randn(num_samples, dim)
    for i in tqdm(range(num_steps)):
        x.requires_grad_()
        log_p = dist.log_prob(x)
        score = torch.autograd.grad(log_p.sum(), x)[0]
        with torch.no_grad():
            noise = torch.randn(num_samples, dim)
            x = x + step_size * score + np.sqrt(2 * step_size) * noise
    return x

# 各サンプルに対して、全軌跡を返すタイプを作ってみた。
def langevin_monte_carlo_trajectory(dist, num_samples, num_steps, step_size):
    # 初期サンプルを乱数から生成
    x = torch.randn(num_samples, dim) # ~N(0,1)のはず。今の場合はずるい気がする、、。
    x_all = []
    for i in tqdm(range(num_steps)):
        x.requires_grad_()
        log_p = dist.log_prob(x)
        score = torch.autograd.grad(log_p.sum(), x)[0]
        with torch.no_grad():
            noise = torch.randn(num_samples, dim) # ~N(0,1)
            x = x + step_size * score + np.sqrt(2 * step_size) * noise
        x_all.append(x)

    x_all = torch.cat(x_all)
    return x_all.detach()