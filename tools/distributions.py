# ライブラリのインポート
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import MixtureSameFamily, Categorical

# 2次元の分布のみ
def make_normal_dist():
    dim=2
    dist = MultivariateNormal(torch.zeros((dim)), torch.eye(dim))
    return dist

def make_mixture_dist():
    # 平均ベクトル
    means = torch.tensor([[0.0, 0.0], [2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]])

    # 共分散行列
    covs = torch.Tensor([
        [[ 1.0,  0.0],
        [ 0.0,  1.0]],

        [[ 0.6,  0.1],
        [ 0.1,  0.9]],

        [[ 0.8, -0.2],
        [-0.2,  0.8]],
        
        [[ 0.3, 0.2],
        [0.2,  0.8]],
    ])

    # 混合係数
    mixture_weights = torch.tensor([0.2, 0.2, 0.4, 0.2])

    # 混合正規分布を作成
    mixture_dist = MixtureSameFamily(
        Categorical(mixture_weights),
        MultivariateNormal(means, covs)
    )

    return mixture_dist