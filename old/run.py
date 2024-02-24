# ライブラリのインポート
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import MixtureSameFamily, Categorical

from tools.distributions import make_normal_dist,make_mixture_dist
from tools.plot import plot_dist,plot_samples
from tools.mcmc import langevin_monte_carlo,langevin_monte_carlo_trajectory

# fig folder作成
os.makedirs("fig",exist_ok=True)
# 乱数シードを固定
# torch.manual_seed(1234)

# 一番簡単な実験
if True:
    # 分布の用意
    dist = make_normal_dist()
    ranges=((-2, 2), (-2, 2))
    plot_dist(dist,"fig/normal_gt.png",ranges)

    # ランジュバン・モンテカルロ法のパラメータ
    num_samples = 100000
    num_steps = 1000
    step_size = 0.001

    # サンプリングの実行
    samples = langevin_monte_carlo(dist, num_samples, num_steps, step_size)
    plot_samples(samples,"fig/normal_lmc.png",ranges)


# 混合ガウス分布の実験
if True:
    # 分布の用意
    dist = make_mixture_dist()
    ranges=((-5, 5), (-5, 5))
    plot_dist(dist,"fig/mixture_gt.png",ranges)

    # ランジュバン・モンテカルロ法のパラメータ
    num_samples = 100000
    num_steps = 1000
    step_size = 0.001

    # サンプリングの実行
    samples = langevin_monte_carlo(dist, num_samples, num_steps, step_size)
    plot_samples(samples,"fig/mixture_lmc.png",ranges)
