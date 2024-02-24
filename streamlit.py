# ライブラリのインポート
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread,imsave
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import MixtureSameFamily, Categorical

from tools.distributions import make_normal_dist,make_mixture_dist
from tools.plot import plot_dist,plot_samples
from tools.mcmc import langevin_monte_carlo,langevin_monte_carlo_trajectory

import streamlit as st
import random
import argparse
from PIL import Image
import math
import natsort
# 名前の省略関数
def shorten_string(string):
    new_string = ""
    splits = string.split(" ")
    for s in splits:
        new_string += s[0]
    
    return new_string
    
def main():
    st.set_page_config(layout='wide')
    save_paths = []
    # fig folder作成
    os.makedirs("fig",exist_ok=True)

    # 乱数シード
    seed = st.sidebar.number_input("seed (-1 means seed is random)",min_value=-1,max_value=255,value=0)
    if seed == -1:
        seed = random.randint(0,255)
    torch.manual_seed(seed)
    st.sidebar.write(f"seed: {seed}")

    # 分布を指定
    dist_names_all = ["normal","mixture"]
    dist_names = st.sidebar.multiselect("dist_name",dist_names_all,default="normal")


    # サンプル数、ステップ数、ステップサイズを選択。
    nums_samples = st.sidebar.multiselect('num_samples:',[10**i for i in range(9)],default=1e5)
    nums_steps = st.sidebar.multiselect('num_steps:',[10**i for i in range(9)],default=1e3)
    steps_size = st.sidebar.multiselect('step_size:',[10**i for i in range(-9,2)],default=1e-3)

    # サンプリング方法の指定
    sampling_methods_all = ["langevin monte carlo", "langevin monte carlo trajectory"]
    sampling_methods = st.sidebar.multiselect("sampling_method",sampling_methods_all,default="langevin monte carlo")

    for dist_name in dist_names:
        if dist_name == "normal":
            dist = make_normal_dist()
            ranges=((-2, 2), (-2, 2))
        elif dist_name == "mixture":
            dist = make_mixture_dist()
            ranges=((-5, 5), (-5, 5))
        else:
            print("error for dist_name")
            print(dist_name)
            sys.exit(0)
        # プロット
        save_path = f"fig/{dist_name}_gt.png"
        if os.path.exists(save_path) == False:
            plot_dist(dist,save_path,ranges)
        save_paths.append(save_path)


        for num_samples in nums_samples:
            for num_steps in nums_steps:
                for step_size in steps_size:
                    for sampling_method in sampling_methods:

                        save_path = f"fig/{dist_name}_{shorten_string(sampling_method)}_{num_samples}_{num_steps}_{step_size}_seed{seed}.png"
                        if os.path.exists(save_path) == False:
                            # サンプリング
                            if sampling_method == "langevin monte carlo":
                                samples = langevin_monte_carlo(dist,num_samples,num_steps,step_size)
                            elif sampling_method == "langevin monte carlo trajectory":
                                samples = langevin_monte_carlo_trajectory(dist,num_samples,num_steps,step_size)

                            # プロット
                            plt = plot_samples(samples,save_path,ranges)
                        save_paths.append(save_path)

    # sort
    save_paths = natsort.natsorted(save_paths)
    # 表示
    num_columns = st.sidebar.selectbox("num_columns:",range(1,5),index=1)
    images = []
    for path in save_paths:
        img = Image.open(path)
        images.append(img)

    cols = st.columns(num_columns)
    for i,image in enumerate(images):
        col = cols[i%num_columns]
        title = save_paths[i].split("/")[-1].split(".png")[0]
        col.write(title)
        col.image(image)

    # なぜかうまくいかない
    # for x in range(num_columns):
    #     st.sidebar.write(num_columns*x)
    #     st.sidebar.write(num_columns*(x+1))
    #     st.sidebar.image(images[num_columns*x:num_columns*(x+1)],use_column_width=True)

    # for i in range(0,len(images),num_columns):
    #     st.sidebar.image(images[i:i+num_columns],use_column_width=True)

    
    



if __name__ == "__main__":
    main()