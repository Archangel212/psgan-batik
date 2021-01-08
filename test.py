import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import re
import math

# experiments_name = "/Users/mac/Deep_Learning/SinGAN/psgan-pytorch/log/Batik_500by500_homogenous_736-900/kernel=4,generator_leakyRelu=0.2,instance_noise_mean=0_std=0.1,label_smoothing=0.0955percent,zl_dim=60,zg_dim=60,learning_rate_g=1e-4,learning_rate_d=4e-4,d_dropout_lastlayer=0.2,bce_with_logits/train_log_20201217_21-20-31.csv"
# # experiments_name = "/Users/mac/Deep_Learning/SinGAN/psgan-pytorch/log/Batik_500by500_homogenous_736-900/kernel=4,generator_leakyRelu=0.2,instance_noise_mean=0_std=0.1,label_smoothing=0.0955percent,zl_dim=60,zg_dim=60,learning_rate_g=1e-4,learning_rate_d=4e-4,d_dropout_lastlayer=0.2/train_log_20201215_20-57-24.csv"
# df_csv = pd.read_csv(experiments_name, index_col=None)

# print(df_csv["total_loss"].mean(), df_csv["discriminator_loss"].mean(), df_csv["generator_loss"].mean())

# std = 0.1
# ln = 5
# std_decay = std / (ln -1 )

# for i in range(ln):
#   print(std,std_decay)
#   std = std - std_decay

# os.chdir("/Users/mac/Deep_Learning/SinGAN/psgan-pytorch/log")

# pths = [pth for pth in Path("/Users/mac/Deep_Learning/SinGAN/psgan-pytorch/log").rglob("*.pth")]

# new_pths = []
# for pth in pths:
#   if not (pth.match("generator_param_fin_10000.pth") or pth.match("generator_param_fin_20000.pth")):
#     print(pth)
#     new_pths.append(pth)
#     # os.remove(pth)

# input = [(math.log(i/10)) for i in range(1,11)]
# input = [(math.log(1- (i/10))) for i in range(1,10)]
# print(input)
# print(math.log(2))

log_dir = "log"
os.chdir(log_dir)
batik_ds = [l for l in os.listdir('.')]

for b in batik_ds:
    if os.path.isdir(b):
        os.chdir(b)
        for model_hyperparameter in os.listdir(os.curdir):
            print(model_hyperparameter)
    os.chdir("../"+os.curdir)
    # if os.path.isdir(b):
    #     for exp_name in os.listdir(b):
    #         print(exp_name)
    