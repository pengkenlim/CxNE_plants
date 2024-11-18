#colors
#tab:blue : #1f77b4
#tab:orange : #ff7f0e
#tab:green : #2ca02c
#tab:red : #d62728
#tab:purple : #9467bd
#tab:brown : #8c564b
#tab:pink : #e377c2
#tab:gray : #7f7f7f
#tab:olive : #bcbd22
#tab:cyan : #17becf
# %%
import os
import sys
import matplotlib.pyplot as plt


# %%



def parse_training_logs(path):
    A_Train = []
    A_Val = []
    A_Test= []
    A_epoch= []

    I_Train = []
    I_Val = []
    I_Test= []
    I_epoch= []
    A_LR=[]
    with open(path, "r") as fin:
        contents = fin.read()
        contents = contents.split("Epoch\tBatch\tMode\ttrain_loss\tval_loss\ttest_loss\tlearn_rate\n")[-1]
    contents = contents.split("\n")
    for line in contents:
        if "Agg" in line:
            line_contents = line.split("\t")
            A_epoch.append(int(line_contents[0]))
            A_Train.append(float(line_contents[-4]))
            A_Val.append(float(line_contents[-3]))
            A_Test.append(float(line_contents[-2]))
            A_LR.append(float(line_contents[-1]))
        elif "Infer" in line:
            line_contents = line.split("\t")
            I_epoch.append(int(line_contents[0]))
            I_Train.append(float(line_contents[-4]))
            I_Val.append(float(line_contents[-3]))
            I_Test.append(float(line_contents[-2]))
    return A_Train, A_Val, A_Test, A_epoch, I_Train, I_Val, I_Test, I_epoch, A_LR
# %%
model_dir = "/mnt/md2/ken/CxNE_plants_data"
model_name = "model_28_relu_loss_05psubgraph_20density_2kepochs_half_agg_GAT32"
training_log_path = os.path.join(model_dir, model_name, "training_logs.txt")

A_Train, A_Val, A_Test, A_epoch, I_Train, I_Val, I_Test, I_epoch, A_LR = parse_training_logs(training_log_path)
LR_change_epoch = []
LR_change = []
LR_prev= 1
for epoch , LR in zip(A_epoch, A_LR):
    if LR < LR_prev:
        LR_prev = LR
        LR_change_epoch.append(epoch)
        LR_change.append(LR)

# %%
import numpy as np
ylim_max = 0.35
ylim_min = 0.15
plt.figure(figsize=(10, 5))
plt.plot(I_epoch,I_Train, c = "#1f77b4")
plt.plot(I_epoch,I_Val, c = "#ff7f0e")
plt.plot(I_epoch,I_Test, c = "#e377c2")
scale = 0.9
for epoch , LR in zip(LR_change_epoch, LR_change):
    plt.plot([epoch,epoch],[ylim_min, ylim_max], "--", c = "#7f7f7f")
    plt.text(epoch+5, ylim_min + (scale*(ylim_max - ylim_min)), f"10^{np.round(np.log10(LR), 3)}", fontsize=8, color='#7f7f7f')
    scale -= 0.1
plt.ylim(ylim_min,ylim_max)
plt.xlim(0,2000)
plt.ylabel("RMSE Loss")
plt.xlabel("No. Epochs")

plt.savefig(os.path.join(model_dir, model_name + "_All_Big.png"))
#plt.savefig(os.path.join(model_dir, model_name + "_train_Big.png"))
#plt.savefig(os.path.join(model_dir, model_name + "_val_Big.png"))

# %%
ylim_max = 0.21
ylim_min = 0.204
plt.figure(figsize=(3, 3))
plt.plot(I_epoch,I_Train, c = "#1f77b4")
plt.plot(I_epoch,I_Val, c = "#ff7f0e")
#plt.plot(I_epoch,I_Test, c = "#e377c2")
plt.ylim(ylim_min,ylim_max)
plt.xlim(1700,2050)
plt.ylabel("RMSE Loss")
plt.xlabel("No. Epochs")

#plt.savefig(os.path.join(model_dir, model_name + "_All_zoomed.png"))
#plt.savefig(os.path.join(model_dir, model_name + "_train_zoomed.png"))
plt.savefig(os.path.join(model_dir, model_name + "_val_zoomed.png"))

# %%
if False:
    plt.clf()
    plt.figure(figsize=(5, 5))
    plt.plot(A_epoch,A_Train, c = "#1f77b4")
    plt.ylim(0.26,0.4)
    plt.scatter(x = I_epoch, y= I_Train, c = "#1f77b4", marker = "+")
    plt.savefig(os.path.join(path, model_name + "_train_.png"))

    plt.clf()
    plt.figure(figsize=(5, 5))
    plt.plot(A_epoch,A_Train, c = "#1f77b4")
    plt.plot(A_epoch,A_Val, c = "#ff7f0e")
    plt.ylim(0.26,0.4)
    plt.scatter(x = I_epoch, y= I_Train, c = "#1f77b4", marker = "+")
    plt.scatter(x = I_epoch, y= I_Val, c = "#ff7f0e", marker = "+")
    plt.savefig(os.path.join(path, model_name + "_val_.png"))