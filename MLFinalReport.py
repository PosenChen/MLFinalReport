#%% 更新說明
"""
Created on Wed May 29 22:22:47 2024
@author: Posen Chen
跑程式的電腦設備： MacBook Air M2
Ver 0.1 從老師提供的程式碼裡剪貼出必要的部分 使用CIFAR-10資料集 調整參數讓程式可順利運行
Ver 0.2 增加簡單計算訓練時間的code 以及刪除修改一些註解
Ver 0.3 超參數置頂 結果可輸出至CSV 但表格內容還沒設計好
Ver 0.4 大改版 改用使用者手動輸入參數來增加測試彈性 卷基層數目參數設定說明 增加參數量的計算 表格內容更新 #測試傳說中的LaNet-5{'model_loss': 1.3637713193893433, 'model_acc': 50.78873801916933}訓練時間： 31.333121061325073
Ver 0.5 增加Confusion Matrix ＆ 輸出CM圖     修正 output chanel 設定10以外的錯誤
Ver 0.6 修正C層stride 不為1時的問題

what in future?
將收集到的時間成本資料y 超參數x... 拿去跑迴歸
觀察不同CNN架構  的時間成本資料 與正確率資料  看看什麼樣的架構的效率較佳  或者前期較佳 後期較佳 與要辨識的目標有什麼關係 能感覺出可能有什麼樣的關係
殘差分析XD 觀察錯誤辨識的情形 是否常出現在某些類別 Q:若只在少數類別間辨識較差ex:小白鷺跟別的白鷺絲 可否能設計加強辨識難點的訓練資料 或者是在目前已訓練完的模型的基礎上再補訓練要加強的部分
結果繪圖
迴圈 想要一次跑多組 超難寫 哭哭
測試其他參數

期末報告及評分方式
以Pytorch建立神經網路模型，在以下超參數進行測試(上課時讓各組各選三個,去嚐試各3種情況)
(2)卷積層的數目  
(3)卷積層filter的數目.   以調整kernel_size, stride, padding 的方式來調整 
(4)pooling的size  直接設定
"""
#%% 載入套件&超參數設定區  超參數僅需設定 CRPF 以大寫英文字母CRPF字串方式設定 並修改第159 160行程式碼後 就可以執行全部的程式碼 再自行手動input參數
import time # 量測量時間成本用
from datetime import datetime # 圖檔命名用 以避免覆蓋   pip install datetime
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt # 若熱力圖僅第一row 有數字的話 請將 matplotlib 降版至3.7.3  pip install matplotlib==3.7.3
import seaborn as sn  # conda install seaborn
import pandas as pd
import numpy as np
import os

# 卷積層的數目 原則：1.第一層為C 2.F後面只能為F或者放最後 3.第一個F前面為P 4.C後面必接R(因為本次不測試不同的activation function) 5.R後面為C或P
# 在nn.Sequential()中物件的數量 就是CRP的總數 如 CRCRPF就  nn.Sequential( sb[0],sb[1],sb[2],sb[3],sb[4] )  ,  nn.Sequential( sc[0],sc[1] )
CRPF = "CRPF" #, 4["CRPF"], 6["CRCRPF"], 7["CRPCRPF"], 8["CRCRCRPF"], 9["CRPCRCRPF", "CRCRPCRPF"] , 10["CRCRCRCRPF", "CRPCRPCRPF"]
print("請159行程式碼 self.block_1 = nn.Sequential(  )  的括弧中填入",CRPF.count("C") + CRPF.count("R") + CRPF.count("P"),"個物件")
print("請在160行程式碼 self.classifier = nn.Sequential(  )  的括弧中填入",CRPF.count("F")+1,"個物件")
# CKS = 3 # [3, 5, 7] # kernel_size= 也就是卷積層filter 大小
# CS = 1 # [1, 2, 3] # stride= 每次filter移動的步數
# CP = 1 # [0, 1, 2] # padding= 圖片外圍多鋪幾層
# PKS = 4 # [3, 4, 5] # nn.MaxPool2d(kernel_size= ) 也就是pooling的大小
# PS = 4 # [1, 2, 3, 4, 5] # nn.MaxPool2d(預設的stride=kernel_size )  # 不在這次報告的測試範圍 就用4
epochs = 3 # 總共跑幾次epochs # 不在這次報告的測試範圍 就用3
BATCH_SIZE = 32 # 設定batch size  每次forward back抓幾筆資料來訓練  # 不在這次報告的測試範圍 就用32
HU = 10 # 設定每次convolution後output_channels的數量  # 不在這次報告的測試範圍 就用10
LR = 0.1 # 學習率  # 不在這次報告的測試範圍 就用0.1
# 不同的optimization方法
#%% 資料處理 & 設定batch_size
# 使用內建資料集 CIFAR10 (因為較容易帶入範例程式碼)
# 訓練資料
train_data = datasets.CIFAR10(
    root="data", # 資料位於工作資料夾內的data資料夾
    train=True, # 此內建資料集友善的提供幫忙分訓練資料的功能 True：訓練資料
    download=True, # 如果root所指定的data資料夾內無資料的話就下載資料
    transform=ToTensor(), # 將圖片由PIL格式轉成Torch tensors
    target_transform=None # 是否要指定各個標籤的名稱    None：使用原來的標籤名
)
# 測試資料
test_data = datasets.CIFAR10(
    root="data",
    train=False, # 此內建資料集友善的提供幫忙分訓練資料的功能 False：測試資料
    download=True,
    transform=ToTensor()
)

# train_data.data.shape # 看資料大小  (50000, 32, 32, 3)
# 提取各個分類的標籤名
class_names = train_data.classes

# 設定一次抓多少資料來訓練或測試
from torch.utils.data import DataLoader
# Turn datasets into iterables (batches)   設定變數 BATCH_SIZE = 32  移到最上面統一管理
train_dataloader = DataLoader(train_data, # dataset to turn into iterable
    batch_size=BATCH_SIZE, # how many samples per batch? 
    shuffle=True # shuffle data every epoch?
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False # don't necessarily have to shuffle the testing data
)
#%% CNN 模型規劃區
sb = [] # nn.Sequential for self.Block_1
sc = [] # nn.Sequential for self.classifier
sc.append(nn.Flatten())
Fcount = 0
CNNdescribe = []

print("建立總共",len(CRPF),"層的CNN，輸入層為彩色圖片32*32*3，這是第C 1 層：")
OC = eval(input("out_channels="))
KS = eval(input("kernel_size="))
CS = eval(input("stride="))
CP = eval(input("padding="))
PR = (KS*KS*3+1)*OC # 參數量計算
sb.append(nn.Conv2d(in_channels=3,out_channels=OC,kernel_size=KS,stride=CS,padding=CP))
IC = OC # 下一層的in_channels
HW = (32 + 2*CP - KS)//CS + 1 # 下一層的Height Width
CNNdescribe.append(["CNN模型：", CRPF, "輸入層為彩色圖片32*32*3 \n 第,1,層為卷積層 kernel_size=", KS, "輸出為",HW,"*",HW,"*",OC,"\n"])

for i in range(1,len(CRPF)):
    if CRPF[i] == "C":
        print("建立總共",len(CRPF),"層的CNN，上一層傳入",HW,"*",HW,"*",IC, " ，這是第C",i+1,"層：")
        OC = eval(input("out_channels="))
        KS = eval(input("kernel_size="))
        CS = eval(input("stride="))
        CP = eval(input("padding="))
        PR = PR + (KS*KS*3+1)*OC # 參數量計算
        sb.append(nn.Conv2d(in_channels=IC,out_channels=OC,kernel_size=KS,stride=CS,padding=CP))
        IC = OC # 下一層的in_channels
        HW = (HW + 2*CP - KS)//CS + 1 # 下一層的Height Width
        CNNdescribe.append(["第",i+1,"層為卷積層 kernel_size=", KS, "輸出為",HW,"*",HW,"*",OC,"\n"])
    elif CRPF[i] == "R":
        sb.append(nn.ReLU())
        CNNdescribe.append(["第",i+1,"層為ReLU層\n"])
    elif CRPF[i] == "P":
        print("建立總共",len(CRPF),"層的CNN，上一層傳入",HW,"*",HW,"*",IC, " ，這是第P",i+1,"層：")
        KS = eval(input("kernel_size="))
        sb.append(nn.MaxPool2d(kernel_size=KS))
        HW = HW / KS # 下一層的Height Width
        CNNdescribe.append(["第",i+1,"層為池化層 使用MaxPool, kernel_size=", KS, "輸出為",HW,"*",HW,"*",OC,"\n"])
    elif CRPF[i] == "F" :
        if Fcount == 0:
            INF = int(HW*HW*IC)
            Fcount += 1
        if i == len(CRPF)-1:
            PR = PR + (INF+1)*10
            sc.append(nn.Linear(in_features=INF, out_features=10))
            CNNdescribe.append(["第",i+1,"層為全連接層 , 輸出feature數為10個\n", "整個模型的參數量為", PR])
        else:
            print("建立總共",len(CRPF),"層的CNN，上一層傳入",INF,"個features，這是第F",i+1,"層：")
            OF = eval(input("out_features="))
            PR = PR + (INF+1)*OF
            sc.append(nn.Linear(in_features=INF, out_features=OF))
            INF = OF
            CNNdescribe.append(["第",i+1,"層為全連接層 , 輸出feature數為",OF,"個\n"])        
    else:
        continue

# paste1 = []
# paste2 = []
# for j in range(CRPF.count("C") + CRPF.count("R") + CRPF.count("P")):
#     paste1.append("sb["+str(j)+"]")
# for j in range(CRPF.count("F")):
#     paste2.append("sb["+str(j)+"]")
# print(paste1,"\n",paste2)
#%% CNN
#設計CNN藍圖
class CIFAR10V1(nn.Module):

    def __init__(self):
        super().__init__()
        self.block_1 = nn.Sequential( sb[0],sb[1],sb[2] )
        self.classifier = nn.Sequential( sc[0],sc[1] )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.classifier(x)
        return x

torch.manual_seed(42)
model_1 = CIFAR10V1()

# Setup loss and optimizer  
loss_fn = nn.CrossEntropyLoss() #    nn.CrossEntropyLoss() 用在 機率 或 類別
optimizer = torch.optim.SGD(params=model_1.parameters(), lr = LR)

torch.manual_seed(42)
start = time.time() # 計時開始 開始時間點
# Train and test model    設定變數epochs = 3  移到最上面集中管理
for epoch in range(epochs):
    print(f"Epoch: {epoch}\n---------")
    ### Training
    train_loss = 0
    # Add a loop to loop through training batches
    for batch, (X, y) in enumerate(train_dataloader):
        model_1.train() 
        # 1. Forward pass
        y_pred = model_1(X)

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulatively add up the loss per epoch 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Print out how many samples have been seen
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

    # Divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss /= len(train_dataloader)
end = time.time() # 結束時間點1
total_time = end - start  # 費時多久1

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.
    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.
    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


#%%
torch.manual_seed(42)
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval() # 評估model的正確率用
    with torch.inference_mode():
        
        # err_ypred = {}
        # err_ytrue = {}
        # for i in torch.Tensor(range(10)):
        #     err_ypred[i] = torch.Tensor(0)
        #     err_ytrue[i] = torch.Tensor(0)
        y_CM_pred = []
        y_CM_true = []
        
        for X, y in data_loader:
            # Make predictions with the model
            y_pred = model(X)
            
            
            
            y_CM_pred.extend((torch.max(torch.exp(y_pred), 1)[1]).data.cpu().numpy())
            y_CM_true.extend(y.data.cpu().numpy())
            # y_pre = y_pred.argmax(dim=1)
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, 
                                y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)
        
        # build CM
        CM = confusion_matrix(y_CM_true, y_CM_pred)
        df_CM = pd.DataFrame(CM / np.sum(CM, axis=1)[:, None], index=[i for i in class_names], columns=[i for i in class_names])
        plt.figure(figsize= (12,7))
        sn.heatmap(df_CM, annot=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        CMfileName = f"CM_{CRPF}_{timestamp}.png"
        plt.savefig(CMfileName)
        
        
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}
            # "err_ypred:": err_ypred,
            # "err_ytrue:": err_ytrue}


# 計算訓練過後的model用在測試資料集的正確率
model_1_results = eval_model(model=model_1, data_loader=test_dataloader,
    loss_fn=loss_fn, accuracy_fn=accuracy_fn
)

#%%


print(model_1)
print(model_1_results)
print("訓練時間：",total_time)

# to_csv
df1 = pd.DataFrame(CNNdescribe)
df2 = pd.DataFrame(model_1_results, index=[0])
df2["training_time"] = total_time
df3 = pd.concat([df1, df2], axis=0)
csv_file_path = 'CNN結果輸出.csv'
if os.path.exists(csv_file_path):
    df3.to_csv(csv_file_path, mode='a', header=False, index=False)
else:
    df3.to_csv(csv_file_path, mode='w', header=True, index=False)

#%% 雜記
# # hello world  傳說中的LeNet-5 將Sigmoid() 改成ReLU()的版本
#         self.block_1 = nn.Sequential(
#             nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1,padding=0), # 32 -> 28
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2), # 28 -> 14
#             nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0), # 14 -> 10
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2), # 10 ->5
#             )

#         self.classifier = nn.Sequential(
#             nn.Flatten(),# DENSE LAYER
#             nn.Linear(in_features=16*5*5, out_features=120),
#             nn.ReLU(),
#             nn.Linear(in_features=120, out_features=84),
#             nn.ReLU(),
#             nn.Linear(in_features=84,out_features=output_shape)
#             ) 

