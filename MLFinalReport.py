"""
Created on Wed May 29 22:22:47 2024
@author: Posen Chen
跑程式的電腦設備： MacBook Air M2
Ver 0.1 從老師提供的程式碼裡剪貼出必要的部分 使用CIFAR-10資料集 調整參數讓程式可順利運行
Ver 0.2 增加簡單計算訓練時間的code 以及刪除修改一些註解
Ver 0.3 超參數置頂 結果可輸出至CSV 但表格內容還沒設計好

模型參數調整計算公式
結果繪圖
迴圈 想要一次跑多組 超難寫 哭哭
測試其他參數

期末報告及評分方式
以Pytorch建立神經網路模型，在以下超參數進行測試(上課時讓各組各選三個,去嚐試各3種情況)
(2)卷積層的數目
(3)卷積層filter的數目.
(4)pooling的size
"""
import time # 量測量時間成本用
import torch
from torch import nn
# import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import os

''' 
超參數設定區    為方便調整參數 將參數們集中置頂
''' # 卷積層的數目 原則：1.第一層為C 2.最後為F 3.F前面為P 4.C後面必接R(因為本次不測試不同的activation function) 5.R後面為C或P
CRPF = "CRCRPF" #, 4["CRPF"], 6["CRCRPF"], 7["CRPCRPF"], 8["CRCRCRPF"], 9["CRPCRCRPF", "CRCRPCRPF"] , 10["CRCRCRCRPF", "CRPCRPCRPF"]
CKS = 3 # [3, 5, 7] # kernel_size= 也就是卷積層filter 大小
CS = 1 # [1, 2, 3] # stride= 每次filter移動的步數
CP = 1 # [0, 1, 2] # padding= 圖片外圍多鋪幾層
PKS = 3 # [3, 4, 5] # nn.MaxPool2d(kernel_size= ) 也就是pooling的大小
PS = 4 # [1, 2, 3, 4, 5] # nn.MaxPool2d(預設的stride=kernel_size )  # 不在這次報告的測試範圍 就用4
epochs = 3 # 總共跑幾次epochs # 不在這次報告的測試範圍 就用3
BATCH_SIZE = 32 # 設定batch size  每次forward back抓幾筆資料來訓練  # 不在這次報告的測試範圍 就用32
HU = 10 # 設定每次convolution後output_channels的數量  # 不在這次報告的測試範圍 就用10
LR = 0.1 # 學習率  # 不在這次報告的測試範圍 就用0.1
# 不同的optimization方法

# 使用內建資料集 CIFAR10
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

## CNN
seq = []

for i in range(len(CRPF)):
    if i == 0:
        seq.append(nn.Conv2d(in_channels=3,out_channels=HU,kernel_size=CKS,stride=CS,padding=CP))
        continue
    elif CRPF[i] == "C":
        seq.append(nn.Conv2d(in_channels=HU,out_channels=HU,kernel_size=CKS,stride=CS,padding=CP))
    elif CRPF[i] == "R":
        seq.append(nn.ReLU())
    elif CRPF[i] == "P":
        seq.append(nn.MaxPool2d(kernel_size=4)),
    else:
        continue

# CRPF.count("C")
# outputw = 32 + CRPF.count("C") * ( 2 * CP - CKS + 1)  # 每經過一層C的大小改變量 一開始的圖片大小為32
# if outputw % == 0:
    
# LS = outputw / PKS

#設計CNN藍圖
class CIFAR10V1(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(seq[0],seq[1],seq[2],seq[3],seq[4])

        self.classifier = nn.Sequential(
            nn.Flatten(),# DENSE LAYER
            # Where did this in_features shape come from? 
            nn.Linear(in_features=hidden_units*8*8, 
                      out_features=output_shape) # output_shape = 共幾類  這種output在 pytorch裡叫 logit
        ) # 因為  nn.CrossEntropyLoss() 的input 用的是 "unnormalized" logits  所以不用在加nn.sigmoid() 不用 normalized 
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.classifier(x)
        return x

torch.manual_seed(42)
model_1 = CIFAR10V1(input_shape=3, 
    hidden_units = HU, 
    output_shape = len(class_names))

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
        for X, y in data_loader:
            # Make predictions with the model
            y_pred = model(X)
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, 
                                y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)
        
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}


# 計算訓練過後的model用在測試資料集的正確率
model_1_results = eval_model(model=model_1, data_loader=test_dataloader,
    loss_fn=loss_fn, accuracy_fn=accuracy_fn
)

print(model_1)
print(model_1_results)
print("訓練時間：",total_time)

#df1 = pd.Series(CRPF) # 未完成
# df2 = pd.Series(total_time)
df3 = pd.DataFrame(model_1_results, index=[0]).T


# to_csv
# listRMSE = []
# listTime = []

#                 listRMSE.append([i,j,k,l,"RMSE=", RMSE_Test16.values, "\n"])
#                 listTime.append([i,j,k,l,"Time=", total_time, "\n"])

# df4 = pd.concat(df2, df3, axis=0)
csv_file_path = 'CNN結果輸出.csv'
if os.path.exists(csv_file_path):
    df3.to_csv(csv_file_path, mode='a', header=False, index=False)
else:
    df3.to_csv(csv_file_path, mode='w', header=True, index=False)
