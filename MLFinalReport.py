"""
Created on Wed May 29 22:22:47 2024
@author: Posen Chen
跑程式的電腦設備： MacBook Air M2
Ver 0.1 從老師提供的程式碼裡剪貼出必要的部分 使用CIFAR-10資料集 調整參數讓程式可順利運行
Ver 0.2 預計增加計算訓練時間的code 以及刪除修改一些註解
"""
import time # 量測量時間成本用
import torch
from torch import nn
# import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

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
# See first training sample
# image, label = train_data[0]
# image, label
#1.1 Input and output shapes of a computer vision model
# What's the shape of the image?
# image.shape
# How many samples are there? 
train_data.data.shape
# See classes
class_names = train_data.classes
class_names

#Let's create DataLoader's for our training and test sets.
from torch.utils.data import DataLoader
# 設定batch size  每次forward back抓幾筆資料來訓練
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data, # dataset to turn into iterable
    batch_size=BATCH_SIZE, # how many samples per batch? 
    shuffle=True # shuffle data every epoch?
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False # don't necessarily have to shuffle the testing data
)

# Check out what's inside the training dataloader
# train_features_batch, train_labels_batch = next(iter(train_dataloader))
# train_features_batch.shape, train_labels_batch.shape

## CNN
#設計CNN藍圖
class CIFAR10V1(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, # 輸入圖片的channel數  灰階=1  RGB=3
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image? 就是filter的大小
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1, # 
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4) #預設的stride=kernel_size  kernel_size=2 will reduce the image to 14*14 for orginal 28*28 image; kernel_size=4 will end up 7*7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),# DENSE LAYER
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*8*8, 
                      out_features=output_shape) # output_shape = 共幾類  這種output在 pytorch裡叫 logit
        ) # 因為  nn.CrossEntropyLoss() 的input 用的是 "unnormalized" logits  所以不用在加nn.sigmoid() 不用 normalized 
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.classifier(x)
        return x

torch.manual_seed(42)
model_1 = CIFAR10V1(input_shape=3, 
    hidden_units=10, 
    output_shape=len(class_names))
model_1


# Setup loss and optimizer  
loss_fn = nn.CrossEntropyLoss() #    nn.CrossEntropyLoss() 用在 機率 或 類別
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)


torch.manual_seed(42)
start = time.time() # 計時開始 開始時間點
# Train and test model 
epochs = 3
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
print(model_1_results)
print("訓練時間：",total_time)