import torch
import torch.nn as nn
import torch.utils.data as data

class train_set(data.Dataset):
    def __init__(self):
        # 初始化
        self.data=[1,2,3,4,5,6,7]
    def __getitem__(self, item):
        return torch.ones((2,2)),self.data[item]
    def __len__(self):
        return len(self.data)

train_set=train_set()
dataloader=data.DataLoader(train_set,batch_size=2,shuffle=False)
# for j in range(1):
#     data_iter=iter(dataloader)
#     num_image=4
#     # 循环:
#     for i in range(num_image):
#         x=next(data_iter)
#         print(type(x),len(x))#<class 'list'> 2
        # print(type(x[0]), x[0].shape)#<class 'torch.Tensor'> torch.Size([2, 2, 2])
        # print(type(x[1]),x[1].shape)#<class 'torch.Tensor'> torch.Size([2])
    # print('---------我是分割线-----------')

for j in range(1):
    # data_iter=iter(dataloader)
    # num_image=4
    # # 循环:
    num_image=4
    for i,data in enumerate(dataloader):
        print(type(data),len(data))#<class 'list'> 2
        print(type(data[0]),data[0].shape)#<class 'torch.Tensor'> torch.Size([2, 2, 2])
        print(type(data[1]),data[1].shape)#<class 'torch.Tensor'> torch.Size([2])
        if i==num_image-1:
            break
    # print('---------我是分割线-----------')