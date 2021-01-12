#from TranNNPro.code.ReverseNetwork import ReverseNetwork
from ReverseNetwork import ReverseNetwork
import os 
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transform
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

def TestNetParam(net:nn.Module):
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)

def TestReverseNetwork():

    input_data = torch.rand(1,3,224,224)
    vgg16net = models.vgg16(pretrained = True)
    TestNetParam(vgg16net)
    #256*3*3*3
    敏感度分析
    # output = vgg16net(input_data)
    # print(vgg16net.features[0])
    
    # criterion = nn.CrossEntropyLoss()
    # optimizer_ft = optim.SGD(vgg16net.parameters(), lr=0.001, momentum=0.9)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    # for name,fc in vgg16net.classifier[6].named_parameters():
    #     print(name)
    #     if name == "weight":
    #         # print("weight")
    #         # #[1000,4029]
    #         # print(fc.grad)
    #         # print(fc.grad_fn)
    #         # print(fc[0].shape)
    #         # print(fc[1])
    #     elif name == "bias":
    #         print("bias:")
    #         print(fc.shape)
        
    #TestNetParam(vgg16net)
    print(output.data.shape)
    _, preds = torch.max(output.data, )
    #loss = criterion(output, )
    output.data.backward()
    #将output tensor 设置为0
    # data_mask = torch.zeros_like(output.data)
    # best_act,best_unit = torch.max(output[0],0)
    # obj_act = output.data[0][110]
    # data_mask[0][110] = obj_act
    #output.data.copy_(data_mask)
    output.data.backward()
    print(output.grad)
    print(output.grad_fn)
    print(output.shape)
    ReverseNetwork.make_step(vgg16net,input_data,unit=100)
if __name__ == "__main__":
    TestReverseNetwork()
    pass