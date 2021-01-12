#from TranNNPro.code.ReverseNetwork import ReverseNetwork
import os 
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transform
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable


from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import math
import time
   


def initNet():
    return models.vgg16(pretrained = True)

def wrapper(x,net,colum_step,row_step,edge_window,w,h):
    n = x.shape[0]   # Get number of samples
    # Create result array 
    # (assumes function returns a single value - adjust this accordingly)
    Y = np.zeros(n)  

    for i in range(n):
        # Reshape 27 elements in to 3x3x3 array
        reshaped_x = np.pad(x[i].reshape((3,edge_window,edge_window)),((0,0),(edge_window*row_step,w-edge_window-edge_window*row_step),(edge_window*colum_step,h-edge_window-edge_window*colum_step)),'constant')
        # add a dim
        input_x =  torch.from_numpy(np.expand_dims(reshaped_x,axis=0))
        input_x = input_x.float()
        # Run function and store result
        output_y = net(input_x)
        Y[i] = output_y[0][81].tolist()
    
    return Y

# def evaluate(X):  # 这里是我们要进行灵敏度分析的模型,接受一个数组,每个数组元素作为模型的一个输入,模型的输出是一个float,干函数返回的时候再讲所有输出并起来
#     return np.array([math.sin(x[0]) + x[1] * math.cos(2 * x[2]) for x in X])

def evaluateTrigger(w,h,pct,net,epoch,num):
    window_triggers = int(w*h*3*pct/num)
    edge_window = int(math.sqrt(window_triggers))
    step = int(w / edge_window)
    results = []
    problem = {
    'num_vars': 3*edge_window*edge_window,
    # 57 * 57 * 3
    'names': [f'x{i+1}' for i in range(3*edge_window*edge_window)],
    'bounds': [[0, 255]]*(3*edge_window*edge_window),
    'groups': ['group1']*(3*edge_window*edge_window)
    }
    for r in range(step):
        for c in range(step):
            time_start=time.time()
            param_values = saltelli.sample(problem, epoch)
            Y = wrapper(param_values,net,c,r,edge_window,w,h)
            Si = sobol.analyze(problem, Y, print_to_console=True)
            results.append(Si)
            time_end=time.time()
            print('col:',c,' row:',r,' time cost',time_end-time_start,'s')
    results


def TestSALib():
    net = initNet()
    time_start=time.time()
    param_values = saltelli.sample(problem, 100)
    time_end=time.time()
    print('time cost',time_end-time_start,'s')
    # Run model (example)
    Y = wrapper(param_values,net)
    print(param_values.shape, Y.shape)
    # Perform analysis (这里运行完成后会自动对结果进行展示)
    Si = sobol.analyze(problem, Y, print_to_console=True)

    # Print the first-order sensitivity indices  一阶灵敏度
    print('S1:', Si['S1'])

    # Print the second-order sensitivity indices   二阶灵敏度
    # print("x2-x3:", Si['S2'][1, 2])
def TestNetParam(net:nn.Module):
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)

def TestReverseNetwork():

    input_data = torch.rand(1,3,224,224)
    output = vgg16net(input_data)
    print(output.data.shape)
    print(output.data[100])
    # TestNetParam(vgg16net)
if __name__ == "__main__":
    net = initNet()
    f = open("output.txt","a")
    results = evaluateTrigger(224,224,0.07,net,1000,3)
    print(results,file = f)
    f.close()
    #TestSALib()
    pass