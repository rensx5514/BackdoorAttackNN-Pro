'''
reference:
https://pytorch.org/docs/stable/torchvision/models.html
'''
import os 
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from skimage.restoration import denoise_tv_bregman
import torch.optim as optim

import imageio
# import scipy.ndimage as nd
# import PIL.Image
#import toollib

# class SensitivityVerificationData():
#     def __init__(self,channal:int,w:int,l:int,net:nn.Module,data = torch.rand(100,100)) -> None:
#         self.channal,self.data,self.w,self.l,self.net = channal,data,w,l,net
#         pass

#     # def MaxActivate():
#     #     self.net.fc2
#     #     pass

#     def checkAndSetEnv(var_env:str) -> bool:
#         if var_env not in os.environ:
#             print("No set os environ",var_env)
#             os.environ[var_env] = '/root/TranNNPro'
#             print("asda")
#         else:
#             print("Get it")
#         pass

class ReverseNetwork():
    def __init__(self) :
        #设置超参数
        self.net = models.vgg16(pretrained = True)
        self.trigger = torch.rand(1,3,224,224)
        self.trigger.requires_grad = True
        self.target_act = 80
        self.best_data = None
        self.count = 0
        self.octaves = [
        {
            'margin': 0,
            'window': 0.3, 
            'iter_n':190,
            'start_denoise_weight':0.001,
            'end_denoise_weight': 0.05,
            'start_step_size':11.,
            'end_step_size':11.
        },
        {
            'margin': 0,
            'window': 0.3,
            'iter_n':150,
            'start_denoise_weight':0.01,
            'end_denoise_weight': 0.08,
            'start_step_size':6.,
            'end_step_size':6.
        },
        {
            'margin': 0,
            'window': 0.3,
            'iter_n':550,
            'start_denoise_weight':0.01,
            'end_denoise_weight': 2,
            'start_step_size':1.,
            'end_step_size':1.
        },
        {
            'margin': 0,
            'window': 0.1,
            'iter_n':30,
            'start_denoise_weight':0.1,
            'end_denoise_weight': 2,
            'start_step_size':3.,
            'end_step_size':3.
        },
        {
            'margin': 0,
            'window': 0.3,
            'iter_n':50,
            'start_denoise_weight':0.01,
            'end_denoise_weight': 2,
            'start_step_size':6.,
            'end_step_size':3.
        }
        ]
        #随机化输入数据

        pass   
    def genTriggerMask(self, w = 224,h = 224,option = 0):
        if option == 0:
            data = imageio.imread('/root/TranNNPro/code/apple4.pgm')
            mask = torch.zeros(w,h)
            for y in range(0, h):
                for x in range(0, w):
                    if x > w - 105 and x < w - 20 and y > h - 105 and y < h - 20:
                        if data[y - (h-105), x - (w-105)] < 50:
                            mask[y, x] = 1
            self.mask = mask
        pass


    def start(self):
        self.genTriggerMask()
        base_img = torch.rand(1,3,224,224)
        self.max_activation(self.net, "", base_img, self.octaves)


    def make_step(self,net: nn.Module,input_data:torch.Tensor,criterion:nn.CrossEntropyLoss,optimizer:optim.SGD,exp_lr_scheduler:optim.lr_scheduler.StepLR,pred_unit,step_size=0.5,end='fc8', clip=True, unit=None, denoise_weight=0.1, margin=0, w=224, h=224):
        # global terminated, best_data, best_score
        learnrate = 100
        self.count += 1
        #input_data.to(device)
        if self.count > 1:
            self.trigger = torch.clone(self.trigger)
        output = net(self.trigger)
        best_act,best_unit = torch.max(output[0],0)
        obj_act = output.data[0][unit].reshape(1)
        optimizer.zero_grad()
        #是否需要将其他值置为0
        target = torch.tensor([81])
        loss = criterion(output, target)
        if input_data.requires_grad != True :
            input_data.requires_grad = True
        if loss.requires_grad != True:
            loss.requires_grad != True
        # loss.retain_grad()
        try:
            loss.backward(retain_graph=True)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("hhhhh zibi ")
            else:
                raise exception
        g = self.trigger.grad
        g *= learnrate
        #是否需要margin
        g *= self.mask
        print("gradient:",g.abs().mean())
        if (g.mean() == 0):
            print('too small abs mean')

        self.trigger = self.trigger + step_size/g.abs().mean()*g
        self.trigger *= self.mask
        
        if self.best_data is None or obj_act > self.target_act:
            self.best_data = torch.clone(self.trigger)
           
        return best_unit, best_act, obj_act
        if clip:
            bias = self.trigger.mean()
            self.trigger = torch.clip(self.trigger,-bias,255-bias)
      
        # what is TV denoising process? 
        # 重置param为0
        #只对triger区域进行
        #再次forward 检查分类器 并return best_unit,best_act,obj_act
        pass
    def max_activation(self,net, layer, base_img, octaves, debug=True, unit=81,clip=True, **step_param):
        #转化通道+减均值
        #image = preprocess(net, base_img)
        # transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(10),
        #       ToTensors
        #     transforms.Normalize()
        # ])
        image = base_img
        #添加loss
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        for e,o in enumerate(octaves):
            #检查是否需要插值 o['scale'] if it exists
            for i in range(o['iter_n']):
                # check image width random crop
                step_size = (o['start_step_size'] + (o['end_step_size'] - o['start_step_size'])*i) / o['iter_n']
                denoise_weight = o['start_denoise_weight'] - ((o['start_denoise_weight'] - o['end_denoise_weight']) * i) / o['iter_n']
                #pred_unit,step_size=0.5,end='fc8', clip=True, unit=None, denoise_weight=0.1, margin=0, w=224, h=224):
                best_unit, best_act, obj_act = self.make_step(net,image,criterion,optimizer_ft,exp_lr_scheduler,pred_unit=66,
                    step_size=step_size,denoise_weight=denoise_weight,unit = 81)
                #xy,end=layer,clip=clip,unit=unit,
                #        step_size=step_size,denoise_weight=denoise_weight,margin=o['margin'],w=w, h=h)
        pass
def testClassify(net):
    pass

def initTest():
    '''
    All pre-trained models expect input images normalized in the same way, 
    i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where 
    H and W are expected to be at least 224. The images have to be loaded 
    in to a range of [0, 1] and then normalized using mean = [0.485, 0.456
    , 0.406] and std = [0.229, 0.224, 0.225]. You can use the following 
    transform to normalize:
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    vgg16 = models.vgg16(pretrained = True)
    feature = torch.nn.Sequential(*list(vgg16.children())[:])
    print(feature)
    """
    DROPOUT
    """
    #vgg16.
    '''
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, num_classes),
    '''
    N, D = 64, 100 
    x = torch.rand(1,3,224,224)
    y = torch.rand(3,4,5,2)

    #print(y)
    #print(x)
    print(vgg16._modules.keys())
    print(vgg16.features(x))
    print(vgg16(x).shape)
    print(vgg16.classifier[6].in_feature)
    #output = vgg16(x)
    #print(output)
    #vgg16.
    #print(output)
    #print(vgg16.classifier[0].zero_grad())
    for p in vgg16.classifier[0].parameters():
        print(p.grad)
    #print(vgg16.classifier[0].grad)
    print(vgg16.classifier[1])
    print(vgg16.classifier[2])
    print(vgg16.classifier[3])
    print("stop")



    pass

if __name__ == "__main__":
    #initTest()
    testRev = ReverseNetwork()
    testRev.start()
    'pass'

