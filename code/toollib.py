import numpy as np
class PictureTool():
    def preprocess(net, img):
       # print img.shape
        return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

    def deprocess(net, img):
        return np.dstack((img + net.transformer.mean['data'])[::-1])

    def filter_part(mask,w,h):
        
