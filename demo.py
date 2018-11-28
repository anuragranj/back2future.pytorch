import torch
from back2future import Model
import numpy as np
from scipy.misc import imread, imresize
from torchvision.transforms import ToTensor
from torch.autograd import Variable

def main():
    model = Model(pretrained='pretrained/back2future_kitti.pth.tar')
    model = model.cuda()
    im_tar, im_refs = fetch_image_tensors()
    im_tar = Variable(im_tar.unsqueeze(0)).cuda()
    im_refs = [Variable(im_r.unsqueeze(0)).cuda() for im_r in im_refs]
    flow_fwd, flow_bwd, occ = model(im_tar, im_refs)
    np.save('outputs.npy', {'flow_fwd':flow_fwd[0].cpu().data.numpy(),
                            'flow_bwd':flow_bwd[0].cpu().data.numpy(),
                            'occ':occ[0].cpu().data.numpy()})
    print('Outputs saved in outputs.npy')
    print('Done!')

def load_as_float(path):
    return imread(path).astype(np.float32)

def fetch_image_tensors():
    im0 = load_as_float('samples/000010_09.png')
    im1 = load_as_float('samples/000010_10.png')
    im2 = load_as_float('samples/000010_11.png')
    scale = Scale(h=256, w=832)
    im012 = scale([im0, im1, im2])
    im_tar = im012[1]
    im_refs = [im012[0], im012[2]]
    return im_tar, im_refs

class Scale(object):
    """Scales images to a particular size"""
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, images):
        in_h, in_w, _ = images[0].shape
        scaled_h, scaled_w = self.h , self.w
        scaled_images = [ToTensor()(imresize(im, (scaled_h, scaled_w))) for im in images]
        return scaled_images

if __name__ == '__main__':
    main()
