import numpy as np
from torch.utils.data import Dataset 
import torchvision.transforms as transforms
import torch
import os 
from PIL import Image

# dataloader
def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def make_dataset(dir, class_to_idx):
    rgb = []
    ss = []
    de = []
    nm = []
    msk = []
    names = [rgb, ss, de, nm, msk]
    dir = os.path.expanduser(dir)
    target_dirs = ['rgb','ss','de','nm','msk']
    for i in range(len(target_dirs)):
        target_name = names[i]
        target_dir = target_dirs[i]
        for target in sorted(os.listdir(os.path.join(dir, target_dir))):
            d = os.path.join(dir, target_dir, target)
            if not os.path.isdir(d):
                continue
            
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    target_name.append(item)
    return rgb, ss, de, nm, msk


class MultitaskDataset(Dataset):
    def __init__(self, root_dir, transform = None, rgb_transform = None, imsize = 128):
        self.transform = transform
        self.rgb_transform = rgb_transform
        self.imsize = imsize
        classes,  class_to_idx = find_classes(os.path.join(root_dir,'rgb'))
        self.rgb, self.ss, self.de, self.nm, self.msk = make_dataset(root_dir, class_to_idx)

    def __getitem__(self, index):
        rgb_path, target = self.rgb[index]
        ss_path, _ = self.ss[index]
        de_path, _ = self.de[index]
        nm_path, _ = self.nm[index]
        msk_path, _ = self.msk[index]
        rgb_item = pil_loader(rgb_path)
        nm_item = pil_loader(nm_path)
        msk_item = pil_loader(msk_path)
        ss_item = np.resize(np.load(ss_path),(self.imsize, self.imsize)).astype('int16')
        de_item = np.resize(np.load(de_path),(self.imsize, self.imsize))
        ss_item = torch.Tensor(ss_item)
        de_item = torch.Tensor(de_item)
        if self.transform is not None:
            nm_item = self.transform(nm_item)
            msk_item = self.transform(msk_item)
        if self.rgb_transform is not None:
            rgb_item = self.rgb_transform(rgb_item)
        sample = {
            'rgb':rgb_item,
            'ss':ss_item,
            'de':de_item,
            'sn':nm_item,
            'msk':msk_item,
            'sc':target
        }
        return sample

    def __len__(self):
        return len(self.rgb)

def nyu_dataset(data_path, imsize):
    rgb_tsf = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    tsf = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
    ])
    nyu_set = MultitaskDataset(data_path, tsf, rgb_tsf, imsize)
    return nyu_set
