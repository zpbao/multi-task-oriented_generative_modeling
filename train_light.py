import torch
import argparse
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split

from taskonomy_losses import *
from utils import *

from sagan_models import Generator, Discriminator
from mgm import MGM_light 
from datasets import nyu_dataset

parser = argparse.ArgumentParser()

parser.add_argument
# basic configurations
parser.add_argument('--data_path', default='./nyu-std/', type=str, help='where to load the data' )
parser.add_argument('--imsize', default = 128, type = int, help = 'input image size')
parser.add_argument('--batch_size', default = 32, type = int, help = 'batch size')
parser.add_argument('--epochs', default = 300, type = int, help = 'MGM training epochs')
parser.add_argument('--gan_epochs', default = 500, type = int, help = 'GAN pre-training epochs')
parser.add_argument('--seed', default = 42, type = int, help = 'random seed for torch')
parser.add_argument('--log_name', default = './log.txt', type = str, help = 'log file name')
parser.add_argument('--generator_save_epoch', default = 10, type = int, help = 'save GAN weights per X epoch')
parser.add_argument('--sample_epoch', default = 10, type = int, help = 'save generated images per x epoch')
parser.add_argument('--save_epoch', default = 10, type = int, help = 'save overall model weight per x epoch')

# model parameters
parser.add_argument('--lr', default = 0.001, type = float, help = 'learning rate for the multi-task model')
parser.add_argument('--post_lr', default = 0.005, type = float, help = 'learning rate for the post net')
parser.add_argument('--z_dim', default = 128, type = int, help = 'z dimension for the gan models')
parser.add_argument('--g_conv_dim', default = 64, type = int, help = 'conv dimension for the gan generator')
parser.add_argument('--num_of_classes', default = 27, type = int, help = 'number of scenes in the dataset')
parser.add_argument('--d_conv_dim', default = 64, type = int, help = 'conv dimension for the gan discriminator')
parser.add_argument('--g_lr', default = 0.0001, type = float, help = 'learning rate for GAN generator')
parser.add_argument('--d_lr', default = 0.0004, type = float, help = 'learning rate for GAN discriminator')
parser.add_argument('--gan_iters', default = 2, type = int, help = 'train G and D x times per step')
parser.add_argument('--num_sm_classes', default = 40, type = int, help = 'num of semantic labels')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # initializations
    opt = parser.parse_args()

    if not os.path.exists('./samples'):
        os.mkdir('./samples')
    if not os.path.exists('./models'):
        os.mkdir('./models')
    
    torch.manual_seed(opt.seed)
    logs = open(opt.log_name,'w')

    dset = nyu_dataset(opt.data_path,opt.imsize)
    label_set, unlabel_set, val_set = split_dataset(dset, 0.8, 0.5)
    label_ld = DataLoader(label_set, opt.batch_size, shuffle=True, num_workers=4,drop_last=True)
    unlabel_ld = DataLoader(unlabel_set, opt.batch_size, shuffle=True, num_workers=4,drop_last=True)
    val_ld = DataLoader(val_set, opt.batch_size, shuffle=True, num_workers=4,drop_last=True)

    model = MGM_light(opt.num_sm_classes, opt.num_of_classes).cuda()
    model.set_optimiziers(lr = opt.lr, post_lr = opt.post_lr, g_lr = opt.g_lr, d_lr = opt.d_lr)

    # train MGM-light
    
    ss_loss = segment_semantic_loss
    de_loss = depth_loss_simple
    sn_loss = normal2_loss
    scene_loss = classification_loss
    self_loss = NTXentLoss('cuda',opt.batch_size,0.5,True)

    losses = {
        'ss': ss_loss,
        'de': de_loss,
        'sn': sn_loss,
        'sc': scene_loss,
        'self': self_loss,
    }

    print('train MGM...')
    for epoch in range(opt.epochs):
        ss_loss_total = 0.0
        de_loss_total = 0.0
        sn_loss_total = 0.0
        scene_loss_total = 0.0
        label_it = iter(label_ld)
        unlabel_it = iter(unlabel_ld)
        for _ in range(len(unlabel_ld)):
            label_sample = next(label_it)
            returned_losses= model.train_label_step(label_sample, losses)
            ss_loss_step, de_loss_step, sn_loss_step, scene_loss_step = returned_losses
            ss_loss_total += ss_loss_step.item()
            de_loss_total += de_loss_step.item()
            sn_loss_total += sn_loss_step.item()
            scene_loss_total += scene_loss_step.item()
            unlabel_sample = next(unlabel_it)
            returned_losses = model.train_unlabel_step(unlabel_sample, losses)
        ss_loss_total /= len(label_ld)
        de_loss_total /= len(label_ld)
        sn_loss_total /= len(label_ld)
        scene_loss_total /= len(label_ld)
        print('mgm epoch:{}|train, ss loss: {:4f}, de loss:{:4f}, sn loss:{:4f}, scene loss:{:4f}'.format(epoch, 
        ss_loss_total, de_loss_total, sn_loss_total, scene_loss_total))
        print('mgm epoch:{}|train, ss loss: {:4f}, de loss:{:4f}, sn loss:{:4f}, scene loss:{:4f}'.format(epoch, 
        ss_loss_total, de_loss_total, sn_loss_total, scene_loss_total), file = logs)
            
        if epoch % opt.save_epoch == 0:
            torch.save(model.state_dict(), './models/mgm-{}.pt'.format(epoch))

    torch.save(model.state_dict(), './models/mgm.pt')
        
if __name__ == '__main__':
    main()