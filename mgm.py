import torch.nn as nn
import torch
import resnet_taskonomy

from torch.nn.utils import spectral_norm
import torchvision.utils as vutils
import utils

from sagan_models import Generator, Discriminator


class Multi_task(nn.Module):
    def __init__(self,num_classes = 40,block=resnet_taskonomy.BasicBlock,layers = [3,4,6,3], size = 1, clr_num = 64, **kwargs):
        super(Multi_task, self).__init__()
        self.encoder = resnet_taskonomy.ResNetEncoder(block, layers, [48,96,192,360],**kwargs) 
        self.num_classes = num_classes
        self.ss_decoder = resnet_taskonomy.Decoder(self.num_classes,base_match=360)
        self.de_decoder = resnet_taskonomy.Decoder(1,base_match=360)
        self.sn_decoder = resnet_taskonomy.Decoder(3,base_match=360)
        # simCLR decoder
        self.clr_decoder = resnet_taskonomy.Decoder(num_classes = clr_num,base_match=360)

    def forward(self, input):
        rep = self.encoder(input)
        ss_out = self.ss_decoder(rep)
        de_out = self.de_decoder(rep)
        sn_out = self.sn_decoder(rep)
        return ss_out, de_out, sn_out
    
    def clr_forward(self, input):
        rep = self.encoder(input)
        clr_out = self.clr_decoder(rep)
        return clr_out
    
    def encoding(self, x):
        return self.encoder(x)

class Post_net(nn.Module):
    def __init__(self, scene_classes, num_classes, block=resnet_taskonomy.BasicBlock,layers = [3,4,3], size = 1,**kwargs):
        super(Post_net, self).__init__()
        self.num_classes = num_classes
        self.scene_classes = scene_classes
        self.ss_encoder = resnet_taskonomy.PostEncoder(block,layers,num_classes,[48,96,192],**kwargs)
        self.de_encoder = resnet_taskonomy.PostEncoder(block,layers,1,[48,96,192],**kwargs)
        self.nm_encoder = resnet_taskonomy.PostEncoder(block,layers,3,[48,96,192],**kwargs)
        self.fc1 = nn.Linear(192*3, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, self.scene_classes)
        self.relu2 = nn.ReLU()
    
    def forward(self, ss, de, nm):
        ss_en = self.ss_encoder(ss)
        de_en = self.de_encoder(de)
        nm_en = self.nm_encoder(nm)
        feat = torch.cat([ss_en, de_en, nm_en], 1)
        f = self.fc1(feat)
        f = self.relu1(f)
        f = self.fc2(f)
        f = self.relu2(f)
        return f 

def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))

class Middle_net(nn.Module):
    def __init__(self, rep_dim, z_dim):
        super(Middle_net,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = snlinear(rep_dim, z_dim)
        self.bn = nn.BatchNorm1d(z_dim)
    
    def forward(self, input):
        feat = self.avgpool(input)
        feat = feat.squeeze()
        out = self.fc(feat)
        out = self.bn(out)
        return out

class MGM(nn.Module):
    def __init__(self,z_dim, g_conv_dim, d_conv_dim, num_sm_classes = 40, num_scenes = 27, base_match = 360, gan_hid_dim = 128):
        super(MGM, self).__init__()
        self.z_dim = z_dim
        self.g_conv_dim = g_conv_dim
        self.d_conv_dim = d_conv_dim
        self.num_sm_classes = num_sm_classes
        self.num_scenes = num_scenes
        self.base_match = base_match
        self.gan_hid_dim = gan_hid_dim
        self.multi_model = Multi_task(num_classes=self.num_sm_classes).cuda()
        self.post_model = Post_net(self.num_scenes,self.num_sm_classes).cuda()
        self.middle_net = Middle_net(self.base_match,self.gan_hid_dim).cuda()

        self.G = Generator(z_dim, g_conv_dim, num_scenes).cuda()
        self.D = Discriminator(d_conv_dim, num_scenes).cuda()
    
    def set_optimiziers(self, lr, post_lr, g_lr, d_lr, weight_decay = 1e-4):
        self.middle_opt = torch.optim.Adam(self.middle_net.parameters(), lr, (0.9, 0.999))
        self.G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), g_lr, (0.9, 0.999))
        self.D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), d_lr, (0.9, 0.999))
        self.multi_opt = torch.optim.Adam(self.multi_model.parameters(), lr, (0.9, 0.999), weight_decay = weight_decay)
        self.post_opt = torch.optim.Adam(self.post_model.parameters(), post_lr, (0.9, 0.999), weight_decay = weight_decay)
        self.encoder_opt = torch.optim.Adam(self.multi_model.encoder.parameters(),  lr, (0.9, 0.999), weight_decay = weight_decay)

    def train_gan_step_mgm(self,x, sc, gan_iters):
        for _ in range(gan_iters):
            # train D
            bs = x.shape[0]
            ones = torch.full((bs,), 1).cuda()
            inst_noise = torch.normal(mean = 0.0, std=0.01, size = (bs,3,128,128)).cuda()
            d_out_real = self.D(x + inst_noise, sc)
            d_loss_real = torch.nn.ReLU()(ones - d_out_real).mean()

            z = torch.randn(bs, self.z_dim).cuda()
            rep = self.multi_model.encoding(x)
            z_real = z + self.middle_net(rep)
            fake_images = self.G(z_real, sc).detach()
            inst_noise = torch.normal(mean = 0.0, std=0.01, size = (bs,3,128,128)).cuda()
            d_out_fake = self.D(fake_images + inst_noise, sc)
            d_loss_fake = torch.nn.ReLU()(ones + d_out_fake).mean()

            d_loss = d_loss_real + d_loss_fake
            self.D_optimizer.zero_grad()
            d_loss.backward()
            self.D_optimizer.step()

            # train G
            z = torch.randn(bs, self.z_dim).cuda()
            rep = self.multi_model.encoding(x)
            z_real = z + self.middle_net(rep)
            fake_images = self.G(z_real, sc)
            inst_noise = torch.normal(mean = 0.0, std=0.01, size = (bs,3,128,128)).cuda()
            g_out_fake = self.D(fake_images + inst_noise, sc.cuda())
            g_loss_fake = -g_out_fake.mean()
            g_loss = g_loss_fake
            self.G_optimizer.zero_grad()
            self.middle_opt.zero_grad()
            self.encoder_opt.zero_grad()
            g_loss.backward()
            self.G_optimizer.step()
            self.encoder_opt.step()
            self.middle_opt.step()
        return g_loss.item(), d_loss.item()
    
    def train_gan_step(self, x, sc, gan_iters):
        # train D
        for _ in range(gan_iters):
            bs = x.shape[0]
            ones = torch.full((bs,), 1).cuda()
            inst_noise = torch.normal(mean = 0.0, std=0.01, size = (bs,3,128,128)).cuda()
            d_out_real = self.D(x + inst_noise, sc)
            d_loss_real = torch.nn.ReLU()(ones - d_out_real).mean()

            z = torch.randn(bs, self.z_dim).cuda()
            rep = self.multi_model.encoding(x)
            z_real = z
            fake_images = self.G(z_real, sc).detach()
            inst_noise = torch.normal(mean = 0.0, std=0.01, size = (bs,3,128,128)).cuda()
            d_out_fake = self.D(fake_images + inst_noise, sc)
            d_loss_fake = torch.nn.ReLU()(ones + d_out_fake).mean()

            d_loss = d_loss_real + d_loss_fake
            self.D_optimizer.zero_grad()
            d_loss.backward()
            self.D_optimizer.step()

            # train G
            z = torch.randn(bs, self.z_dim).cuda()
            rep = self.multi_model.encoding(x)
            z_real = z 
            fake_images = self.G(z_real, sc)
            inst_noise = torch.normal(mean = 0.0, std=0.01, size = (bs,3,128,128)).cuda()
            g_out_fake = self.D(fake_images + inst_noise, sc.cuda())
            g_loss_fake = -g_out_fake.mean()
            g_loss = g_loss_fake
            self.G_optimizer.zero_grad()
            g_loss.backward()
            self.G_optimizer.step()
        return g_loss.item(), d_loss.item()
    
    def sample_gan(self,sc,path):
        bs = sc.shape[0]
        z = torch.randn(bs, self.z_dim).cuda()
        self.G.eval()
        fake_images = self.G(z, sc)
        self.G.train()
        sample_images = utils.denorm(fake_images.detach())
        # Save batch images
        vutils.save_image(sample_images, path)
        # Delete output
        del fake_images
    
    def sample_gan_mgm(self,x,sc,path):
        bs = sc.shape[0]
        z = torch.randn(bs, self.z_dim).cuda()
        rep = self.multi_model.encoding(x)
        z_real = z + self.middle_net(rep)
        self.G.eval()
        fake_images = self.G(z_real, sc)
        self.G.train()
        sample_images = utils.denorm(fake_images.detach())
        # Save batch images
        vutils.save_image(sample_images, path)
        # Delete output
        del fake_images

    def train_CLR(self,rgb, self_loss):
        clr_trans = utils.simCLR_trans()
        rgb = (rgb + 1)*255.0 / 2 
        bs = rgb.shape[0]
        xi = rgb.clone()
        xj = rgb.clone()
        for k in range(bs):
            xi[k] = clr_trans(xi[k])
            xj[k] = clr_trans(xj[k])
        xi = torch.autograd.Variable(xi).cuda()
        xj = torch.autograd.Variable(xj).cuda()
        xi_pred = self.multi_model.clr_forward(xi)
        xj_pred = self.multi_model.clr_forward(xj)
        self_supervised_loss = self_loss(xi_pred,xj_pred)
        self.multi_opt.zero_grad()
        self_supervised_loss.backward()
        self.multi_opt.step()
        del xi_pred,xj_pred 
        return self_supervised_loss.item()

    def train_mgm(self, sample, losses, gan_iters):
        rgb = sample['rgb'].cuda()
        ss = sample['ss'].cuda().long()
        de = sample['de'].cuda()
        sn = sample['sn'].cuda()
        msk = sample['msk'].cuda()
        sc = sample['sc'].cuda().long()
        bs = rgb.shape[0]

        ss_loss = losses['ss']
        de_loss = losses['de']
        sn_loss = losses['sn']
        scene_loss = losses['sc']
        self_loss = losses['self']

        x = torch.autograd.Variable(rgb).cuda()
        y1 = torch.autograd.Variable(ss).cuda()
        y2 = torch.autograd.Variable(de).cuda()
        y3 = torch.autograd.Variable(sn).cuda()
        y4 = torch.autograd.Variable(msk).cuda()

        # multi-task learning
        y1_pred, y2_pred, y3_pred = self.multi_model(x)

        loss1 = ss_loss(y1_pred, y1, torch.ones_like(y1).cuda())
        loss2 = de_loss(y2_pred.squeeze(), y2, torch.ones_like(y2).cuda())
        loss3 = sn_loss(y3_pred, y3, y4)

        multi_task_loss = loss1 + loss2 + loss3

        self.multi_opt.zero_grad()
        multi_task_loss.backward()
        self.multi_opt.step()

        # refinement network learning
        y1_pred, y2_pred, y3_pred = self.multi_model(x)
        cls_pred = self.post_model(y1_pred,y2_pred,y3_pred)
        loss4 = scene_loss(cls_pred, sc)
        self.encoder_opt.zero_grad()
        self.post_opt.zero_grad()
        loss4.backward()
        self.post_opt.step()
        self.encoder_opt.step()

        # self-supervised learning
        loss5 = self.train_CLR(rgb,self_loss)

        # train GAN
        gloss, dloss = self.train_gan_step_mgm(rgb, sc, gan_iters)

        # train GAN with refinement and self-supervision
        z = torch.randn(bs, self.z_dim).cuda()
        rep = self.multi_model.encoding(rgb.cuda())
        z_real = z + self.middle_net(rep)
        fake_images = self.G(z_real, sc).detach()
        fake_images = torch.autograd.Variable(fake_images).cuda()
        y1_pred, y2_pred, y3_pred = self.multi_model(fake_images)
        cls_pred = self.post_model(y1_pred,y2_pred,y3_pred)
        loss6 = scene_loss(cls_pred, sc)
        self.encoder_opt.zero_grad()
        loss6.backward()
        self.encoder_opt.step()

        loss7 = self.train_CLR(fake_images,self_loss)

        returned_losses = [loss1, loss2, loss3, loss4]
        return returned_losses

class MGM_light(nn.Module):
    def __init__(self, num_sm_classes = 40, num_scenes = 27, base_match = 360):
        super(MGM_light, self).__init__()
        self.num_sm_classes = num_sm_classes
        self.num_scenes = num_scenes
        self.base_match = base_match
        self.multi_model = Multi_task(num_classes=self.num_sm_classes).cuda()
        self.post_model = Post_net(self.num_scenes,self.num_sm_classes).cuda()

    def set_optimiziers(self, lr, post_lr, g_lr, d_lr, weight_decay = 1e-4):
        self.multi_opt = torch.optim.Adam(self.multi_model.parameters(), lr, (0.9, 0.999), weight_decay = weight_decay)
        self.post_opt = torch.optim.Adam(self.post_model.parameters(), post_lr, (0.9, 0.999), weight_decay = weight_decay)
        self.encoder_opt = torch.optim.Adam(self.multi_model.encoder.parameters(),  lr, (0.9, 0.999), weight_decay = weight_decay)

    def train_CLR(self,rgb, self_loss):
        clr_trans = utils.simCLR_trans()
        rgb = (rgb + 1)*255.0 / 2 
        bs = rgb.shape[0]
        xi = rgb.clone()
        xj = rgb.clone()
        for k in range(bs):
            xi[k] = clr_trans(xi[k])
            xj[k] = clr_trans(xj[k])
        xi = torch.autograd.Variable(xi).cuda()
        xj = torch.autograd.Variable(xj).cuda()
        xi_pred = self.multi_model.clr_forward(xi)
        xj_pred = self.multi_model.clr_forward(xj)
        self_supervised_loss = self_loss(xi_pred,xj_pred)
        self.multi_opt.zero_grad()
        self_supervised_loss.backward()
        self.multi_opt.step()
        del xi_pred,xj_pred 
        return self_supervised_loss.item()

    def train_label_step(self, sample, losses):
        rgb = sample['rgb'].cuda()
        ss = sample['ss'].cuda().long()
        de = sample['de'].cuda()
        sn = sample['sn'].cuda()
        msk = sample['msk'].cuda()
        sc = sample['sc'].cuda().long()
        bs = rgb.shape[0]

        ss_loss = losses['ss']
        de_loss = losses['de']
        sn_loss = losses['sn']
        scene_loss = losses['sc']
        self_loss = losses['self']

        x = torch.autograd.Variable(rgb).cuda()
        y1 = torch.autograd.Variable(ss).cuda()
        y2 = torch.autograd.Variable(de).cuda()
        y3 = torch.autograd.Variable(sn).cuda()
        y4 = torch.autograd.Variable(msk).cuda()

        # multi-task learning
        y1_pred, y2_pred, y3_pred = self.multi_model(x)

        loss1 = ss_loss(y1_pred, y1, torch.ones_like(y1).cuda())
        loss2 = de_loss(y2_pred.squeeze(), y2, torch.ones_like(y2).cuda())
        loss3 = sn_loss(y3_pred, y3, y4)

        multi_task_loss = loss1 + loss2 + loss3

        self.multi_opt.zero_grad()
        multi_task_loss.backward()
        self.multi_opt.step()

        # refinement network learning
        y1_pred, y2_pred, y3_pred = self.multi_model(x)
        cls_pred = self.post_model(y1_pred,y2_pred,y3_pred)
        loss4 = scene_loss(cls_pred, sc)
        self.encoder_opt.zero_grad()
        self.post_opt.zero_grad()
        loss4.backward()
        self.post_opt.step()
        self.encoder_opt.step()

        # self-supervised learning
        loss5 = self.train_CLR(rgb,self_loss)

        returned_losses = [loss1, loss2, loss3, loss4]
        return returned_losses

    def train_unlabel_step(self, sample, losses):
        rgb = sample['rgb'].cuda()
        sc = sample['sc'].cuda().long()
        bs = rgb.shape[0]


        scene_loss = losses['sc']
        self_loss = losses['self']

        x = torch.autograd.Variable(rgb).cuda()

        # multi-task learning
        y1_pred, y2_pred, y3_pred = self.multi_model(x)

        cls_pred = self.post_model(y1_pred,y2_pred,y3_pred)
        loss6 = scene_loss(cls_pred, sc)
        self.encoder_opt.zero_grad()
        loss6.backward()
        self.encoder_opt.step()

        loss7 = self.train_CLR(rgb,self_loss)

        returned_losses = [loss6,loss7]
        return returned_losses




    

        
        

    
    