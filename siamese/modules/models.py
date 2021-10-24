import torch
import torchmetrics 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from . import backbone

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torchmetrics 
import torch.optim as optim

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        if not torch.cuda.is_available():
            euclidean_distance = euclidean_distance.cpu()
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

class SiameseNet(pl.LightningModule):
    def __init__(self, feature_extractor: nn.Module):
        super(SiameseNet, self).__init__()
        self.feature_extractor = feature_extractor
        
        self.criterion = ContrastiveLoss()

        self.trn_loss: torchmetrics.AverageMeter = torchmetrics.AverageMeter()
        self.val_loss: torchmetrics.AverageMeter = torchmetrics.AverageMeter()


        
    def _forward_once(self, x: torch.Tensor):
        return self.feature_extractor(x)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        main = self._forward_once(x)
        comp = self._forward_once(y)
        return main, comp
    
    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        
    def shared_step(self, batch, batch_idx):
        main_imgs, comp_imgs, labels= batch
        
        main_feature, comp_feature = self.forward(main_imgs, comp_imgs)
        loss = self.criterion(main_feature, comp_feature, labels)
        
        return loss
        
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        trn_loss = self.trn_loss(loss)
        self.log('trn_step_loss', trn_loss, prog_bar=True, logger=True)
        return loss
    
    def training_epoch_end(self, outs):
        self.log('trn_epoch_loss', self.trn_loss.compute(), logger=True)
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        val_loss = self.val_loss(loss)
        self.log('val_step_loss', val_loss, prog_bar=True, logger=True)
        return loss
    
    def validation_epoch_end(self, outs):
        self.log('val_epoch_loss', self.val_loss.compute(), logger=True)
        
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0005)
    
def siamese_net(pretrained=True, backbone_name="mobilenetv2", encoder_digit=64, **kwargs):
    if backbone_name.startswith("resnet"):
        backbone_model = backbone.resnet_backbone(pretrained_backbone=pretrained, 
                                                  encoder_digit=encoder_digit, **kwargs)
    else:
        backbone_model = backbone.mobilenetv2_backbone(pretrained_backbone=pretrained,
                                                       encoder_digit=encoder_digit, **kwargs)
        
    siamese_net = SiameseNet(feature_extractor=backbone_model)
    return siamese_net