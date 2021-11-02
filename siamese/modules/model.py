import torch
import torchmetrics 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from . import backbone
from siamese.trainer.loss import ContrastiveLoss
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torchmetrics 
import torch.optim as optim

class SiameseTask(pl.LightningModule):
    def __init__(self, feature_extractor: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.criterion = ContrastiveLoss()

        self.trn_loss: torchmetrics.AverageMeter = torchmetrics.AverageMeter()
        self.val_loss: torchmetrics.AverageMeter = torchmetrics.AverageMeter()


    def _forward_once(self, x: torch.Tensor):
        return self.feature_extractor(x)
    
    def _forward(self, x: torch.Tensor, y:torch.Tensor):
        # x,y = data
        main = self._forward_once(x)
        comp = self._forward_once(y)
        
        return main, comp
    
    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        
    def shared_step(self, batch, batch_idx):
        main_imgs, comp_imgs, labels = batch
        main_feature, comp_feature = self._forward(main_imgs, comp_imgs)
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
        return optim.RMSprop(self.parameters(), lr=0.0001, alpha=0.99, eps=1e-8, weight_decay=0.00005, momentum=0.9)
     
def siamese_net(pretrained=True, backbone_name="mobilenetv2", encoder_digit=64, **kwargs):
    if backbone_name.startswith("resnet"):
        version = int(backbone_name.split('resnet')[-1])
        backbone_model = backbone.resnet_backbone(pretrained_backbone=pretrained, 
                                                  encoder_digit=encoder_digit, 
                                                  version=version, **kwargs)
    elif backbone_name=="signet":
        backbone_model = backbone.SigNetBackbone()
        
    else:
        backbone_model = backbone.mobilenetv2_backbone(pretrained_backbone=pretrained,
                                                       encoder_digit=encoder_digit, **kwargs)
        
    siamese_net = SiameseTask(feature_extractor=backbone_model)
    return siamese_net

if __name__=="__main__":
    module = siamese_net(backbone_name='signet')
