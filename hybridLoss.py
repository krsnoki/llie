

'''
************** kalyani kolte: Combined loss for quality measure ************
'''
import torch
import torch.nn as nn
import torch.functional as F


from torchvision import models

class VGGperceptualLoss(nn.Module):
    def __init__(self, layer_ids = [2, 7, 14, 21]):
        super(VGGperceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.layers = nn.ModuleList([vgg[i] for i in layer_ids]).eval()
        for params in self.parameters():
            params.requires_grad = False # no grads >> 

    def forward(self, x, y):
        loss = 0
        for layer in self.layers:
            x, y = layer(x), layer(y)
            loss += F.l1_loss(x, y)
        return loss
    
class HybridLoss(nn.Module):
    def __init__(self,lambda_l1 = 1.0, lambda_perceptual = 0.1, lambda_gan= 0.01 ):
        super(HybridLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss= VGGperceptualLoss()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lamda_gan = lambda_gan

    def forward(self, pred, target, pred_fake, is_real=True):
        #combining l1, perceptual and adversial loss
        l1 = self.l1_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        adversial = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake) if is_real else torch.zeros_like(pred_fake))

        # total loss = l1 + perceptual + adversial 
        total_loss = (self.lambda_l1 * l1) + (self.lambda_perceptual * perceptual) + (self.lambda_gan * adversial)

        return total_loss, {'L1': l1.item()}, {'Perceptual': perceptual.item()}, {'Adversial': adversial.item()}