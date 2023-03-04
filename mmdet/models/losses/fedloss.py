import torch
import torch.nn as nn
import torch.nn.functional as F
from .accuracy import accuracy
from ..builder import LOSSES
import pandas as pd
import numpy as np

       
def get_frequent_indices(lvis_file,threshold=100):
    lvis = pd.read_csv(lvis_file)
    frequent_mask = lvis['img_freq'].values[1:]>threshold
    return torch.tensor(frequent_mask,device='cuda')


def load_class_freq(lvis_file, freq_weight=0.5):
    lvis = pd.read_csv(lvis_file)
    cat_info = torch.tensor(lvis['img_freq'].values[1:],device='cuda')
    freq_weight = cat_info.float() ** freq_weight
    return freq_weight


def get_fed_loss_inds(
    gt_classes, num_sample_cats=50, C=1203, \
    weight=None, fed_cls_inds=-1):
    appeared = torch.unique(gt_classes) # C'
    prob = appeared.new_ones(C + 1).float()
    prob[-1] = 0
    if len(appeared) < num_sample_cats:
        if weight is not None:
            prob[:C] = weight.float().clone()
        prob[appeared] = 0
        if fed_cls_inds > 0:
            prob[fed_cls_inds:] = 0
        more_appeared = torch.multinomial(
            prob, num_sample_cats - len(appeared),
            replacement=False)
        appeared = torch.cat([appeared, more_appeared])
    return appeared


@LOSSES.register_module()
class FedLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 use_classif='gumbel',
                 num_classes=1203,
                 fed_loss_num_cat=50,
                 freq_weight=0.5,
                 lvis_files='./lvis_files/idf_1204.csv'):
        super(FedLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        
        self.fed_loss_num_cat=fed_loss_num_cat
        self.freq_weight=load_class_freq(lvis_files,freq_weight)
        
        self.use_classif=use_classif
        self.num_classes=num_classes
        
        self.custom_cls_channels = True
        self.custom_activation = True
        self.custom_accuracy = True
        
        if self.use_classif == 'cls_rel':
            self.frequent_classes=get_frequent_indices(lvis_files)
        
    def get_cls_channels(self, num_classes):
        """Get custom classification channels.

        Args:
            num_classes (int): The number of classes.

        Returns:
            int: The custom classification channels.
        """
        assert num_classes == self.num_classes
        return num_classes
    
    def get_accuracy(self, cls_score, labels):
        
        pos_inds = labels < self.num_classes
        acc_classes = accuracy(cls_score[pos_inds], labels[pos_inds])
        acc = dict()
        acc['acc_classes'] = acc_classes
        return acc
    
    def get_activation(self, cls_score):
        """Get custom activation of cls_score.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C).

        Returns:
            torch.Tensor: The custom activation of cls_score with shape
                 (N, C).
        """
        if self.use_classif=='gumbel':
            scores = 1/(torch.exp(torch.exp(-cls_score)))
        elif self.use_classif=='normal':
            scores=1/2+torch.erf(cls_score/(2**(1/2)))/2
        elif self.use_classif=='cls_rel':
            scores=self.class_relative_act(cls_score,inference=True)
        else:
            scores=torch.sigmoid(cls_score)
        
        dummpy_prob = scores.new_zeros((scores.size(0), 1))
        scores = torch.cat([scores, dummpy_prob], dim=1)
        
        return scores
    
    def class_relative_act(self,pred,inference=False):
        frequent_classes = self.frequent_classes
        freq_mask = torch.zeros_like(pred)
        freq_mask[:,frequent_classes] = 1.0
        
        rc_mask= torch.zeros_like(pred)
        rc_mask[:,~frequent_classes] = 1.0

        if inference is False:
            normits = torch.clamp(pred,min=-5.0,max=5.0)
            gombits = torch.clamp(pred,min=-4.0,max=10.0)
            pred_normal= 1/2+torch.erf(normits/(2**(1/2)))/2
            pred_gumbel= 1/(torch.exp(torch.exp(-gombits)))
        else:
            pred_normal= 1/2+torch.erf(pred/(2**(1/2)))/2
            pred_gumbel= 1/(torch.exp(torch.exp(-pred)))

        pestim = pred_gumbel*rc_mask + pred_normal*freq_mask
        

        return pestim

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        
        self.n_i, self.n_c = cls_score.size()

        self.pred_class_logits = cls_score

        if self.use_classif =='normal':
            cls_score=torch.clamp(cls_score,min=-5,max=5)
            pestim=1/2+torch.erf(cls_score/(2**(1/2)))/2
            cls_loss = self.cross_entropy_loss(pestim,label,with_logits=False)
        elif self.use_classif =='gumbel':
            cls_score=torch.clamp(cls_score,min=-4,max=10)
            pestim= 1/(torch.exp(torch.exp(-(cls_score))))
            cls_loss = self.cross_entropy_loss(pestim,label,with_logits=False)
        elif self.use_classif =='cls_rel':
            pestim=self.class_relative_act(cls_score)
            cls_loss = self.cross_entropy_loss(pestim,label,with_logits=False)
        else:
            cls_loss = self.cross_entropy_loss(cls_score,label)


        return self.loss_weight * cls_loss
    
    
    def cross_entropy_loss(self, pred_class_logits, gt_classes,with_logits=True):
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0] # This is more robust than .sum() * 0.

        B = pred_class_logits.shape[0]
        C = pred_class_logits.shape[1] 

        target = pred_class_logits.new_zeros(B, C + 1)
        target[range(len(gt_classes)), gt_classes] = 1 # B x (C + 1)
        target = target[:, :C] # B x C

        weight = 1
        
        appeared = get_fed_loss_inds(
            gt_classes, 
            num_sample_cats=self.fed_loss_num_cat,
            C=C,
            weight=self.freq_weight)
        appeared_mask = appeared.new_zeros(C + 1)
        appeared_mask[appeared] = 1 # C + 1
        appeared_mask = appeared_mask[:C]
        fed_w = appeared_mask.view(1, C).expand(B, C)
        weight = weight * fed_w.float()
        
        if with_logits is True:
            cls_loss = F.binary_cross_entropy_with_logits(
                pred_class_logits, target, reduction='none') # B x C
        else:
            cls_loss = F.binary_cross_entropy(
                pred_class_logits, target, reduction='none') # B x C
        loss =  torch.sum(cls_loss * weight) / B
        
        return loss
