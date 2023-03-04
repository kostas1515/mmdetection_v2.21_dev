import torch
import torch.nn as nn
import torch.nn.functional as F
from .accuracy import accuracy
from ..builder import LOSSES
import pandas as pd


def get_image_count_frequency(version="v0_5"):
    if version == "v0_5":
        from mmdet.utils.lvis_v0_5_categories import get_image_count_frequency
        return get_image_count_frequency()
    elif version == "v1":
        from mmdet.utils.lvis_v1_0_categories import get_image_count_frequency
        return get_image_count_frequency()
    elif version == "openimage":
        from mmdet.utils.openimage_categories import get_instance_count
        return get_instance_count()
    else:
        raise KeyError(f"version {version} is not supported")
        
def get_frequent_indices(lvis_file,threshold=100):
    lvis = pd.read_csv(lvis_file)
    frequent_mask = lvis['img_freq'].values[1:]>threshold
    return torch.tensor(frequent_mask,device='cuda')

def get_weights(lvis_file):
    lvis = pd.read_csv(lvis_file)
    weights = lvis['base10_obj'].values[1:] # remove bg entry
    return torch.tensor(weights,device='cuda',dtype=torch.float).unsqueeze(0)

@LOSSES.register_module()
class DropLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 lambda_=0.00177,
                 version="v0_5",
                 use_classif='gumbel',
                 num_classes=1203,
                 use_iif=False,
                 lvis_files='./lvis_files/idf_1204.csv'):
        super(DropLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.lambda_ = lambda_
        self.version = version
        self.freq_info = torch.FloatTensor(get_image_count_frequency(version))
        self.use_classif=use_classif
        self.use_iif=use_iif
        self.num_classes=num_classes
        self.custom_cls_channels = True
        self.custom_activation = True
        self.custom_accuracy = True
        
        if self.use_classif == 'cls_rel':
            self.frequent_classes=get_frequent_indices(lvis_files)
            
        if self.use_iif is True:
            self.iif_weights=get_weights(lvis_files)

        num_class_included = torch.sum(self.freq_info < self.lambda_)
        print(f"set up DropLoss (version {version}), {num_class_included} classes included.")
        
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
        
        if self.use_iif is True:
            cls_score=self.iif_weights*cls_score.clone()
        
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
        
        if self.use_iif is True:
            cls_score=self.iif_weights*cls_score.clone()
        
        
        self.n_i, self.n_c = cls_score.size()

        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c + 1)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target[:, :self.n_c]

        target = expand_label(cls_score, label)
        drop_w = 1 - self.threshold_func() * (1 - target)
        
        if self.use_classif =='normal':
            cls_score=torch.clamp(cls_score,min=-5,max=5)
            pestim=1/2+torch.erf(cls_score/(2**(1/2)))/2
            cls_loss = F.binary_cross_entropy(pestim, target,reduction='none')
        elif self.use_classif =='gumbel':
            cls_score=torch.clamp(cls_score,min=-4,max=10)
            pestim= 1/(torch.exp(torch.exp(-(cls_score))))
            cls_loss = F.binary_cross_entropy(pestim, target,reduction='none')
        elif self.use_classif =='cls_rel':
            pestim=self.class_relative_act(cls_score)
            cls_loss = F.binary_cross_entropy(pestim, target,reduction='none')
        else:
            cls_loss = F.binary_cross_entropy_with_logits(cls_score, target,reduction='none')


        cls_loss = torch.sum(cls_loss * drop_w) / self.n_i

        return self.loss_weight * cls_loss
    

    def exclude_func_and_ratio(self):
        
        # instance-level weight
        bg_ind = self.n_c
        weight = (self.gt_classes != bg_ind)

        gt_classes    = self.gt_classes[weight]
        exclude_ratio = torch.mean((self.freq_info[gt_classes] < self.lambda_).float())

        weight = weight.float().view(self.n_i, 1).expand(self.n_i, self.n_c)

        return weight, exclude_ratio

    def threshold_func(self):
        # class-level weight
        weight = self.pred_class_logits.new_zeros(self.n_c)
        weight[self.freq_info < self.lambda_] = 1
        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)


        fg, ratio = self.exclude_func_and_ratio()
        bg = 1 - fg
        random = torch.rand_like(bg) * bg

        random = torch.where(random>ratio, torch.ones_like(random), torch.zeros_like(random))
        
        weight = (random + fg) * weight

        return weight
    
    def get_multi_activation(self, cls_score):
        """Get custom activation of cls_score.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C).

        Returns:
            torch.Tensor: The custom activation of cls_score with shape
                 (N, C).
        """
        probability_weights = torch.softmax(cls_score[:,-3:],dim=-1)

        scores_g = 1/(torch.exp(torch.exp(-cls_score[:,:-3])))
        scores_s = torch.sigmoid(cls_score[:,:-3])
        scores_n=1/2+torch.erf(cls_score[:,:-3]/(2**(1/2)))/2
        scores= scores_g*probability_weights[:,0:1]+scores_s*probability_weights[:,1:2]+scores_n*probability_weights[:,2:]
            
        
        return scores