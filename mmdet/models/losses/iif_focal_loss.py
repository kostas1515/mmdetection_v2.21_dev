import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..builder import LOSSES
from .utils import weight_reduce_loss
from .accuracy import accuracy
from ..builder import LOSSES
from .utils import weight_reduce_loss


@LOSSES.register_module()
class IIFFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 use_custom_activation=True,
                 reduction='mean',
                 loss_weight=1.0,
                 gamma=2.0,
                 alpha=0.25,
                 path='./lvis_files/idf_1204.csv',
                 **kwargs):
        """Gumbel_FocalLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(IIFFocalLoss, self).__init__()

        self.use_sigmoid = use_sigmoid
        self.use_custom_activation = use_custom_activation
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        
        self.custom_activation = True
        self.custom_accuracy = True
        
        if variant =='log_adj':
            self.iif_weights =pd.read_csv(path)['raw'].values.tolist()
            self.log_adj=True
        else:
            self.iif_weights =pd.read_csv(path)[variant].values.tolist()
            self.log_adj=False
            
        self.iif_weights = self.iif_weights[1:] #+1 for bg
        self.iif_weights = torch.tensor(self.iif_weights,device='cuda',dtype=torch.float).unsqueeze(0)
        
    def get_activation(self, cls_score):
        
        scores = torch.sigmoid(cls_score*self.iif_weights)
        
        return scores
    
    def get_accuracy(self, cls_score, labels):
        """Get custom accuracy w.r.t. cls_score and labels.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C).
            labels (torch.Tensor): The learning label of the prediction.

        Returns:
            Dict [str, torch.Tensor]: The accuracy for objectness and classes,
                 respectively.
        """
        acc_classes = accuracy(cls_score, labels)
        acc = dict()
        acc['acc_classes'] = acc_classes
        
        return acc
    
    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            num_classes = pred.size(1)
            target = F.one_hot(target, num_classes=num_classes + 1)
            target = target[:, :num_classes]
            target = target.type_as(pred)
            
            pestim = torch.sigmoid(pred*self.iif_weights)

            pt = (1 - pestim) * target + pestim * (1 - target)
            focal_weight = (self.alpha * target + (1 - self.alpha) *
                            (1 - target)) * pt.pow(self.gamma)
            
            loss = F.binary_cross_entropy(pestim, target,reduction='none')*focal_weight
#             print(loss)
            
#             loss_cls=loss.sum()/(target.sum()+1)
            if weight is not None:
                if weight.shape != loss.shape:
                    if weight.size(0) == loss.size(0):
                        # For most cases, weight is of shape (num_priors, ),
                        #  which means it does not have the second axis num_class
                        weight = weight.view(-1, 1)
                    else:
                        # Sometimes, weight per anchor per class is also needed. e.g.
                        #  in FSAF. But it may be flattened of shape
                        #  (num_priors x num_class, ), while loss is still of shape
                        #  (num_priors, num_class).
                        assert weight.numel() == loss.numel()
                        weight = weight.view(loss.size(0), -1)
                assert weight.ndim == loss.ndim
            loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
            
#             loss_cls = weight_reduce_loss(loss,reduction=reduction,avg_factor=avg_factor)

        else:
            raise NotImplementedError
            
        return loss
        


    