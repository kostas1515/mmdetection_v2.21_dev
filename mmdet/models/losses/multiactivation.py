import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..builder import LOSSES
from .utils import weight_reduce_loss
from .accuracy import accuracy

def get_image_count_frequency(version="v0_5"):
    if version == "v0_5":
        from mmdet.utils.lvis_v0_5_categories import get_image_count_frequency
        return get_image_count_frequency()
    elif version == "v1":
        from mmdet.utils.lvis_v1_0_categories import get_image_count_frequency
        return get_image_count_frequency()
    else:
        raise KeyError(f"version {version} is not supported")

@LOSSES.register_module()
class MultiActivation(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=1203,
                 loss_cls='bce',
                 lambda_=0.0011,
                 version="v1",
                 class_heads=2,
                 with_kd=False,
                 with_obj=False):
        """CrossEntropyLoss.

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
        super(MultiActivation, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.lambda_ = lambda_
        self.version = version
        self.freq_info = torch.FloatTensor(get_image_count_frequency(version)).cuda()
        
        # custom output channels of the classifier
        self.custom_cls_channels = True
        # custom activation of cls_score
        self.custom_activation = True
        # custom accuracy of the classsifier
        self.custom_accuracy = True

        self.class_heads = class_heads
        self.with_obj = with_obj
        self.with_kd=with_kd
        

        if loss_cls=='bce':
            self.cls_criterion = self.binary_cross_entropy
        elif loss_cls == 'focal':
            self.cls_criterion = self.focal_loss
        elif loss_cls == 'droploss':
            self.cls_criterion = self.droploss

    def get_objectness(self, cls_score):
        #predicts whether box is bg, aka 1 -> bg, 0 -> fg
        if self.class_heads==3:
            return cls_score[:,-4:-3].sigmoid()
        else:
            return cls_score[:,-2:-1].sigmoid()
            
    
    def get_activation(self, cls_score):
        """Get custom activation of cls_score.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, 3*C).

        Returns:
            torch.Tensor: The custom activation of cls_score with shape
                 (N, C).
        """
        scores = self.get_multi_act(cls_score)
        if self.with_obj is True:
            scores = scores * (1.0-self.get_objectness(cls_score))

        dummpy_prob = scores.new_zeros((scores.size(0), 1))
        scores = torch.cat([scores, dummpy_prob], dim=1)
        
        return scores

    def get_multi_act(self,pred):
        pestim_g = 1/(torch.exp(torch.exp(-(torch.clamp(pred[:,:self.num_classes],min=-4,max=10)))))
#         pestim_n=1/2+torch.erf(torch.clamp(pred[:,self.num_classes:2*self.num_classes],min=-5,max=6)/(2**(1/2)))/2
        pestim_n=(torch.clamp(pred[:,self.num_classes:2*self.num_classes],min=-12.0,max=12.0)).sigmoid()
        
        if self.class_heads==3:
            pestim_l=torch.sigmoid(pred[:,2*self.num_classes:3*self.num_classes])
            probs = torch.softmax(pred[:,-3:],dim=-1)
        else:
            probs = pred[:,-1:].sigmoid()
        if self.class_heads == 3:
            p_final = (probs[:,0:1]*pestim_g+probs[:,1:2]*pestim_n+probs[:,2:]*pestim_l)
        else:
            p_final = probs*pestim_g+(1-probs)*pestim_n
        return p_final
    
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
        """Get custom accuracy w.r.t. cls_score and labels.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C).
            labels (torch.Tensor): The learning label of the prediction.

        Returns:
            Dict [str, torch.Tensor]: The accuracy for objectness and classes,
                 respectively.
        """
        pos_inds = labels<self.num_classes
        scores = self.get_multi_act(cls_score)
        acc_classes = accuracy(scores[pos_inds], labels[pos_inds])
        acc = dict()
        acc['acc_classes'] = acc_classes
        if self.with_obj is True:
            obj_labels = (labels == self.num_classes).long()
            cls_score_objectness = self.get_objectness(cls_score)
            acc_objectness = accuracy(cls_score_objectness, obj_labels)
            acc['acc_objectness'] = acc_objectness
        return acc

    

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        total_loss=dict()
        loss_cls = self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        total_loss['loss_cls']=loss_cls
        if self.with_kd is True:
            total_loss['kd_loss'] = self.loss_weight * self.kd_loss(
            cls_score,
            **kwargs)
        
        if self.with_obj is True:
            total_loss['obj_loss'] = self.loss_weight * self.objectness_loss(
            cls_score,label,
            **kwargs)
            
        return total_loss


    def binary_cross_entropy(self,
                         pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=-100):
        """Calculate the binary CrossEntropy loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, 1).
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            class_weight (list[float], optional): The weight for each class.
            ignore_index (int | None): The label index to be ignored.
                If None, it will be set to default value. Default: -100.

        Returns:
            torch.Tensor: The calculated loss.
        """
        # The default value of ignore_index is the same as F.cross_entropy

        self.n_i, _ = pred.size()

        def expand_label(pred, gt_classes):
            n_i = pred.size()[0]
            target = pred.new_zeros(n_i, self.num_classes + 1)
            target[torch.arange(n_i), gt_classes] = 1
            return target[:, :self.num_classes]

        if weight is not None:
            weight = weight.float()

        target = expand_label(pred, label)

        p_final = self.get_multi_act(pred)
        
#         loss = -(target.float()*torch.log(p_final)+(1.0-target.float())*torch.log(1-p_final))
        loss = F.binary_cross_entropy(
            p_final, target.float(), reduction='none')
        
        # do the reduction for the weighted loss
        loss[torch.isnan(loss)]=0.0
        loss[torch.isinf(loss)]=0.0
        
        loss = loss.sum()/self.n_i
        # loss=torch.clamp(loss,min=0,max=30)
        
        return loss


    def focal_loss(self,
                    pred,
                    label,
                    weight=None,
                    alpha=0.25,
                    gamma=2,
                    reduction='mean',
                    avg_factor=None,
                    class_weight=None,
                    ignore_index=-100):
        """Calculate the binary CrossEntropy loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, 1).
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            class_weight (list[float], optional): The weight for each class.
            ignore_index (int | None): The label index to be ignored.
                If None, it will be set to default value. Default: -100.

        Returns:
            torch.Tensor: The calculated loss.
        """
        # The default value of ignore_index is the same as F.cross_entropy

        self.n_i, _ = pred.size()

        def expand_label(pred, gt_classes):
            n_i = pred.size()[0]
            target = pred.new_zeros(n_i, self.num_classes + 1)
            target[torch.arange(n_i), gt_classes] = 1
            return target[:, :self.num_classes]

        if weight is not None:
            weight = weight.float()


        target = expand_label(pred, label)
        
        p_final = self.get_multi_act(pred)

        pt = (1 - p_final) * target + p_final * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) *
                        (1 - target)) * pt.pow(gamma)

        loss = F.binary_cross_entropy(
            p_final, target.float(), reduction='none')*focal_weight
        
        # do the reduction for the weighted loss
        loss[torch.isnan(loss)]=0.0
        loss = loss.sum()/self.n_i
        # loss=torch.clamp(loss,min=0,max=30)
        
        return loss

    def kd_loss(self,
                pred,
                tau=2.0,
                gamma=2.0):
        """Calculate the binary CrossEntropy loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, 1).
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            class_weight (list[float], optional): The weight for each class.
            ignore_index (int | None): The label index to be ignored.
                If None, it will be set to default value. Default: -100.

        Returns:
            torch.Tensor: The calculated loss.
        """
        # The default value of ignore_index is the same as F.cross_entropy

        self.n_i, _ = pred.size()
        pestim_g = 1/(torch.exp(torch.exp(-(torch.clamp(pred[:,0:self.num_classes],min=-4,max=10)))))
        pestim_n=1/2+torch.erf(torch.clamp(pred[:,self.num_classes:2*self.num_classes],min=-5,max=6)/(2**(1/2)))/2

        soft_pestim_g = pestim_g/tau
        soft_pestim_n = pestim_n/tau

        focal_weight = torch.abs(soft_pestim_g-soft_pestim_n).pow(gamma)

        kd_loss1 = F.binary_cross_entropy(soft_pestim_g,soft_pestim_n,reduction='none')
        kd_loss2 = F.binary_cross_entropy(soft_pestim_n,soft_pestim_g,reduction='none')
        kd_loss = ((kd_loss1+kd_loss2)*focal_weight)/2
        kd_loss[torch.isnan(kd_loss)]=0.0
        kd_loss = kd_loss.sum()

        # print(kd_loss)

        return kd_loss

    
    def droploss(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        
        self.n_i, _ = cls_score.size()
        
        self.gt_classes = label
        p_final = self.get_multi_act(cls_score)
        self.pred_class_logits = p_final

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.num_classes + 1)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target[:, :self.num_classes]

        target = expand_label(p_final, label)
        drop_w = 1 - self.threshold_func() * (1 - target)
        
#         cls_loss = -(target.float()*torch.log(p_final)+(1.0-target.float())*torch.log(1-p_final))
        cls_loss = F.binary_cross_entropy(p_final, target,reduction='none')
        cls_loss = torch.sum(cls_loss * drop_w) / self.n_i

        return cls_loss
    
    def objectness_loss(self,
                        cls_score,
                        labels):
        
        n_i, _ = cls_score.size()
        
        obj_labels = (labels == self.num_classes).float().unsqueeze(1)
        p_final = self.get_objectness(cls_score)
        loss_cls_objectness = F.binary_cross_entropy(p_final, obj_labels,reduction='none')
        loss = loss_cls_objectness.sum()/n_i
        
        return loss

    def exclude_func_and_ratio(self):
        
        # instance-level weight
        bg_ind = self.num_classes
        weight = (self.gt_classes != bg_ind)

        gt_classes    = self.gt_classes[weight]
        exclude_ratio = torch.mean((self.freq_info[gt_classes] < self.lambda_).float())

        weight = weight.float().view(self.n_i, 1).expand(self.n_i, self.num_classes)

        return weight, exclude_ratio

    def threshold_func(self):
        # class-level weight
        weight = self.pred_class_logits.new_zeros(self.num_classes)
        weight[self.freq_info < self.lambda_] = 1
        weight = weight.view(1, self.num_classes).expand(self.n_i, self.num_classes)


        fg, ratio = self.exclude_func_and_ratio()
        bg = 1 - fg
        random = torch.rand_like(bg) * bg

        random = torch.where(random>ratio, torch.ones_like(random), torch.zeros_like(random))
        
        weight = (random + fg) * weight

        return weight
    