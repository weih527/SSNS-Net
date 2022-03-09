from __future__ import print_function, division
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch
import numpy as np

class DiceLoss(_Loss):
    def __init__(self, size_average=True, reduce=True, smooth=100.0):
        super(DiceLoss, self).__init__(size_average, reduce)
        self.smooth = smooth
        self.reduce = reduce

    def dice_loss(self, input_y, target):
        loss = 0.

        for index in range(input_y.size()[0]):
            iflat = input_y[index].view(-1)
            tflat = target[index].view(-1)
            intersection = (iflat * tflat).sum()

            loss += 1 - ((2. * intersection + self.smooth) /
                         ((iflat**2).sum() + (tflat**2).sum() + self.smooth))

        # size_average=True for the dice loss
        return loss / float(input_y.size()[0])

    def dice_loss_batch(self, input_y, target):
        iflat = input_y.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        loss = 1 - ((2. * intersection + self.smooth) /
                    ((iflat**2).sum() + (tflat**2).sum() + self.smooth))
        return loss

    def forward(self, input_y, target):
        assert target.requires_grad is False
        if not (target.size() == input_y.size()):
            raise ValueError(
                "Target size ({}) must be the same as input size ({})".format(target.size(), input_y.size()))

        if self.reduce:
            loss = self.dice_loss(input_y, target)
        else:
            loss = self.dice_loss_batch(input_y, target)
        return loss


class WeightedMSELoss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(WeightedMSELoss, self).__init__(size_average, reduce)

    @staticmethod
    def weighted_mse_loss(input_y, target, weight):
        s1 = torch.prod(torch.tensor(input_y.size()[2:]).float())
        s2 = input_y.size()[0]
        norm_term = (s1 * s2).cuda()
        return torch.sum(weight * (input_y - target) ** 2) / norm_term

    def forward(self, input_y, target, weight):
        assert target.requires_grad is False
        return self.weighted_mse_loss(input_y, target, weight)


class MSELoss(_Loss):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, input_y, target, weight):
        assert target.requires_grad is False
        return F.mse_loss(input_y, target)


class BCELoss(_Loss):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, input_y, target, weight):
        assert target.requires_grad is False
        weight = torch.ones_like(target)
        return F.binary_cross_entropy(input_y, target, weight)


# define a customized loss function for future development
class WeightedBCELoss(_Loss):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()

    def forward(self, input_y, target, weight):
        assert target.requires_grad is False
        return F.binary_cross_entropy(input_y, target, weight)


class WeightedBCELoss_cbst(_Loss):
    def __init__(self):
        super(WeightedBCELoss_cbst, self).__init__()
    
    def binary_cross_entropy(self, input_y, target, weight):
        input_y = torch.clamp(input_y, min=0.000001, max=0.999999)
        loss = -weight * (target * torch.log(input_y) + (1 - target) * torch.log(1 - input_y))
        loss = torch.sum(loss) / torch.sum(weight != 0).float()  ### change
        return loss

    def forward(self, input_y, target, weight):
        assert target.requires_grad is False
        loss = self.binary_cross_entropy(input_y, target, weight)
        return loss


class WeightedBCELoss_wo_label(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(WeightedBCELoss_wo_label, self).__init__(size_average, reduce)
        self.size_average = size_average
        self.reduce = reduce
    
    def binary_cross_entropy(self, input_y, target, weight):
        loss = -weight * (target * torch.log(input_y) + (1 - target) * torch.log(1 - input_y))
        loss = torch.sum(loss) / torch.numel(target)  ### change
        return loss
    
    def gen_weight_mask(self, target, thresd):
        assert thresd >= 0.0 and thresd <= 1.0
        target.requires_grad = False
        up_thresd = thresd
        down_thresd = 1.0 - thresd
        mask = torch.zeros_like(target)
        target[target > up_thresd] = 1
        mask[target == 1] = 1
        num_1 = torch.sum(mask)
        target[target < down_thresd] = 0
        mask[target == 0] = 1
        num = torch.sum(mask)
        weight_factor = num_1 / num
        mask[target == 1] *= (1 - weight_factor) / weight_factor
        # mask[target == 0] *= weight_factor
        return target, mask, weight_factor

    def weighted_bce_loss_wo_label(self, input_y, thresd):
        target, weight, _ = self.gen_weight_mask(input_y.clone().detach(), thresd)
        loss = F.binary_cross_entropy(input_y, target, weight, self.size_average, self.reduce)
        # loss = self.binary_cross_entropy(input_y, target, weight)
        return loss, target, weight

    def weighted_bce_loss_mix(self, input_y, target, weight, thresd):
        # split
        batch_size = target.size(0)
        for k in range(batch_size):
            if torch.max(target[k:k+1, ...]) == 0:
                target[k:k+1, ...], weight[k:k+1, ...], _ = self.gen_weight_mask(input_y[k:k+1, ...].clone().detach(), thresd)
        loss = F.binary_cross_entropy(input_y, target, weight, self.size_average, self.reduce)
        return loss, target, weight

    def forward(self, input_y, target, weight, mode=0, thresd=1.0):
        # assert target.requires_grad is False
        if mode == 0:
            loss = F.binary_cross_entropy(input_y, target, weight, self.size_average, self.reduce)
            return loss, target, weight
        elif mode == 1:
            loss, target, weight = self.weighted_bce_loss_wo_label(input_y, thresd)
            return loss, target, weight
        else:
            loss, target, weight = self.weighted_bce_loss_mix(input_y, target, weight, thresd)
            return loss, target, weight


class WeightedBCELoss_wo_label_onlylabeled(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(WeightedBCELoss_wo_label_onlylabeled, self).__init__(size_average, reduce)
        self.size_average = size_average
        self.reduce = reduce
    
    def binary_cross_entropy(self, input_y, target, weight):
        loss = -weight * (target * torch.log(input_y) + (1 - target) * torch.log(1 - input_y))
        loss = torch.sum(loss) / torch.numel(target)  ### change
        return loss
    
    def gen_weight_mask(self, target, thresd):
        assert thresd >= 0.0 and thresd <= 1.0
        target.requires_grad = False
        up_thresd = thresd
        down_thresd = 1.0 - thresd
        mask = torch.zeros_like(target)
        target[target > up_thresd] = 1
        mask[target == 1] = 1
        num_1 = torch.sum(mask)
        target[target < down_thresd] = 0
        mask[target == 0] = 1
        num = torch.sum(mask)
        weight_factor = num_1 / num
        mask[target == 1] *= (1 - weight_factor) / weight_factor
        # mask[target == 0] *= weight_factor
        return target, mask, weight_factor

    def weighted_bce_loss_wo_label(self, input_y, thresd):
        target, weight, _ = self.gen_weight_mask(input_y.clone().detach(), thresd)
        loss = F.binary_cross_entropy(input_y, target, weight, self.size_average, self.reduce)
        # loss = self.binary_cross_entropy(input_y, target, weight)
        return loss, target, weight

    def weighted_bce_loss_mix(self, input_y, target, weight, thresd):
        # split
        # batch_size = target.size(0)
        # for k in range(batch_size):
        #     if torch.max(target[k:k+1, ...]) == 0:
        #         target[k:k+1, ...], weight[k:k+1, ...], _ = self.gen_weight_mask(input_y[k:k+1, ...].clone().detach(), thresd)
        loss = F.binary_cross_entropy(input_y, target, weight, self.size_average, self.reduce)
        return loss, target, weight

    def forward(self, input_y, target, weight, mode=0, thresd=1.0):
        # assert target.requires_grad is False
        if mode == 0:
            loss = F.binary_cross_entropy(input_y, target, weight, self.size_average, self.reduce)
            return loss, target, weight
        elif mode == 1:
            loss, target, weight = self.weighted_bce_loss_wo_label(input_y, thresd)
            return loss, target, weight
        else:
            loss, target, weight = self.weighted_bce_loss_mix(input_y, target, weight, thresd)
            return loss, target, weight

class TripletMarginLoss(_Loss):
    def __init__(self, margin=0.5):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
    
    def forward(self, inputs):
        assert inputs.shape[0] == 3, "Batchsize must be 3"
        assert len(inputs.shape) == 2, "Dim must be 2"
        anchor = inputs[0:1]
        pos = inputs[1:2]
        neg = inputs[2:3]
        anchor = F.normalize(anchor, p=2, dim=-1)
        pos = F.normalize(pos, p=2, dim=-1)
        neg = F.normalize(neg, p=2, dim=-1)
        # error_pos = torch.mm(anchor, pos.t)
        # error_neg = torch.mm(anchor, neg.t)
        error_pos = torch.sum(anchor * pos)
        error_pos = (1 - error_pos) / 2
        error_neg = torch.sum(anchor * neg)
        error_neg = (1 - error_neg) / 2
        loss = error_pos - error_neg + self.margin
        if loss < 0:
            loss *= 0
        return loss

class DiscriminativeLoss_3d(_Loss):
    # reference: 
    def __init__(self, delta_var, delta_dist, norm, n_instance=2,
                 size_average=True, reduce=True, usegpu=True):
        super(DiscriminativeLoss_3d, self).__init__(size_average)

        self.reduce = reduce
        assert self.size_average
        assert self.reduce
        self.usegpu = usegpu

        self.delta_var = float(delta_var)
        self.delta_dist = float(delta_dist)
        self.norm = int(norm)
        self.n_instance = n_instance

        assert self.norm in [1, 2]

    def discriminative_loss3d(self, inputs, target, n_objects, max_n_objects):
        alpha = beta = 1.0
        gamma = 0.001

        bs, n_filters, height, width = inputs.size()

    def forward(self, inputs, target, n_objects, max_n_objects):
        assert target.requires_grad is False
        return self.discriminative_loss3d(inputs, target, n_objects, max_n_objects)


if __name__ == "__main__":
    import pdb
    for i in range(10):
        arr1 = np.random.random((1, 14, 160, 160))
        pdb.set_trace()
        arr2 = np.random.random((1, 14, 160, 160))
        weight = np.random.random((1, 14, 160, 160))
        arr1 = torch.from_numpy(arr1)
        arr2 = torch.from_numpy(arr2)
        weight = torch.from_numpy(weight)

        Loss = WeightedBCELoss_wo_label()
        loss1 = Loss.binary_cross_entropy(arr1, arr2, weight)
        loss2 = F.binary_cross_entropy(arr1, arr2, weight)
        print(loss1, loss2)