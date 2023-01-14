from loguru import logger

import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # coarse
        self.correct_thr = config["fine_correct_thr"]  # 1 represents within window
        self.c_pos_w = config["pos_weight"]
        self.c_neg_w = config["neg_weight"]
        # fine
        self.fine_type = config["fine_type"]

    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        if self.config["coarse_type"] == "focal":
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            alpha = self.config["focal_alpha"]
            gamma = self.config["focal_gamma"]

            loss_pos = (
                -alpha
                * torch.pow(1 - conf[conf_gt == 1], gamma)
                * (conf[conf_gt == 1]).log()
            )
            loss_neg = (
                -(1 - alpha)
                * torch.pow(conf[conf_gt == 0], gamma)
                * (1 - conf[conf_gt == 0]).log()
            )
            if weight is not None:
                loss_pos = loss_pos * weight[conf_gt == 1]
                loss_neg = loss_neg * weight[conf_gt == 0]
            
            if loss_pos.shape[0] == 0:
                logger.warning('len of loss pos is zero!')
                loss_mean = self.c_neg_w * loss_neg.mean()
            elif loss_neg.shape[0] == 0:
                logger.warning('len of loss neg is zero!')
                loss_mean = self.c_pos_w * loss_pos.mean()
            else:
                loss_pos_mean = loss_pos.mean()
                loss_neg_mean = loss_neg.mean()
                loss_mean = self.c_pos_w * loss_pos_mean + self.c_neg_w * loss_neg_mean
            
            return loss_mean
            # each negative element has smaller weight than positive elements. => higher negative loss weight
        else:
            raise NotImplementedError

    def compute_fine_loss(self, expec_f, expec_f_gt):
        if self.fine_type == "l2_with_std":
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt)
        else:
            raise NotImplementedError()

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 3] <x, y, std>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = (
            torch.linalg.norm(expec_f_gt, ord=float("inf"), dim=1) < self.correct_thr
        )

        # use std as weight that measures uncertainty
        std = expec_f[:, 2]
        inverse_std = 1.0 / torch.clamp(std, min=1e-10)
        weight = (
            inverse_std / torch.mean(inverse_std)
        ).detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if correct_mask.sum() == 0:
            if (
                self.training
            ):  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
                weight[0] = 1e-6
            else:
                return None

        # l2 loss with std
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(
            -1
        )
        loss = (offset_l2 * weight[correct_mask]).mean()

        return loss

    @torch.no_grad()
    def compute_c_weight(self, data):
        if "mask0" in data:
            c_weight = (
                data["mask0"].flatten(-2)[..., None]
                * data["mask1"].flatten(-2)[:, None]
            )
        else:
            c_weight = None
        return c_weight

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        c_weight = self.compute_c_weight(data)

        loss_c = self.compute_coarse_loss(
            data["conf_matrix"], data["conf_matrix_gt"], weight=c_weight
        )
        loss = loss_c * self.config["coarse_weight"]
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        # 2. fine-level loss
        if 'expec_f' in data:
            loss_f = self.compute_fine_loss(data["expec_f"], data["expec_f_gt"])
            if loss_f is not None:
                loss += loss_f * self.config["fine_weight"]
                loss_scalars.update({"loss_f": loss_f.clone().detach().cpu()})
            else:
                assert self.training is False
                loss_scalars.update({"loss_f": torch.tensor(1.0)})  # 1 is the upper bound

        loss_scalars.update({"loss": loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
