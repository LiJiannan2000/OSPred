import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class CoxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad = {}

    def save_grad(self, name):
        def hook(grad):
            self.grad[name] = grad
        return hook

    def forward(self, risk, ostime, death):
        risk = risk.float()

        risk_set = torch.cumsum(torch.exp(risk).flip(dims=(0,)), dim=0).flip(dims=(0,))

        diff = (risk - torch.log(risk_set)) * death
        loss = -torch.sum(diff)
        if torch.sum(death) == 0:
            return loss
        loss = loss / torch.sum(death)

        return loss


def distillation_loss(def_post, out1, out2, out3, out4, data_def, teacher_out1, teacher_out2, teacher_out3, teacher_out4, temperature=1.0):
    data_def = data_def[:, 1:, :].flatten(start_dim=1)
    ld = F.mse_loss(def_post / temperature, data_def / temperature)
    l1 = F.mse_loss(out1 / temperature, teacher_out1 / temperature)
    l2 = F.mse_loss(out2 / temperature, teacher_out2 / temperature)
    l3 = F.mse_loss(out3 / temperature, teacher_out3 / temperature)
    l4 = F.mse_loss(out4 / temperature, teacher_out4 / temperature)
    loss = [ld, l1, l2, l3, l4]
    return loss
