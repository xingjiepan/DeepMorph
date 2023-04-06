# This module is re-implemented based on the code from
# https://github.com/wjNam/Relative_Attributing_Propagation/blob/master/modules/layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_divide(a, b):
    return a / (b + b.eq(0).type(b.type()) * 1e-9) * b.ne(0).type(b.type())


def lrp_forward_hook(module, input, output):
    module.X_lrp = input[0].detach().clone()
    module.X_lrp.requires_grad = True
    module.Y_lrp = output.detach().clone()


class LRPBase(nn.Module):
    def __init__(self):
        super(LRPBase, self).__init__()
        self.register_forward_hook(lrp_forward_hook)

    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C
        
class Flatten(nn.Flatten):
    def forward(self, input):
        self.input_shape = input.shape
        return super(Flatten, self).forward(input)
        
class Add():
    def __init__(self):
        pass
        
    def add(self, A, B):
        self.A_lrp = A.detach().clone()
        self.A_lrp.requires_grad = True
        self.B_lrp = B.detach().clone()
        self.B_lrp.requires_grad = True
        
        return A + B
    
    def lrp(self, R):
        Z = self.A_lrp + self.B_lrp
        S = safe_divide(R, Z)
        Ca = torch.autograd.grad(Z, self.A_lrp, S, retain_graph=True)[0]
        Cb = torch.autograd.grad(Z, self.B_lrp, S, retain_graph=True)[0]
     
        return (self.A_lrp * Ca).detach(), (self.B_lrp * Cb).detach()
        
        
class Linear(nn.Linear, LRPBase):
    def lrp(self, R):
        alpha = 2
        beta = 1
    
        return self.lrp_ab(R, alpha, beta)
    
    def lrp_ab(self, R, alpha, beta):
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X_lrp, min=0)
        nx = torch.clamp(self.X_lrp, max=0)
        
        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1, bias=None)
            Z2 = F.linear(x2, w2, bias=None)
            S1 = safe_divide(R, Z1)
            S2 = safe_divide(R, Z2)
            C1 = x1 * self.gradprop(Z1, x1, S1)[0]
            C2 = x2 * self.gradprop(Z2, x2, S2)[0]
            
            return C1 + C2
        
        pos_R = f(pw, nw, px, nx)
        neg_R = f(nw, pw, px, nx)
        
        return (alpha * pos_R - beta * neg_R).detach()
        
class Conv2d(nn.Conv2d, LRPBase):
    def lrp(self, R):
        alpha = 2
        beta = 1
        
        return self.lrp_ab(R, alpha, beta)
    
    def lrp_ab(self, R, alpha, beta):
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X_lrp, min=0)
        nx = torch.clamp(self.X_lrp, max=0)
        
        def f(w1, w2, x1, x2):
            Z1 = F.conv2d(x1, w1, bias=None, stride=self.stride, padding=self.padding)
            Z2 = F.conv2d(x2, w2, bias=None, stride=self.stride, padding=self.padding)
            S1 = safe_divide(R, Z1)
            S2 = safe_divide(R, Z2)
            C1 = x1 * self.gradprop(Z1, x1, S1)[0]
            C2 = x2 * self.gradprop(Z2, x2, S2)[0]
            return C1 + C2
        
        pos_R = f(pw, nw, px, nx)
        neg_R = f(nw, pw, px, nx)
        
        return (alpha * pos_R - beta * neg_R).detach()
    
class ConvTranspose2d(nn.ConvTranspose2d, LRPBase):
    def lrp(self, R):
        alpha = 2
        beta = 1
        
        return self.lrp_ab(R, alpha, beta)
    
    def lrp_ab(self, R, alpha, beta):
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X_lrp, min=0)
        nx = torch.clamp(self.X_lrp, max=0)
        
        def f(w1, w2, x1, x2):
            Z1 = F.conv_transpose2d(x1, w1, bias=None, stride=self.stride, padding=self.padding)
            Z2 = F.conv_transpose2d(x2, w2, bias=None, stride=self.stride, padding=self.padding)
            S1 = safe_divide(R, Z1)
            S2 = safe_divide(R, Z2)
            C1 = x1 * self.gradprop(Z1, x1, S1)[0]
            C2 = x2 * self.gradprop(Z2, x2, S2)[0]
            return C1 + C2
        
        pos_R = f(pw, nw, px, nx)
        neg_R = f(nw, pw, px, nx)
        
        return (alpha * pos_R - beta * neg_R).detach()
        