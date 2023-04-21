import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math

global lens
lens = 0

global thresh
thresh = 0

# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer

# define approximate firing function
class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply

class SCNN(nn.Module):
    def __init__(self, cfg, device, qat=False, qat_bits=8, thresh_qat=[]):
        super(SCNN, self).__init__()
        # parameter
        global lens
        lens = cfg['lens']
        global thresh
        thresh = cfg['thresh']
        # qat infer must have thresh_qat
        if qat: 
            assert(len(thresh_qat) != 0)
            # thresh of every layer are different
            self.thresh_qat = thresh_qat
        self.batch_size = cfg['batch_size']
        self.nc = cfg['nc']
        self.decay = cfg['decay']
        self.time_window = cfg['time_window']
        self.device = device
        self.qat = qat
        self.qat_bits = qat_bits
        
        self.encode = []
        self.shapes = []
        self.names = []
        self.layers = None
        m = None
        # data of input [n, c, h, w]
        shape = [self.batch_size, cfg['planes'], cfg['height'], cfg['width']]
        for name, c in cfg['net']:
            self.names.append(name)
            new_shape = []
            new_shape.append(self.batch_size)
            if name == 'Conv':
                assert(len(shape) == 4)
                m = nn.Conv2d(shape[1], c[0], kernel_size=c[1], stride=c[2], padding=c[3], bias=False)
                new_shape.append(c[0])
                new_shape.append(math.floor((shape[2] + 2 * c[3] - c[1]) / c[2] + 1))
                new_shape.append(math.floor((shape[3] + 2 * c[3] - c[1]) / c[2] + 1))
            elif name == 'Pool2d':
                assert(len(shape) == 4)
                m = nn.MaxPool2d(kernel_size=c[0], stride=c[1], padding=c[2])
                new_shape.append(shape[1])
                new_shape.append(math.floor((shape[2] + 2 * c[2] - c[0]) / c[1] + 1))
                new_shape.append(math.floor((shape[3] + 2 * c[2] - c[0]) / c[1] + 1))
            elif name == 'Fc':
                if(len(shape) == 4):
                    # last layer isn't linear
                    m = nn.Linear(shape[1]*shape[2]*shape[3], c[0], bias=False)
                elif(len(shape) == 2):
                    m = nn.Linear(shape[1], c[0], bias=False)
                else:
                    raise IndexError("length of shape must be 2 or 4")
                new_shape.append(c[0])
            else:
                raise IndexError("Invilid layer, only support Conv, Pool2d and Fc")
            shape = new_shape
            self.shapes.append(new_shape)
            self.encode.append(m)
        self.layers = nn.ModuleList(self.encode)

    def get_output_shape(self):
        return self.shapes

    def forward(self, input):
        # data initial
        # all data
        qat_layer_index = 0
        data_mem = []
        data_spike = []
        final_spike = torch.zeros(self.shapes[-1], device=self.device)
        for i in range(len(self.names)):
            data_mem.append(torch.zeros(self.shapes[i], device=self.device))
            data_spike.append(torch.zeros(self.shapes[i], device=self.device))

        for step in range(self.time_window): # simulation time steps
            # code
            x = input > torch.rand(input.size(), device=self.device) # prob. firing
            for i in range(len(self.names)):
                x = x.float() if i == 0 else data_spike[i-1]
                x = x.view(self.batch_size, -1) if (len(x.shape) == 4 and len(data_spike[i].shape) == 2) else x
                if self.names[i] != 'Pool2d':
                    # Pool2d don't need time window
                    if self.qat:
                        global thresh
                        thresh = self.thresh_qat[qat_layer_index]
                        qat_layer_index = 0 if (qat_layer_index == len(self.thresh_qat) - 1) else (qat_layer_index + 1)
                    data_mem[i], data_spike[i] = self.mem_update(self.layers[i], x, data_mem[i], data_spike[i])
                else:
                    data_spike[i] = self.layers[i](x)
            final_spike += data_spike[-1]

        outputs = final_spike / self.time_window
        return outputs
    
    # membrane potential update
    def mem_update(self, ops, x, mem, spike):
        # mem = mem * self.decay * (1. - spike) + ops(x)
        # spike = act_fun(mem) # act_fun : approximation firing function
        mem = mem + ops(x)
        if self.qat:
            mem = torch.clamp(mem, min=-2**(self.qat_bits-1), max=2**(self.qat_bits-1)-1)
        spike = act_fun(mem)
        # get integer
        if self.qat:
            mem = torch.trunc(mem * self.decay * (1. - spike))
        else:
            mem = mem * self.decay * (1. - spike)
        return mem, spike
    
