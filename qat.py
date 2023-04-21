from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time
import numpy as np
import argparse
import yaml
from model.snn import*
from utils import qat

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='cfg file')
    parser.add_argument('--weight', type=str, required=True, help='weight file')
    parser.add_argument('--bits', type=int, default=8, help='quantify bits')
    parser.add_argument('--inlayers', action='store_true', help='quantified by stratification')
    parser.add_argument('--save', type=str, required=True, help='save dictionary')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_opt()
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    thresh = cfg['thresh']

    # quantitative weight
    state_dict = torch.load(args.weight)
    if args.inlayers:
        state_dict, thresh_qat = qat.qat_weight_inlayers(state_dict, args.bits, thresh)
    else:
        state_dict, thresh_qat = qat.qat_weight(state_dict, args.bits, thresh)

    # save weight and qat_ratio
    if not os.path.isdir(args.save): os.makedirs(args.save)
    torch.save(state_dict, args.save + '/qat.pt')
    with open(args.save + '/thresh_qat.txt', 'w') as fw:
        for thresh in thresh_qat:
            fw.write(f'{thresh} ')