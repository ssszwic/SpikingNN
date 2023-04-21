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
from matplotlib import pyplot as plt

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='weight file')
    parser.add_argument('--save', type=str, required=True, help='save dictionary')
    parser.add_argument('--qat', action='store_true', help='save qat weight')
    parser.add_argument('--plot', action='store_true', help='plot histogram')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_opt()
    # make floder
    save_dir = args.save[0:-1] if args.save[-1] == '/' else args.save
    save_dir = save_dir
    if not os.path.isdir(save_dir):  
        os.makedirs(save_dir)
    # read parameter
    state_dict = torch.load(args.weights)

    index = 0
    weights = []
    for layer, tensor in state_dict.items():
        weight = np.ravel(tensor.cpu().numpy())
        basic_name = save_dir + '/' + str(index)
        if args.qat:
            weight = weight.astype(np.int32)
            fmt='%d'
        else:
            fmt='%.6f'
        np.savetxt(basic_name + '.txt', weight, fmt=fmt)
        index += 1
        weights.append(weight)
        if args.plot:
            plt.figure()
            n, bins, patches = plt.hist(weight, bins=100)
            plt.savefig(basic_name + '.png')
    
    # plot all weight
    if args.plot:
        weight_all = weights[0]
        for i in range(1, len(weights)):
            weight_all = np.append(weight_all, weights[i])
        plt.figure()
        n, bins, patches = plt.hist(weight_all, bins=100)
        plt.savefig(save_dir + '/all.png')

    # snn.load_state_dict(state_dict)
    # print(snn)



