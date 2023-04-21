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
    parser.add_argument('--device', type=str, default='0', help='device')
    parser.add_argument('--qat', action='store_true', help='qat val')
    parser.add_argument('--thresh_qat', type=str, default='null', help='thresh qat file path')
    parser.add_argument('--bits', type=int, default=8, help='quantify bits')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_opt()
    # select device
    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # read cfg
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    data_path = cfg['dataset']
    batch_size = cfg['batch_size']
    epoch = cfg['epoch']
    learning_rate = cfg['learning_rate']

    # prepare test data
    test_set = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # quantitative weight
    state_dict = torch.load(args.weight)
    thresh_qat = []
    if args.qat:
        assert args.thresh_qat != 'null', 'need thresh_qat file for qat'
        with open(args.thresh_qat, 'r') as fr:
            content = fr.read()
            numbers = content.split()
            thresh_qat = list(map(float, numbers))
    snn_qat = SCNN(cfg, device, qat=args.qat, qat_bits=args.bits, thresh_qat=thresh_qat)
    snn_qat.load_state_dict(state_dict)
    snn_qat.to(device)
    
    # val
    criterion = nn.MSELoss()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = snn_qat(inputs)
            labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), labels_)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
    acc = 100. * float(correct) / float(total)
    print('acc: ', acc)