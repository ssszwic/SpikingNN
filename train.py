from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time
import numpy as np
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter
from model.snn import*
from utils import plot

def train(cfg, save_dir, device):
    if not os.path.isdir(save_dir):  
        os.makedirs(save_dir)
    # save cfg file
    with open(save_dir + '/cfg.yaml', 'w') as f:
        yaml.dump(data=cfg, stream=f, allow_unicode=True)

    # argparse parameter
    num_epochs = cfg['epoch']
    batch_size = cfg['batch_size']
    data_path = cfg['dataset']
    learning_rate = cfg['learning_rate']

    # load dataset
    train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_set = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # initial writer
    writer = SummaryWriter(log_dir=save_dir + '/log')

    # initial module
    snn = SCNN(cfg, device)
    snn.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)

    best_acc = 0  # best test accuracy
    best_epoch = 0
    acc_record = list([])
    loss_train_record = list([])
    loss_test_record = list([])

    for epoch in range(num_epochs):
        running_loss = 0
        epoch_train_loss = 0
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            snn.zero_grad()
            optimizer.zero_grad()

            images = images.float().to(device)
            # save input data
            # array = images.cpu().numpy().reshape(1, -1)
            # np.savetxt('tensor.txt', array)
            outputs = snn(images)
            labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), labels_)
            running_loss += loss.item()
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i+1)%100 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                        %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,running_loss/100 ))
                running_loss = 0
                print('Time elasped:', time.time()-start_time)
        correct = 0
        total = 0
        optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)
        loss_train_record.append(epoch_train_loss)
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)

        # var
        with torch.no_grad():
            epoch_val_loss = 0
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                optimizer.zero_grad()
                outputs = snn(inputs)
                labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
                loss = criterion(outputs.cpu(), labels_)
                epoch_val_loss += loss
                _, predicted = outputs.cpu().max(1)
                total += float(targets.size(0))
                correct += float(predicted.eq(targets).sum().item())
            loss_test_record.append(epoch_val_loss)
            writer.add_scalar('Loss/test', epoch_val_loss, epoch)

        print('Iters:', epoch, 'Test Accuracy of the model on the 10000 test images: %.3f\n' % (100 * correct / total))
        acc = float(correct) / float(total)
        acc_record.append(acc)
        writer.add_scalar('Accuarcy', acc, epoch)

        # save best result
        if best_acc < acc:
            torch.save(snn.state_dict(), save_dir + '/best.pt')
            best_acc = acc
            best_epoch = epoch + 1
    
    # recode model and train information ##########################################################################
    # save final result
    torch.save(snn.state_dict(), save_dir + '/final.pt')
    # param string
    param_str = 'parameter:\n'
    for name, p in snn.named_parameters():
        param_str +=  'name' + ': ' + str(p.shape) + '\n'
    # output shape string
    shape_str = 'output shape:\n'
    output_shapes = snn.get_output_shape()
    for i in range(len(output_shapes)):
        shape_str += 'layers.' + str(i) + str(output_shapes[i]) + '\n'
    with open(save_dir + '/result.txt', 'w') as f:
        f.write('acc_record: ' + str(acc_record) + '\n\n')
        f.write('loss_train_record: ' + str(loss_train_record) + '\n\n')
        f.write('loss_test_record: ' + str(loss_test_record) + '\n\n')
        f.write('best_acc: ' + str(best_acc) + ' for epoch: ' + str(best_epoch) + '\n\n')
        f.write(str(snn) + '\n\n')
        f.write(shape_str + '\n')
        f.write(param_str + '\n')
        
    # print cur ##########################################################################
    plot.plot_curve(save_dir, loss_train_record, loss_test_record, acc_record)

    print(acc_record)
    print('best_acc:', best_acc, 'for epoch: ', best_epoch)
    print(snn)
    print(param_str)
    print(shape_str)
    print('result save at ' + save_dir)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='cfg file')
    parser.add_argument('--device', type=str, default='0', help='device')
    parser.add_argument('--save', type=str, required=True, help='save dictionary')
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
    # train
    save_dir = args.save[0:-1] if args.save[-1] == '/' else args.save
    train(cfg, save_dir, device)
