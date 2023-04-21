from matplotlib import pyplot as plt

def plot_curve(save_dir, loss_train_record, loss_test_record, acc_record):
    # loss_train_record
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.cla()
    x = range(1, len(loss_train_record)+1)
    y = loss_train_record
    plt.title('Train loss vs. epoches', fontsize=10)
    plt.plot(x, y, '.-')
    plt.xlabel('epoches', fontsize=10)
    plt.ylabel('Train loss', fontsize=10)
    
    # loss_test_record
    plt.subplot(1, 3, 2)
    plt.cla()
    x = range(1, len(loss_test_record)+1)
    y = loss_test_record
    plt.title('Test loss vs. epoches', fontsize=10)
    plt.plot(x, y, '.-')
    plt.xlabel('epoches', fontsize=10)
    plt.ylabel('Test loss', fontsize=10)

    # acc_record
    plt.subplot(1, 3, 3)
    plt.cla()
    x = range(1, len(acc_record)+1)
    y = acc_record
    plt.title('Accuracy vs. epoches', fontsize=10)
    plt.plot(x, y, '.-')
    plt.xlabel('epoches', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)

    plt.savefig(save_dir + '/curve')
