import matplotlib.pyplot as plt


def plot_history(name, history):
    # 显示训练和测试的损失和精度
    train_loss = history['train_loss']
    train_acc = history['train_acc']
    test_loss = history['test_loss']
    test_acc = history['test_acc']
    with open(name + "_train_loss.txt", 'w') as f:
        for t_l in train_loss:
            f.write(str(t_l))
            f.write("\n")

    with open(name + "_train_acc.txt", 'w') as f:
        for t_a in train_acc:
            f.write(str(t_a))
            f.write("\n")

    with open(name + "_test_loss.txt", 'w') as f:
        for t_l in test_loss:
            f.write(str(t_l))
            f.write("\n")

    with open(name + "_test_acc.txt", 'w') as f:
        for t_a in test_acc:
            f.write(str(t_a))
            f.write("\n")

    plt.plot(range(len(train_loss)), train_loss, label='train loss')
    plt.plot(range(len(train_acc)), train_acc, label='train acc')
    plt.plot(range(len(test_loss)), test_loss, label='test loss')
    plt.plot(range(len(test_acc)), test_acc, label='test acc')
    plt.xlabel('epochs')
    plt.ylabel('scores')
    plt.title(name)
    plt.legend()
    plt.savefig('{}.jpg'.format(name))
