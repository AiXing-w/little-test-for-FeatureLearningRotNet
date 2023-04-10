import matplotlib.pyplot as plt


def compare(rotationName, superviseName, titleName):
    rotation_acc = []
    with open(rotationName) as f:
        for acc in f.readlines():
            rotation_acc.append(float(acc.strip()))

    supervise_acc = []
    with open(superviseName) as f:
        for acc in f.readlines():
            supervise_acc.append(float(acc.strip()))
    plt.plot(range(len(rotation_acc)), rotation_acc, label='rotation')
    plt.plot(range(len(supervise_acc)), supervise_acc, label='supervise')
    plt.xlabel('epochs')
    plt.ylabel('scores')
    plt.title(titleName)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    compare("resNet18_rotation_train_acc.txt", "resNet18_supervise_train_acc.txt", "train acc")
    compare("resNet18_rotation_train_loss.txt", "resNet18_supervise_train_loss.txt", "train loss")
    compare("resNet18_rotation_test_acc.txt", "resNet18_supervise_test_acc.txt", "test acc")
    compare("resNet18_rotation_test_loss.txt", "resNet18_supervise_test_loss.txt", "test loss")