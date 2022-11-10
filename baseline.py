import torch
from cnn import Net
from data_provider import testloader
import time
PATH = './trained_model/cifar_net.pth'
net = Net()
net.load_state_dict(torch.load(PATH))


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
# net = net.to(device)
t1 = time.time()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images
        labels = labels
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
t2 = time.time()
print(f"time use sis {t2 - t1}")
print(
    f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
