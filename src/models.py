from resnet import ResNet18

from vgg import VGG

def get_model(data):
    if data == 'cifar10':
        model = ResNet18(num_classes=10)
    elif data == 'cifar100':
        model = VGG('VGG16',num_classes=100)

    return model
         
