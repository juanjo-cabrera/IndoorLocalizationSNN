import torch
import torch.nn as nn


vgg16 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)


class SiameseNetwork(nn.Module):
    # DEFINIMOS LA ESTRUCTURA DE LA RED NEURONAL
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = vgg16.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 16))
        self.fc1 = nn.Sequential(
            nn.Linear(4 * 16 * 512, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    # Aqui definimos como estan conectadas las capas entre sí, como pasamos el input al output, para cada red
    def forward(self, x):  # toma con la variable x el input
        verbose = False

        if verbose:
            print("Input: ", x.size())

        output = self.cnn1(x)

        if verbose:
            print("Output matricial: ", output.size())

        output = self.avgpool(output)
        if verbose:
            print("Output avgpool: ", output.size())
        output = torch.flatten(output, 1)
        output = self.fc1(output)
        return output

class SingleNetwork(nn.Module):
    #DEFINIMOS LA ESTRUCTURA DE LA RED NEURONAL
    PLANES = (500, 500, 5)
    def __init__(self):
        super(SingleNetwork, self).__init__()
        self.cnn1 = vgg16.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 16))
        self.fc1 = nn.Sequential(
          nn.Linear(4*16*512, self.PLANES[0]),
          nn.ReLU(inplace=True),
          nn.Linear(self.PLANES[0], self.PLANES[1]),
          nn.ReLU(inplace=True),
          nn.Linear(self.PLANES[1], self.PLANES[2]))
    def forward(self, x):
        verbose = False
        if verbose:
            print("Input: ", x.size())
        output = self.cnn1(x)
        if verbose:
            print("Output matricial: ", output.size())
        output = self.avgpool(output)
        if verbose:
            print("Output avgpool: ", output.size())
        output = torch.flatten(output, 1)
        output = self.fc1(output)
        return output

class VGG16_500_500_5(SingleNetwork):
    PLANES = (500, 500, 5)

class VGG16_4096_4096_1000(SingleNetwork):
    PLANES = (4096, 4096, 1000)

class VGG16_1000_1000_10(SingleNetwork):
    PLANES = (1000, 1000, 10)


class VGG16_500_500_5_s20(SingleNetwork):
    PLANES = (500, 500, 5)


class VGG16_500_500_5_s30(SingleNetwork):
    PLANES = (500, 500, 5)


class VGG16_500_500_5_s40(SingleNetwork):
    PLANES = (500, 500, 5)


class VGG16_500_500_5_s60(SingleNetwork):
    PLANES = (500, 500, 5)


class VGG16_500_500_5_s70(SingleNetwork):
    PLANES = (500, 500, 5)


class VGG16_500_500_5_s80(SingleNetwork):
    PLANES = (500, 500, 5)


vgg16_500_500_5 = VGG16_500_500_5()
vgg16_4096_4096_1000 = VGG16_4096_4096_1000()
vgg16_1000_1000_10 = VGG16_1000_1000_10()

print(vgg16_500_500_5)
print(vgg16_4096_4096_1000)
print(vgg16_1000_1000_10)

siamese_network = SiameseNetwork()


vgg16_500_500_5_s20 = VGG16_500_500_5_s20()
vgg16_500_500_5_s30 = VGG16_500_500_5_s30()
vgg16_500_500_5_s40 = VGG16_500_500_5_s40()
vgg16_500_500_5_s60 = VGG16_500_500_5_s60()
vgg16_500_500_5_s70 = VGG16_500_500_5_s70()
vgg16_500_500_5_s80 = VGG16_500_500_5_s80()
