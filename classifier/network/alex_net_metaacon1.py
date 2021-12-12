from classifier import *
from acon import *

class AlexNetMetaAcon1(Module):
    def get_conv1(self, depth: Tuple[int, int, int]):
        d1, d2, d3 = depth
        layers = []
        layers += [nn.Conv2d(3, 16, (3, 3), padding='same')]
        if d1 == 1:
            layers += [nn.MaxPool2d(2)]
        layers += [nn.BatchNorm2d(16)]
        layers += [MetaAconC(16)]
        for i in range(d1-1):
            layers += [nn.Conv2d(16, 16, (3, 3), padding='same')]
            if i == d1-2:
                layers += [nn.MaxPool2d(2)]
            layers += [nn.BatchNorm2d(16)]
            layers += [MetaAconC(16)]

        layers += [nn.Conv2d(16, 64, (3, 3), padding='same')]
        if d2 == 1:
            layers += [nn.MaxPool2d(2)]
        layers += [nn.BatchNorm2d(64)]
        layers += [MetaAconC(64)]
        for i in range(d2 - 1):
            layers += [nn.Conv2d(64, 64, (3, 3), padding='same')]
            if i == d2 - 2:
                layers += [nn.MaxPool2d(2)]
            layers += [nn.BatchNorm2d(64)]
            layers += [MetaAconC(64)]
        return nn.Sequential(*layers)

    def get_conv2(self, n_way: int, depth: Tuple[int, int, int]):
        d1, d2, d3 = depth
        layers = []
        layers += [nn.Conv2d(64 * n_way, 256, (3, 3), padding='same')]
        layers += [nn.BatchNorm2d(256)]
        layers += [MetaAconC(256)]
        for i in range(d3 - 1):
            layers += [nn.Conv2d(256, 256, (3, 3), padding='same')]
            layers += [nn.BatchNorm2d(256)]
            layers += [MetaAconC(256)]
        layers += [nn.Conv2d(256, 64, (3, 3), padding='same'),
                   nn.MaxPool2d(2),
                   nn.BatchNorm2d(64),
                   MetaAconC(64),]
        return nn.Sequential(*layers)

    def get_dense1(self, n_way: int):
        return nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * n_way * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

    def get_dense2(self, n_way: int):
        return nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_way * 1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

    def __init__(self,
                 n_way: int,
                 depth: Tuple[int, int, int],):
        super(AlexNetMetaAcon1, self).__init__()
        self.n_way = n_way
        self.depth = depth
        conv1 = [self.get_conv1(depth) for _ in range(n_way)]
        conv2 = [self.get_conv2(n_way, depth) for _ in range(n_way)]
        dense1 = [self.get_dense1(n_way) for _ in range(n_way)]
        dense2 = [self.get_dense2(n_way) for _ in range(n_way)]
        self.last_layer = nn.Linear(n_way * 1024, 100)
        for i in range(n_way):
            setattr(self, f'conv1_way{i}', conv1[i])
            setattr(self, f'conv2_way{i}', conv2[i])
            setattr(self, f'dense1_way{i}', dense1[i])
            setattr(self, f'dense2_way{i}', dense2[i])

    def forward(self, x):
        output = []
        for i in range(self.n_way):
            conv1 = getattr(self, f'conv1_way{i}')
            output.append(conv1(x))
        x = torch.cat(output, dim=1) # concatenate in the channel dimension
        output = []
        for i in range(self.n_way):
            conv2 = getattr(self, f'conv2_way{i}')
            output.append(conv2(x))
        x = torch.cat(output, dim=1)
        x = Function.flatten(x)
        output = []
        for i in range(self.n_way):
            dense1 = getattr(self, f'dense1_way{i}')
            output.append(dense1(x))
        x = torch.cat(output, dim=1)
        output = []
        for i in range(self.n_way):
            dense2 = getattr(self, f'dense2_way{i}')
            output.append(dense2(x))
        x = torch.cat(output, dim=1)
        return self.last_layer(x)

