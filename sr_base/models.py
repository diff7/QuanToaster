from torch import nn


class SRCNN(nn.Module):
    def __init__(self, num_channels=1, ker_one=9, ker_two=5, ker_three=5):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=ker_one, padding=ker_one // 2
        )
        self.conv2 = nn.Conv2d(
            64, 32, kernel_size=ker_two, padding=ker_two // 2
        )
        self.conv3 = nn.Conv2d(
            32, num_channels, kernel_size=ker_three, padding=ker_three // 2
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
