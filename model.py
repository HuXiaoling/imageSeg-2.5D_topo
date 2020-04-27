import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
        nn.ReLU(inplace=True),
    )


class U_Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(U_Net, self).__init__()

        self.conv1_block = double_conv(in_channels,out_channels)
        self.conv2_block = double_conv(32,64)
        self.conv3_block = double_conv(64, 128)
        self.conv4_block = double_conv(128, 256)
        self.conv5_block = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) # the stride of the window. Default value is kernel_size

        # Up 1
        self.up_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv_up_1 = double_conv(512, 256)

        # Up 2
        self.up_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv_up_2 = double_conv(256, 128)

        # Up 3
        self.up_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2) # for 1250 * 1250 kernel_size=3, stride=2
        self.conv_up_3 = double_conv(128, 64)

        # Up 4
        self.up_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.conv_up_4 = double_conv(64, 32)

        # Final output
        self.conv_final = nn.Conv2d(in_channels=32, out_channels=2,
                                    kernel_size=1, padding=0, stride=1)
        self.softmax = nn.Softmax2d()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        print('input', x.shape)
        # Down 1
        conv1 = self.conv1_block(x)
        print('after conv1', conv1.shape)
        x = self.maxpool(conv1)
        print('before conv2', x.shape)
        # x = self.dropout(x)
        # Down 2
        conv2 = self.conv2_block(x)
        print('after conv2', conv2.shape)
        x = self.maxpool(conv2)
        print('before conv3', x.shape)

        # Down 3
        conv3 = self.conv3_block(x)
        print('after conv3', conv3.shape)
        x = self.maxpool(conv3)
        print('before conv4', x.shape)

        # # # Down 4
        conv4 = self.conv4_block(x)
        print('after conv4', conv4.shape)
        x = self.maxpool(conv4)
        # # Midpoint
        print('before conv5', x.shape)
        x = self.conv5_block(x)
        print('after conv5', x.shape)

        # Up 1
        x =  self.up_1(x)
        print('up_1', x.shape)
        x = torch.cat([x, conv4], dim=1)
        print('after cat_4', x.shape)

        x = self.conv_up_1(x)
        print('after conv_4', x.shape)

        # # Up 2
        x = self.up_2(x)
        print('up_2', x.shape)
        x = torch.cat([x, conv3], dim=1)
        print('after cat_3', x.shape)
        x = self.conv_up_2(x)
        print('after conv_3', x.shape)

        # Up 3
        x = self.up_3(x)
        print('up_3', x.shape)
        x = torch.cat([x, conv2], dim=1)
        print('after cat_2', x.shape)
        x = self.conv_up_3(x)
        print('after conv_2', x.shape)

        # Up 4
        x = self.up_4(x)
        print('up_4', x.shape)
        x = torch.cat([x, conv1], dim=1)
        print('after cat_1', x.shape)
        x = self.conv_up_4(x)
        print('after conv_1', x.shape)

        # Final output
        x = self.conv_final(x)
        print('final: ', x.shape)
        likelihood_map = self.softmax(x)[:,1,:,:]
        print(likelihood_map.shape)

        return x, likelihood_map


if __name__ == "__main__":

    # A full forward pass
    im = torch.randn(2, 1, 1250, 1250)
    model = U_Net(1, 32)
    x, likelihood_map = model(im)
    # print(x,likelihood_map)
    print(x.shape)
    del model
    del x
    print(x.shape)
