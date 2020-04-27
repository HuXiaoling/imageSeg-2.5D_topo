import torch
import torch.nn as nn


def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def conv_trans_block_3d(in_dim, out_dim, activation, kernel_size, stride, padding, output_padding):
    return nn.Sequential(
    	# 64, 3, kernel_size=4, stride=(2, 4, 4), bias=False, padding=(1, 8, 8)
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
        nn.BatchNorm3d(out_dim),
        activation,)


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),)

class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)
        
        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.pool_5 = max_pooling_3d()
        
        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation)
        
        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), output_padding=(0,1,1))
        self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation,)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), output_padding=(0,1,1))
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), output_padding=(0,1,1))
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation, kernel_size=(2,3,3), stride=(1,2,2), padding=(0,1,1), output_padding=(0,1,1))
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation, kernel_size=(4,3,3), stride=(1,2,2), padding=(0,1,1), output_padding=(0,1,1))
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)
        
        # Output
        self.out = conv_block_3d(self.num_filters, out_dim, activation)
    
    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x) # -> [1, 32, 3, 1250, 1250]
        # print('down_1', down_1.shape)
        pool_1 = self.pool_1(down_1) # -> [1, 32, 1, 625, 625]
        # print('pool_1', pool_1.shape)

        down_2 = self.down_2(pool_1) # -> [1, 64, 1, 625, 625]
        # print('down_2', down_2.shape)
        pool_2 = self.pool_2(down_2) # -> [1, 64, 1, 312, 312]
        # print('pool_2', pool_2.shape)

        down_3 = self.down_3(pool_2) # -> [1, 128, 1, 312, 312]
        # print('down_3', down_3.shape)
        pool_3 = self.pool_3(down_3) # -> [1, 128, 1, 156, 156]
        # print('pool_3', pool_3.shape)

        # down_4 = self.down_4(pool_3) # -> [1, 256, 1, 156, 156]
        # print('down_4', down_4.shape)
        # pool_4 = self.pool_4(down_4) # -> [1, 256, 1, 78, 78]
        # print('pool_4', pool_4.shape)

        # down_5 = self.down_5(pool_4) # -> [1, 64, 8, 8, 8]
        # pool_5 = self.pool_5(down_5) # -> [1, 64, 4, 4, 4]
        
        # Bridge
        bridge = self.down_4(pool_3) # -> [1, 512, 1, 78, 78]
        # print('bridge', bridge.shape)

        
        # Up sampling
        # trans_1 = self.trans_1(bridge) # -> [1, 128, 8, 8, 8]
        # concat_1 = torch.cat([trans_1, down_5], dim=1) # -> [1, 192, 8, 8, 8]
        # up_1 = self.up_1(concat_1) # -> [1, 64, 8, 8, 8]
        
        # trans_2 = self.trans_2(bridge) # -> [1, 512, 1, 156, 156]
        # print('trans_2', trans_2.shape)
        # concat_2 = torch.cat([trans_2, down_4], dim=1) # -> [1, 96, 16, 16, 16]
        # up_2 = self.up_2(concat_2) # -> [1, 256, 1, 156, 156]
        # print('up_2',up_2.shape)
        # 
        trans_3 = self.trans_3(bridge) # -> [1, 256, 1, 312, 312]
        # print('trans_3', trans_3.shape)
        concat_3 = torch.cat([trans_3, down_3], dim=1) # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3) # -> [1, 128, 1, 312, 312]
        # print('up_3', up_3.shape)

        
        trans_4 = self.trans_4(up_3) # -> [1, 128, 1, 625, 625]
        # print('trans_4', trans_4.shape)
        concat_4 = torch.cat([trans_4, down_2], dim=1) # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4) # -> [1, 64, 1, 625, 625]
        # print('up_4', up_4.shape)
        
        trans_5 = self.trans_5(up_4) # -> [1, 64, 3, 1250, 1250]
        # print('trans_5', trans_5.shape)
        concat_5 = torch.cat([trans_5, down_1], dim=1) # -> [1, 12, 128, 128, 128]
        up_5 = self.up_5(concat_5) # -> [1, 32, 3, 1250, 1250]
        # print('up_5', up_5.shape)
        
        # Output
        out = self.out(up_5) # -> [1, 2, 3, 1250, 1250]
        # print('out', out.shape)
        return out

if __name__ == "__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  image_size = 256
  x = torch.Tensor(1, 1, 5, image_size, image_size)
  x.to(device)
  print("x size: {}".format(x.size()))
  
  model = UNet(in_dim=1, out_dim=2, num_filters=32)
  
  out = model(x)
  print("out size: {}".format(out.size()))
  # print(out.shape,out)


