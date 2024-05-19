import torch
import torch.nn as nn


class Alexnet(nn.Module):

    def __init__(self,ch_in,cls_num):
        super().__init__()
        self.cls = cls_num
        self.ch_in = ch_in
        self.sequence = nn.Sequential(
            nn.Conv2d(self.ch_in, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=6 * 6 * 256, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1000),
            nn.ReLU(),
        )
        self.cls = nn.Linear(in_features=1000, out_features=self.cls)

        self._initialize_weights()


    def forward(self,x):
        x = self.sequence(x)
        x = self.linear(x)
        out = self.cls(x)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight,0)
                nn.init.constant_(m.bias,0)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_data = torch.randn(size=(1,3,227,227),dtype=torch.float32,device=device)
    model = Alexnet(ch_in=3,cls_num=3)
    model = model.to(device=device)
    output = model(input_data)
    print(output.shape)
