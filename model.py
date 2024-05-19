import torch
import torch.nn as nn


class Alexnet(nn.Module):

    def __init__(self,ch_in,cls_num):
        super().__init__()
        self.cls = cls_num
        self.ch_in = ch_in
        # TODO put your code hier, 'Initialization'
        self._initialize_weights()


    def forward(self,x):
        # TODO put your code hier
        out = x

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

class VGG(nn.Module):

    def __init__(self,ch_in,cls_num):
        super().__init__()
        self.cls = cls_num
        self.ch_in = ch_in
        # TODO put your code hier, 'Initialization'
        self._initialize_weights()


    def forward(self,x):
        # TODO put your code hier
        out = x

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

    # TODO create your created model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_data = torch.randn(size=(1,3,227,227),dtype=torch.float32,device=device)
    model = Alexnet(ch_in=3,cls_num=3)
    model = model.to(device=device)
    output = model(input_data)
    print(output.shape)
