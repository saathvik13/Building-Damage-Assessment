import torch
import torch.nn.functional as F
from torchvision.models import resnet34,resnet18,resnet50
from torch.nn import Linear, Module, ModuleList, Sequential
from torch import Tensor


#SiameseEncoder is where we import the 
class SiameseEncoder(Module):
    def __init__(self, diff: bool=True) -> None:
        super().__init__()
        self.diff = diff
        self.model = resnet34(pretrained=True, progress=False)
        self.model = Sequential(*(list(self.model.children())[:-1]))
    
    def forward(self, x: Tensor) -> torch.Tensor:
        x = x.reshape((-1, 6, 128, 128))
        x1 = x[:,:3,:,:]
        x2 = x[:,3:,:,:]
        x1 = self.model(x1)
        x2 = self.model(x2)
        if self.diff:
            x = torch.add(input=x1, other=x2, alpha=-1)
        else:
            x = torch.cat((x1, x2), dim=1)
        return x.flatten(start_dim=1)
    
    @torch.no_grad()
    def get_output_shape(self) -> int:
        x = torch.rand(98304)
        x = self.forward(x)
        return x.shape[1]


class SiameseCNN(Module):
    def __init__(self, hidden_channels: int, num_classes: int, dropout_rate: float, diff: bool=True) -> None:
        super().__init__()
        
        self.dropout_rate = dropout_rate

        self.encoder = SiameseEncoder(diff)
        self.fc1 = Linear(self.encoder.get_output_shape(), hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.fc2 = Linear(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.out = Linear(hidden_channels, num_classes)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.out(x)
        return F.log_softmax(x, dim=1)