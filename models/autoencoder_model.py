
import torch.nn.functional as F
import torch
from torch import nn

class model(nn.Module):
    """
    Autoencoder for learning latent space representation
    architecture simplified for only one hidden layer
    """

    def __init__(self, latent_dim, input_dim):
        super(model, self).__init__()

        # encoder
        self.enc1 = nn.Linear(in_features=input_dim, out_features=784)
        self.enc2 = nn.Linear(in_features=784, out_features=256)
        self.enc3 = nn.Linear(in_features=256, out_features=128)
        self.enc4 = nn.Linear(in_features=128, out_features=64)
        self.enc5 = nn.Linear(in_features=64, out_features=latent_dim)

        # decoder

        self.dec1 = nn.Linear(in_features=latent_dim, out_features=64)
        self.dec2 = nn.Linear(in_features=64, out_features=128)
        self.dec3 = nn.Linear(in_features=128, out_features=256)
        self.dec4 = nn.Linear(in_features=256, out_features=784)
        self.dec5 = nn.Linear(in_features=784, out_features=input_dim)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))

        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        return x

    def predict(self, x):
        with torch.no_grad():

            x = torch.tensor(x)

            x = F.relu(self.enc1(x.float()))
            x = F.relu(self.enc2(x))
            x = F.relu(self.enc3(x))
            x = F.relu(self.enc4(x))
            x = F.relu(self.enc5(x))

        return x

