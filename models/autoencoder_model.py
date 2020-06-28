
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
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        device = torch.device(dev)
        with torch.no_grad():

            x = torch.tensor(x)
            x = x.to(device)

            x = F.relu(self.enc1(x.float()))
            x = F.relu(self.enc2(x))
            x = F.relu(self.enc3(x))
            x = F.relu(self.enc4(x))
            x = F.relu(self.enc5(x))

        return x


class Autoencoder():
    def __init__(self, latent_dim=32, epochs=10, batch_size=2, input_dim=1026):
        self.history = []
        self.model = model(latent_dim, input_dim)
        self.num_epochs = epochs

    def fit(self, X):

        learning_rate = 1e-3
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        train_loss = []
        # setting up the device
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        device = torch.device(dev)
        print(device)
        self.model.to(device)

        for epoch in range(self.num_epochs):

            for data in X:
                d = torch.tensor(data)
                d = torch.autograd.Variable(d)
                d = d.unsqueeze(0)
                d = d.to(device)

                # forward pass
                output = self.model(d.float())
                loss = criterion(output, d.float())

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # log
            loss = loss.item()
            train_loss.append(loss)
            print('Epoch {} of {}, Train Loss: {:.3f}'.format(epoch + 1, self.num_epochs, loss))

