import torch
import torch.nn as nn
import torch.nn.functional as F

class ForceVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, seq_length):
        super().__init__()
        self.seq_length = seq_length
        self.encoder = nn.LSTM(input_dim, hidden_dim, 3, batch_first=True, dropout=0.2)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.LSTM(latent_dim, hidden_dim, 3, batch_first=True, dropout=0.2)
        self.output = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
        mu = self.fc_mu(h_n[-1])
        logvar = self.fc_logvar(h_n[-1])
        z = self.reparameterize(mu, logvar)
        z_repeated = z.unsqueeze(1).repeat(1, x.size(1), 1)
        output, _ = self.decoder(z_repeated)
        return self.output(output), mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + std * torch.randn_like(std)

class TorqueVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, seq_length):
        super().__init__()
        self.seq_length = seq_length
        self.encoder = nn.LSTM(input_dim, hidden_dim, 3, batch_first=True, dropout=0.1)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.LSTM(latent_dim, hidden_dim, 3, batch_first=True, dropout=0.1)
        self.output = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
        mu = self.fc_mu(h_n[-1])
        logvar = self.fc_logvar(h_n[-1])
        z = self.reparameterize(mu, logvar)
        z_repeated = z.unsqueeze(1).repeat(1, x.size(1), 1)
        output, _ = self.decoder(z_repeated)
        return self.output(output), mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + std * torch.randn_like(std)

def force_loss(recon, target, mu, logvar, config):
    seq_len = target.shape[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    BCE = F.mse_loss(recon[:, :seq_len], target[:, :seq_len], reduction='sum')
    smoothness = F.mse_loss(recon[:, :-1], recon[:, 1:], reduction='sum')
    physics = torch.mean(torch.relu(torch.norm(recon[:, :seq_len], dim=2) - 10))
    return config["reconstruction_weight"]*BCE + config["physics_weight"]*physics + config["smoothness_weight"]*smoothness + config["reconstruction_weight"]*KLD

def torque_loss(recon, target, mu, logvar, config):
    seq_len = target.shape[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    BCE = F.mse_loss(recon[:, :seq_len], target[:, :seq_len], reduction='sum')
    smoothness = F.mse_loss(recon[:, :-1], recon[:, 1:], reduction='sum')
    physics = torch.mean(torch.relu(torch.norm(recon[:, :seq_len], dim=2) - 2))
    return config["reconstruction_weight"]*BCE + config["physics_weight"]*physics + config["smoothness_weight"]*smoothness + config["reconstruction_weight"]*KLD