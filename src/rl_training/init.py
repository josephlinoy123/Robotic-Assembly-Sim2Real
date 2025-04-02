import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import optuna

# ---------------- Configuration ---------------- #
CONFIG_FORCE = {
    "data_root": "C:/Users/josep/robot/data",
    "output_dir": os.path.join("C:/Users/josep/robot/data", "output_split_models"),
    "tasks": 10,
    "force_cols": list(range(16, 19)),
    "seq_length": 100,
    "epochs": 300,
    "batch_size": 10,
    "latent_dim": 64,
    "hidden_dim_lstm": 128,
    "learning_rate": 1e-3,
    "early_stop_patience": 50,
    "reconstruction_weight": 1.0,
    "physics_weight": 0.01,  # Reduced physics weight
    "smoothness_weight": 0.000001,  # Reduced smoothness weight
    "validation_ratio": 0.2,
    "optuna_trials": 50,
    "optuna_epochs": 50,
    "use_standard_scaler": False,
}

CONFIG_TORQUE = {
    "data_root": "C:/Users/josep/robot/data",
    "output_dir": os.path.join("C:/Users/josep/robot/data", "output_split_models"),
    "tasks": 10,
    "torque_cols": list(range(19, 22)),
    "seq_length": 100,
    "epochs": 300,
    "batch_size": 10,
    "latent_dim": 64,
    "hidden_dim_lstm": 256,
    "learning_rate": 5e-4,
    "early_stop_patience": 50,
    "reconstruction_weight": 1.0,
    "physics_weight": 0.01,  # Reduced physics weight
    "smoothness_weight": 0.000001,  # Reduced smoothness weight
    "validation_ratio": 0.2,
    "optuna_trials": 1000,
    "optuna_epochs": 300,
    "use_standard_scaler": True,
}

# Create directories
for folder in ["force", "torque", "generated", "metrics", "plots"]:
    os.makedirs(os.path.join(CONFIG_FORCE["output_dir"], folder), exist_ok=True)

# ---------------- Data Loading & Preprocessing ---------------- #
def load_data(config, is_force=True):
    data_train, data_val, scalers = [], [], []
    cols = config["force_cols"] if is_force else config["torque_cols"]
    
    for task in range(1, config["tasks"] + 1):
        # Load real data
        real_path = os.path.join(config["data_root"], "real", f"Task{task:02d}_real.csv")
        real_df = pd.read_csv(real_path, header=None)
        real_data = real_df.iloc[:, cols].values
        
        # Load simulated data
        sim_path = os.path.join(config["data_root"], "simulated", f"Task{task:02d}_sim.csv")
        sim_df = pd.read_csv(sim_path, header=None)
        sim_data = sim_df.iloc[:, cols].values
        
        # Pad sequences
        padded_real = pad_or_truncate(real_data, config["seq_length"])
        padded_sim = pad_or_truncate(sim_data, config["seq_length"])
        
        # Combine real and simulated data
        combined_data = np.vstack((padded_real, padded_sim))
        
        # Split data
        train_data, val_data = train_test_split(
            combined_data, test_size=config["validation_ratio"], random_state=42
        )
        
        # Scaler - Fit on training data only
        if config["use_standard_scaler"]:
            scaler = StandardScaler().fit(train_data)
        else:
            scaler = MinMaxScaler().fit(train_data)
        scalers.append(scaler)
        
        # Normalize - Transform both train and val
        train_data_norm = scaler.transform(train_data)
        val_data_norm = scaler.transform(val_data)
        
        data_train.append(train_data_norm)
        data_val.append(val_data_norm)
        
    return np.array(data_train), np.array(data_val), scalers

# ---------------- Pad/Truncate Function ---------------- #
def pad_or_truncate(arr, seq_length):
    """Pads or truncates a 2D array along the first axis (time) to seq_length."""
    if len(arr) < seq_length:
        return np.pad(arr, ((0, seq_length - len(arr)), (0, 0)), mode='constant')
    elif len(arr) > seq_length:
        return arr[:seq_length]
    else:
        return arr

# ---------------- LSTM-VAE Models ---------------- #
class ForceVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, seq_length):
        super().__init__()
        self.seq_length = seq_length
        self.encoder = nn.LSTM(input_dim, hidden_dim, 4, batch_first=True, dropout=0.2)  # Increased layers
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.LSTM(latent_dim, hidden_dim, 4, batch_first=True, dropout=0.2)  # Increased layers
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
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

class TorqueVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, seq_length):
        super().__init__()
        self.seq_length = seq_length
        self.encoder = nn.LSTM(input_dim, hidden_dim, 4, batch_first=True, dropout=0.1)  # Increased layers
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.LSTM(latent_dim, hidden_dim, 4, batch_first=True, dropout=0.1)  # Increased layers
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
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

# ---------------- Loss Functions ---------------- #
def force_loss(recon, target, mu, logvar, config):
    seq_len = target.shape[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    BCE = F.mse_loss(recon[:, :seq_len], target[:, :seq_len], reduction='sum')
    smoothness = F.mse_loss(recon[:, :-1], recon[:, 1:], reduction='sum')
    physics = torch.mean(torch.relu(torch.norm(recon[:, :seq_len], dim=2) - 10))
    return config["reconstruction_weight"] * BCE + config["physics_weight"] * physics + config["smoothness_weight"] * smoothness + KLD

def torque_loss(recon, target, mu, logvar, config):
    seq_len = target.shape[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    BCE = F.mse_loss(recon[:, :seq_len], target[:, :seq_len], reduction='sum')
    smoothness = F.mse_loss(recon[:, :-1], recon[:, 1:], reduction='sum')
    physics = torch.mean(torch.relu(torch.norm(recon[:, :seq_len], dim=2) - 2))
    return config["reconstruction_weight"] * BCE + config["physics_weight"] * physics + config["smoothness_weight"] * smoothness + KLD

# ---------------- Training Functions ---------------- #
def train_optuna(trial, config, is_force=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data, _ = load_data(config, is_force)

    model_type = ForceVAE if is_force else TorqueVAE

    # --- Hyperparameter Suggestions ---
    latent_dim = trial.suggest_int('latent_dim', 32, 128)
    hidden_dim_lstm = trial.suggest_int('hidden_dim_lstm', 128, 256)
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)

    # --- Loss Function Adjustments (for Torque) ---
    if not is_force:
        temp_config = config.copy()
        temp_config["physics_weight"] = trial.suggest_float("physics_weight", 0.001, 1.0, log=True)
        temp_config["smoothness_weight"] = trial.suggest_float("smoothness_weight", 1e-7, 1e-4, log=True)

        def adjusted_torque_loss(recon, target, mu, logvar):
            return torque_loss(recon, target, mu, logvar, temp_config)

        loss_func = adjusted_torque_loss
    else:
        loss_func = lambda recon, target, mu, logvar: force_loss(recon, target, mu, logvar, config)

    # --- Model Creation ---
    model = model_type(3, latent_dim, hidden_dim_lstm, config["seq_length"]).to(device)
    model.encoder.dropout = dropout_rate
    model.decoder.dropout = dropout_rate

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    patience = 0

    for epoch in range(config["optuna_epochs"]):
        model.train()
        train_loss = 0
        for i in range(0, len(train_data), config["batch_size"]):
            batch = torch.FloatTensor(train_data[i:i+config["batch_size"]]).to(device)
            recon, mu, logvar = model(batch)
            loss = loss_func(recon, batch, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            train_loss += loss.item()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_data), config["batch_size"]):
                batch = torch.FloatTensor(val_data[i:i+config["batch_size"]]).to(device)
                recon, mu, logvar = model(batch)
                val_loss += loss_func(recon, batch, mu, logvar).item()

        avg_val_loss = val_loss / len(val_data)
        trial.report(avg_val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience = 0
        else:
            patience += 1
            if patience >= config["early_stop_patience"]:
                break

    return best_loss

def train_final(config, is_force=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data, scalers = load_data(config, is_force)
    model_type = ForceVAE if is_force else TorqueVAE
    loss_func = force_loss if is_force else torque_loss

    model = model_type(
        input_dim=3,
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim_lstm"],
        seq_length=config["seq_length"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)  # L2 regularization
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # Learning rate scheduler

    best_loss = float('inf')
    patience = 0
    train_losses = []
    val_losses = []

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0
        for i in range(0, len(train_data), config["batch_size"]):
            batch = torch.FloatTensor(train_data[i:i+config["batch_size"]]).to(device)
            recon, mu, logvar = model(batch)
            loss = loss_func(recon, batch, mu, logvar, config)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            train_loss += loss.item()
            optimizer.step()

        avg_train_loss = train_loss / len(train_data)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_data), config["batch_size"]):
                batch = torch.FloatTensor(val_data[i:i+config["batch_size"]]).to(device)
                recon, mu, logvar = model(batch)
                val_loss += loss_func(recon, batch, mu, logvar, config).item()

        avg_val_loss = val_loss / len(val_data)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience = 0
            torch.save(model.state_dict(), os.path.join(config["output_dir"], "force" if is_force else "torque", f"best_model_task_{epoch}.pth"))
        else:
            patience += 1
            if patience >= config["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step()  # Update learning rate

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", color="blue", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", color="red", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"{'Force' if is_force else 'Torque'} Model Learning Curves", fontsize=14)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(config["output_dir"], "plots", f"{'force' if is_force else 'torque'}_learning_curves.png"), dpi=300)
    plt.close()

    return model, scalers

# ---------------- Main Execution ---------------- #
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Force Model ---
    print("Starting Force Model Training...")
    force_study = optuna.create_study(direction='minimize')
    force_study.optimize(lambda t: train_optuna(t, CONFIG_FORCE, is_force=True), n_trials=CONFIG_FORCE["optuna_trials"])
    print("\nBest Force Model Hyperparameters:")
    print(force_study.best_params)
    print(f"Best Force Model Validation Loss: {force_study.best_value}")
    CONFIG_FORCE.update(force_study.best_params)
    force_model, force_scalers = train_final(CONFIG_FORCE, is_force=True)
    print("Force Model Training Complete.")

    # --- Torque Model ---
    print("Training Torque Model...")
    torque_study = optuna.create_study(direction='minimize')
    torque_study.optimize(lambda t: train_optuna(t, CONFIG_TORQUE, is_force=False), n_trials=CONFIG_TORQUE["optuna_trials"])
    print("\nBest Torque Model Hyperparameters:")
    print(torque_study.best_params)
    print(f"Best Torque Model Validation Loss: {torque_study.best_value}")
    CONFIG_TORQUE.update(torque_study.best_params)
    torque_model, torque_scalers = train_final(CONFIG_TORQUE, is_force=False)
    print("Torque Model Training Complete.")

    print("Starting Data Generation...")
    try:
        generate_data(force_model, torque_model, force_scalers, torque_scalers, device)
        print("Data generation, metrics, and plots created successfully!")
    except Exception as e:
        print(f"Error during data generation: {e}")
    print("Data Generation Process Finished (with or without errors).")

    print("Training and data generation complete! Check 'generated', 'metrics', and 'plots' folders for outputs.")