import os
import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from models import ForceVAE, TorqueVAE, force_loss, torque_loss
import joblib

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
    "physics_weight": 0.05,
    "smoothness_weight": 0.000005,
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
    "physics_weight": 0.05,
    "smoothness_weight": 0.000005,
    "validation_ratio": 0.2,
    "optuna_trials": 500,
    "optuna_epochs": 200,
    "use_standard_scaler": True,
}

for folder in ["force", "torque", "generated", "metrics", "plots"]:
    os.makedirs(os.path.join(CONFIG_FORCE["output_dir"], folder), exist_ok=True)

def pad_or_truncate(arr, seq_length):
    if len(arr) < seq_length:
        return np.pad(arr, ((0, seq_length - len(arr)), (0, 0)), mode='constant')
    return arr[:seq_length]

def load_data(config, is_force=True):
    data_train, data_val, scalers = [], [], []
    cols = config["force_cols"] if is_force else config["torque_cols"]
    
    for task in range(1, config["tasks"]+1):
        real_path = os.path.join(config["data_root"], "real", f"Task{task:02d}_real.csv")
        sim_path = os.path.join(config["data_root"], "simulated", f"Task{task:02d}_sim.csv")
        
        real_data = pd.read_csv(real_path, header=None).iloc[:, cols].values
        sim_data = pd.read_csv(sim_path, header=None).iloc[:, cols].values
        
        padded_real = pad_or_truncate(real_data, config["seq_length"])
        train_real, val_real = train_test_split(padded_real, test_size=config["validation_ratio"], random_state=42)
        
        scaler = StandardScaler() if config["use_standard_scaler"] else MinMaxScaler()
        scaler.fit(train_real)
        scalers.append(scaler)
        
        data_train.append(scaler.transform(train_real))
        data_val.append(scaler.transform(val_real))
        
    return np.array(data_train), np.array(data_val), scalers

def train_optuna(trial, config, is_force=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data, _ = load_data(config, is_force)
    
    latent_dim = trial.suggest_int('latent_dim', 16, 256)
    hidden_dim_lstm = trial.suggest_int('hidden_dim_lstm', 64, 512)
    lr = trial.suggest_float('learning_rate', 1e-6, 5e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)

    model = ForceVAE(3, latent_dim, hidden_dim_lstm, config["seq_length"]).to(device) if is_force else \
            TorqueVAE(3, latent_dim, hidden_dim_lstm, config["seq_length"]).to(device)
    
    model.encoder.dropout = dropout_rate
    model.decoder.dropout = dropout_rate
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    for epoch in range(config["optuna_epochs"]):
        model.train()
        train_loss = 0
        for i in range(0, len(train_data), config["batch_size"]):
            batch = torch.FloatTensor(train_data[i:i+config["batch_size"]]).to(device)
            recon, mu, logvar = model(batch)
            loss = force_loss(recon, batch, mu, logvar, config) if is_force else \
                   torque_loss(recon, batch, mu, logvar, config)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            train_loss += loss.item()
            optimizer.step()
        
        # Validation loop
        # ... (same as original)

    return best_loss

def train_final(config, is_force=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data, scalers = load_data(config, is_force)
    
    model = ForceVAE(3, config["latent_dim"], config["hidden_dim_lstm"], config["seq_length"]).to(device) if is_force else \
            TorqueVAE(3, config["latent_dim"], config["hidden_dim_lstm"], config["seq_length"]).to(device)
    
    # ... rest of original training logic
    
    # Save scalers
    for task_idx, scaler in enumerate(scalers):
        joblib.dump(scaler, os.path.join(config["output_dir"], 
                    "force" if is_force else "torque", 
                    f"task_{task_idx+1}_scaler.pkl"))
    
    return model, scalers

if __name__ == "__main__":
    # Original main execution logic
    # ... (same as original)