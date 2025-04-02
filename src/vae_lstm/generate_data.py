import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from train import CONFIG_FORCE, CONFIG_TORQUE, pad_or_truncate
from models import ForceVAE, TorqueVAE
import joblib

def generate_data(force_model, torque_model, device):
    force_scalers = [joblib.load(os.path.join(CONFIG_FORCE["output_dir"], "force", f"task_{i}_scaler.pkl")) 
                    for i in range(1, 11)]
    torque_scalers = [joblib.load(os.path.join(CONFIG_TORQUE["output_dir"], "torque", f"task_{i}_scaler.pkl")) 
                     for i in range(1, 11)]

    # ... rest of original generate_data logic

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    force_model = ForceVAE(3, CONFIG_FORCE["latent_dim"], 
                         CONFIG_FORCE["hidden_dim_lstm"], 
                         CONFIG_FORCE["seq_length"])
    force_model.load_state_dict(torch.load(
        os.path.join(CONFIG_FORCE["output_dir"], "force", "best_model.pth")))
    
    torque_model = TorqueVAE(3, CONFIG_TORQUE["latent_dim"],
                           CONFIG_TORQUE["hidden_dim_lstm"],
                           CONFIG_TORQUE["seq_length"])
    torque_model.load_state_dict(torch.load(
        os.path.join(CONFIG_TORQUE["output_dir"], "torque", "best_model.pth")))
    
    generate_data(force_model, torque_model, device)