import os
import random
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_or_generate_data(file_normal=None, file_fault=None, points_per_sample=30000, num_samples=30):
    normal_data, fault_data = [], []
    if file_normal is not None and file_fault is not None:
        try:
            df_norm = pd.read_excel(file_normal)
            df_fault = pd.read_excel(file_fault)
            vec_norm = df_norm.select_dtypes(include=[np.number]).values.flatten()
            vec_fault = df_fault.select_dtypes(include=[np.number]).values.flatten()

            num_norm = len(vec_norm) // points_per_sample
            num_fault = len(vec_fault) // points_per_sample
            for i in range(num_norm):
                normal_data.append(vec_norm[i * points_per_sample: (i + 1) * points_per_sample])
            for i in range(num_fault):
                fault_data.append(vec_fault[i * points_per_sample: (i + 1) * points_per_sample])
        except Exception:
            normal_data, fault_data = [], []

    if len(normal_data) == 0 or len(fault_data) == 0:
        t = np.linspace(0, 0.6, points_per_sample, endpoint=False)
        for _ in range(num_samples):
            noise_amplitude = np.random.uniform(0.04, 0.08)
            white_noise = np.random.normal(0, noise_amplitude, points_per_sample)
            grid_hum = 0.05 * np.sin(2 * np.pi * 50 * t + np.random.uniform(0, 2 * np.pi))

            t_impact = np.random.uniform(0.045, 0.055)
            dt = np.maximum(0.0, t - t_impact)
            main_impact = 1.5 * np.sin(2 * np.pi * 120 * dt) * np.exp(-18 * dt) * (t >= t_impact)
            harmonics = 0.6 * np.sin(2 * np.pi * 2400 * dt) * np.exp(-35 * dt) * (t >= t_impact)
            normal_data.append(main_impact + harmonics + white_noise + grid_hum)

            t_impact_f = np.random.uniform(0.09, 0.12)
            dt_f = np.maximum(0.0, t - t_impact_f)
            main_f = 0.9 * np.sin(2 * np.pi * 95 * dt_f) * np.exp(-10 * dt_f) * (t >= t_impact_f)
            fric_1 = 0.7 * np.sin(2 * np.pi * 1850 * dt_f) * np.exp(-7 * dt_f) * (t >= t_impact_f)
            fric_2 = 0.45 * np.sin(2 * np.pi * 3100 * dt_f) * np.exp(-9 * dt_f) * (t >= t_impact_f)
            fault_data.append(main_f + fric_1 + fric_2 + white_noise + grid_hum)

    return np.array(normal_data, dtype=np.float32), np.array(fault_data, dtype=np.float32)


def slice_partition(signals, labels, window_size=1000, stride=300):
    x_list, y_list = [], []
    for sig, label in zip(signals, labels):
        num_windows = (len(sig) - window_size) // stride + 1
        for i in range(num_windows):
            start = i * stride
            segment = sig[start:start + window_size]
            seg_mean = np.mean(segment)
            seg_std = np.std(segment) + 1e-7
            x_list.append((segment - seg_mean) / seg_std)
            y_list.append(label)
    X = np.expand_dims(np.array(x_list, dtype=np.float32), axis=1)
    y = np.array(y_list, dtype=np.int64)
    return X, y


class RigorousVQCLayer(nn.Module):
    def __init__(self, num_qubits=4, num_layers=2):
        super(RigorousVQCLayer, self).__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.var_params = nn.Parameter(torch.randn(num_layers, num_qubits, 2) * 0.1)
        cnot = torch.zeros((2, 2, 2, 2), dtype=torch.complex64)
        cnot[0, 0, 0, 0] = 1.0
        cnot[0, 1, 0, 1] = 1.0
        cnot[1, 1, 1, 0] = 1.0
        cnot[1, 0, 1, 1] = 1.0
        self.register_buffer('cnot', cnot)

    def forward(self, x_classical):
        batch_size = x_classical.size(0)
        theta_enc = torch.tanh(x_classical) * np.pi
        psi = torch.zeros((batch_size, 2, 2, 2, 2), dtype=torch.complex64, device=x_classical.device)
        psi[:, 0, 0, 0, 0] = 1.0 + 0.0j

        for l in range(self.num_layers):
            for q in range(self.num_qubits):
                ry_angle = theta_enc[:, q] + self.var_params[l, q, 0]
                rz_angle = self.var_params[l, q, 1]
                cos_y, sin_y = torch.cos(ry_angle / 2.0), torch.sin(ry_angle / 2.0)
                exp_z0, exp_z1 = torch.exp(-1j * (rz_angle / 2.0)), torch.exp(1j * (rz_angle / 2.0))

                U = torch.stack([
                    torch.stack([cos_y * exp_z0, -sin_y * exp_z0], dim=-1),
                    torch.stack([sin_y * exp_z1, cos_y * exp_z1], dim=-1)
                ], dim=-2).to(torch.complex64)

                if q == 0:
                    psi = torch.einsum('bij,bjklm->biklm', U, psi)
                elif q == 1:
                    psi = torch.einsum('bij,bkjlm->bkilm', U, psi)
                elif q == 2:
                    psi = torch.einsum('bij,bkljm->bklim', U, psi)
                elif q == 3:
                    psi = torch.einsum('bij,bklmj->bklmi', U, psi)

            psi = torch.einsum('ijxy,bxykl->bijkl', self.cnot, psi)
            psi = torch.einsum('ijxy,bkxyl->bkijl', self.cnot, psi)
            psi = torch.einsum('ijxy,bklxy->bklij', self.cnot, psi)

        prob_density = torch.real(psi * torch.conj(psi))
        exp_z0 = prob_density[:, 0, :, :, :].sum(dim=(1, 2, 3)) - prob_density[:, 1, :, :, :].sum(dim=(1, 2, 3))
        exp_z1 = prob_density[:, :, 0, :, :].sum(dim=(1, 2, 3)) - prob_density[:, :, 1, :, :].sum(dim=(1, 2, 3))
        exp_z2 = prob_density[:, :, :, 0, :].sum(dim=(1, 2, 3)) - prob_density[:, :, :, 1, :].sum(dim=(1, 2, 3))
        exp_z3 = prob_density[:, :, :, :, 0].sum(dim=(1, 2, 3)) - prob_density[:, :, :, :, 1].sum(dim=(1, 2, 3))
        return torch.stack([exp_z0, exp_z1, exp_z2, exp_z3], dim=1)


class VQC_QNetwork(nn.Module):
    def __init__(self, input_channels=1, num_actions=2):
        super(VQC_QNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16), nn.LeakyReLU(0.1), nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.LeakyReLU(0.1), nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.1), nn.AdaptiveAvgPool1d(16)
        )
        self.fc_to_quantum = nn.Sequential(
            nn.Linear(64 * 16, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 4)
        )
        self.vqc = RigorousVQCLayer(num_qubits=4, num_layers=2)
        self.q_head = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, num_actions)
        )

    def forward(self, x):
        feat = self.feature_extractor(x).view(x.size(0), -1)
        q_angles = self.fc_to_quantum(feat)
        quantum_out = self.vqc(q_angles)
        return self.q_head(quantum_out)


class ReplayBuffer:
    def __init__(self, capacity=8000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)