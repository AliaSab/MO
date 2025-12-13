"""
Модуль для построения моделей глубокого обучения для прогнозирования временных рядов.
Включает: MLP, TCN, N-BEATS, N-HiTS, RNN, LSTM, GRU, трансформеры, гибриды, бейзлайны.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


# ==================== Базовые модели ====================

class MLP(nn.Module):
    """Многослойный перцептрон для временных рядов."""
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], horizon=48, dropout=0.2):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, horizon))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, seq_len * features)
        batch_size = x.size(0)
        seq_len = x.size(1)
        features = x.size(2) if x.dim() == 3 else 1
        x = x.view(batch_size, seq_len * features)
        return self.network(x)


class TCNBlock(nn.Module):
    """Temporal Convolutional Network блок."""
    
    def __init__(self, num_channels, kernel_size=2, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, num_channels, kernel_size, 
                              padding=(kernel_size - 1), dilation=1)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size,
                              padding=(kernel_size - 1), dilation=1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        return out + x[:, :, :out.size(2)]


class TCN(nn.Module):
    """Temporal Convolutional Network."""
    
    def __init__(self, input_size, num_channels=[64, 128, 256], kernel_size=2, 
                 dropout=0.2, horizon=48):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size,
                                    padding=(kernel_size - 1) * dilation_size, 
                                    dilation=dilation_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], horizon)
        
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        out = self.tcn(x)
        # Берем последний временной шаг
        out = out[:, :, -1]
        return self.linear(out)


# ==================== N-BEATS ====================

class NBeatsBlock(nn.Module):
    """Блок N-BEATS."""
    
    def __init__(self, input_size, theta_size, basis_function, 
                 num_layers=4, layer_size=512, dropout=0.1):
        super(NBeatsBlock, self).__init__()
        self.basis_function = basis_function
        
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_size if i == 0 else layer_size, layer_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        self.layers = nn.Sequential(*layers)
        self.theta_f = nn.Linear(layer_size, theta_size)
        self.theta_b = nn.Linear(layer_size, theta_size)
        
    def forward(self, x):
        x = self.layers(x)
        theta_f = self.theta_f(x)
        theta_b = self.theta_b(x)
        return theta_f, theta_b


class TrendBasis(nn.Module):
    """Базисная функция для тренда."""
    
    def __init__(self, degree=2, backcast_size=336, forecast_size=48):
        super(TrendBasis, self).__init__()
        self.degree = degree
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        
    def forward(self, theta):
        t = torch.arange(self.backcast_size, dtype=theta.dtype, device=theta.device)
        t = t / self.backcast_size
        
        backcast = torch.zeros(theta.size(0), self.backcast_size, device=theta.device)
        for i in range(self.degree + 1):
            c = theta[:, i:i+1]
            backcast += c * (t ** i)
        
        t_f = torch.arange(self.forecast_size, dtype=theta.dtype, device=theta.device)
        t_f = t_f / self.forecast_size
        
        forecast = torch.zeros(theta.size(0), self.forecast_size, device=theta.device)
        for i in range(self.degree + 1):
            c = theta[:, i:i+1]
            forecast += c * (t_f ** i)
        
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    """Базисная функция для сезонности."""
    
    def __init__(self, harmonics=1, backcast_size=336, forecast_size=48):
        super(SeasonalityBasis, self).__init__()
        self.harmonics = harmonics
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        
    def forward(self, theta):
        period = self.backcast_size / self.harmonics
        
        t = torch.arange(self.backcast_size, dtype=theta.dtype, device=theta.device)
        t = 2 * np.pi * t / period
        
        backcast = torch.zeros(theta.size(0), self.backcast_size, device=theta.device)
        for i in range(self.harmonics):
            backcast += theta[:, 2*i:2*i+1] * torch.sin((i + 1) * t)
            backcast += theta[:, 2*i+1:2*i+2] * torch.cos((i + 1) * t)
        
        t_f = torch.arange(self.forecast_size, dtype=theta.dtype, device=theta.device)
        t_f = 2 * np.pi * t_f / period
        
        forecast = torch.zeros(theta.size(0), self.forecast_size, device=theta.device)
        for i in range(self.harmonics):
            forecast += theta[:, 2*i:2*i+1] * torch.sin((i + 1) * t_f)
            forecast += theta[:, 2*i+1:2*i+2] * torch.cos((i + 1) * t_f)
        
        return backcast, forecast


class NBeats(nn.Module):
    """N-BEATS модель."""
    
    def __init__(self, input_size=336, horizon=48, stack_types=['trend', 'seasonality'],
                 num_blocks=3, num_layers=4, layer_size=512, dropout=0.1, lookback=None):
        super(NBeats, self).__init__()
        self.input_size = lookback if lookback else input_size
        self.horizon = horizon
        self.stack_types = stack_types
        self.num_blocks = num_blocks
        
        self.blocks = nn.ModuleList()
        
        for stack_type in stack_types:
            for _ in range(num_blocks):
                if stack_type == 'trend':
                    basis = TrendBasis(degree=2, backcast_size=self.input_size, forecast_size=horizon)
                    theta_size = 3  # degree + 1
                elif stack_type == 'seasonality':
                    basis = SeasonalityBasis(harmonics=1, backcast_size=self.input_size, forecast_size=horizon)
                    theta_size = 2  # 2 * harmonics
                else:
                    raise ValueError(f"Unknown stack type: {stack_type}")
                
                block = NBeatsBlock(self.input_size, theta_size, basis, num_layers, layer_size, dropout)
                self.blocks.append(block)
        
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, seq_len)
        if x.dim() == 3:
            x = x[:, :, 0]  # Берем первый признак
        
        seq_len = x.size(1)
        
        # Обрабатываем размер последовательности
        if seq_len > self.input_size:
            # Обрезаем до нужной длины
            x = x[:, -self.input_size:]
        elif seq_len < self.input_size:
            # Дополняем нулями слева
            pad_size = self.input_size - seq_len
            x = torch.nn.functional.pad(x, (pad_size, 0), mode='constant', value=0)
        
        forecast = torch.zeros(x.size(0), self.horizon, device=x.device)
        
        for block in self.blocks:
            theta_f, theta_b = block(x)
            backcast, block_forecast = block.basis_function(theta_f)
            forecast += block_forecast
            x = x - backcast
        
        return forecast


class NHiTS(nn.Module):
    """N-HiTS: Neural Hierarchical Interpolation для временных рядов (упрощенная версия)."""
    
    def __init__(self, input_size=336, horizon=48, num_stacks=3, num_blocks=2,
                 num_layers=4, layer_size=512, dropout=0.1, pool_kernel_sizes=[2, 2, 2], lookback=None):
        super(NHiTS, self).__init__()
        self.input_size = lookback if lookback else input_size
        self.horizon = horizon
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.pool_kernel_sizes = pool_kernel_sizes
        
        self.stacks = nn.ModuleList()
        
        for stack_idx in range(num_stacks):
            stack_blocks = nn.ModuleList()
            pool_size = pool_kernel_sizes[stack_idx] if stack_idx < len(pool_kernel_sizes) else 2
            
            for _ in range(num_blocks):
                # Упрощенный блок N-HiTS
                if stack_idx == 0:
                    basis = TrendBasis(degree=2, backcast_size=self.input_size, forecast_size=horizon)
                    theta_size = 3
                else:
                    basis = SeasonalityBasis(harmonics=1, backcast_size=self.input_size, forecast_size=horizon)
                    theta_size = 2
                
                block = NBeatsBlock(self.input_size, theta_size, basis, num_layers, layer_size, dropout)
                stack_blocks.append(block)
            
            self.stacks.append(stack_blocks)
        
    def forward(self, x):
        if x.dim() == 3:
            x = x[:, :, 0]
        
        seq_len = x.size(1)
        
        # Обрабатываем размер последовательности
        if seq_len > self.input_size:
            # Обрезаем до нужной длины
            x = x[:, -self.input_size:]
        elif seq_len < self.input_size:
            # Дополняем нулями слева
            pad_size = self.input_size - seq_len
            x = torch.nn.functional.pad(x, (pad_size, 0), mode='constant', value=0)
        
        forecast = torch.zeros(x.size(0), self.horizon, device=x.device)
        
        for stack in self.stacks:
            stack_forecast = torch.zeros(x.size(0), self.horizon, device=x.device)
            residual = x.clone()
            
            for block in stack:
                theta_f, theta_b = block(residual)
                backcast, block_forecast = block.basis_function(theta_f)
                stack_forecast += block_forecast
                residual = residual - backcast
            
            forecast += stack_forecast
            x = residual
        
        return forecast


# ==================== Рекуррентные модели ====================

class RNN(nn.Module):
    """Простая RNN."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, horizon=48):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size, horizon)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Берем последний скрытый слой
        return self.linear(out)


class LSTM(nn.Module):
    """LSTM модель."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, horizon=48):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size, horizon)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.linear(out)


class GRU(nn.Module):
    """GRU модель."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, horizon=48):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size, horizon)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.linear(out)


class BiLSTM(nn.Module):
    """Двунаправленный LSTM."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, horizon=48):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, horizon)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.linear(out)


class BiGRU(nn.Module):
    """Двунаправленный GRU."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, horizon=48):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0,
                         bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, horizon)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.linear(out)


# ==================== Трансформеры ====================

class PositionalEncoding(nn.Module):
    """Positional encoding для трансформеров."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    """Transformer модель для временных рядов."""
    
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3,
                 dim_feedforward=512, dropout=0.1, horizon=48):
        super(Transformer, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder = nn.Linear(d_model, horizon)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        batch_size = x.size(0)
        x = x.transpose(0, 1)  # (seq_len, batch, features)
        x = self.input_projection(x)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        x = x[:, -1, :]  # Берем последний шаг
        return self.decoder(x)


# ==================== Гибридные модели ====================

class CNNLSTM(nn.Module):
    """Гибридная CNN-LSTM модель."""
    
    def __init__(self, input_size, cnn_filters=64, lstm_hidden=128, 
                 num_layers=2, dropout=0.2, horizon=48):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(cnn_filters, lstm_hidden, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(lstm_hidden, horizon)
        
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        # (batch, filters, seq_len) -> (batch, seq_len, filters)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.linear(out)


class CNNGRU(nn.Module):
    """Гибридная CNN-GRU модель."""
    
    def __init__(self, input_size, cnn_filters=64, gru_hidden=128,
                 num_layers=2, dropout=0.2, horizon=48):
        super(CNNGRU, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gru = nn.GRU(cnn_filters, gru_hidden, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(gru_hidden, horizon)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.linear(out)


# ==================== Бейзлайны ====================

class DLinear(nn.Module):
    """DLinear: простой линейный слой с декомпозицией."""
    
    def __init__(self, input_size, horizon=48, moving_avg=25, lookback=None):
        super(DLinear, self).__init__()
        self.lookback = lookback if lookback else input_size
        self.moving_avg = min(moving_avg, self.lookback)
        # Используем padding='same' для сохранения размера
        self.decomposition = nn.AvgPool1d(kernel_size=self.moving_avg, stride=1, padding=self.moving_avg//2)
        self.linear_season = nn.Linear(self.lookback, horizon)
        self.linear_trend = nn.Linear(self.lookback, horizon)
        
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, seq_len)
        if x.dim() == 3:
            # Берем первый признак и обрезаем до lookback
            x_seq = x[:, :, 0]  # (batch, seq_len)
            # Обрезаем до нужной длины
            if x_seq.size(1) > self.lookback:
                x_seq = x_seq[:, -self.lookback:]
            elif x_seq.size(1) < self.lookback:
                # Дополняем нулями слева
                pad_size = self.lookback - x_seq.size(1)
                x_seq = torch.nn.functional.pad(x_seq, (pad_size, 0), mode='constant', value=0)
            x = x_seq.unsqueeze(1)  # (batch, 1, lookback)
        elif x.dim() == 2:
            # Обрезаем до нужной длины
            if x.size(1) > self.lookback:
                x = x[:, -self.lookback:]
            elif x.size(1) < self.lookback:
                pad_size = self.lookback - x.size(1)
                x = torch.nn.functional.pad(x, (pad_size, 0), mode='constant', value=0)
            x = x.unsqueeze(1)  # (batch, 1, lookback)
        
        # Теперь x имеет размер (batch, 1, lookback)
        # Декомпозиция
        x_trend = self.decomposition(x)  # (batch, 1, lookback) - размер сохраняется благодаря padding
        x_season = x - x_trend
        
        # Обрезаем до lookback на случай, если padding дал больше
        x_trend = x_trend[:, :, -self.lookback:]  # (batch, 1, lookback)
        x_season = x_season[:, :, -self.lookback:]  # (batch, 1, lookback)
        
        x_trend = x_trend.squeeze(1)  # (batch, lookback)
        x_season = x_season.squeeze(1)  # (batch, lookback)
        
        trend = self.linear_trend(x_trend)
        season = self.linear_season(x_season)
        
        return trend + season


class NLinear(nn.Module):
    """NLinear: нормализованный линейный слой."""
    
    def __init__(self, input_size, horizon=48, lookback=None):
        super(NLinear, self).__init__()
        self.lookback = lookback if lookback else input_size
        self.linear = nn.Linear(self.lookback, horizon)
        
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, seq_len)
        if x.dim() == 3:
            x = x[:, :, 0]  # Берем первый признак
        
        seq_len = x.size(1)
        
        # Обрезаем до нужной длины, если нужно
        if seq_len != self.lookback:
            x = x[:, -self.lookback:]
        
        # Нормализация: вычитаем последнее значение
        x_last = x[:, -1:]
        x = x - x_last
        pred = self.linear(x)
        return pred + x_last.expand_as(pred)


class Naive(nn.Module):
    """Наивный прогноз: последнее значение."""
    
    def __init__(self, horizon=48):
        super(Naive, self).__init__()
        self.horizon = horizon
        
    def forward(self, x):
        if x.dim() == 3:
            x = x[:, :, 0]
        last_value = x[:, -1:]
        return last_value.expand(-1, self.horizon)


class SeasonalNaive(nn.Module):
    """Сезонный наивный прогноз."""
    
    def __init__(self, season_length=7, horizon=48):
        super(SeasonalNaive, self).__init__()
        self.season_length = season_length
        self.horizon = horizon
        
    def forward(self, x):
        if x.dim() == 3:
            x = x[:, :, 0]
        seasonal_values = x[:, -self.season_length:]
        n_repeats = (self.horizon + self.season_length - 1) // self.season_length
        forecast = seasonal_values.repeat(1, n_repeats)[:, :self.horizon]
        return forecast


# ==================== Продвинутые трансформеры ====================

class ProbSparseAttention(nn.Module):
    """ProbSparse Self-Attention для Informer."""
    
    def __init__(self, d_model, nhead, dropout=0.1, factor=5):
        super(ProbSparseAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.factor = factor
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Проекции Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # Упрощенная версия ProbSparse: выбираем top-k queries
        U = min(self.factor * int(np.log(seq_len)), seq_len)
        
        # Вычисляем attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # Применяем softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Применяем attention к V
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(out)


class Informer(nn.Module):
    """Informer: эффективный трансформер для длинных последовательностей."""
    
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3,
                 dim_feedforward=512, dropout=0.1, horizon=48, factor=5):
        super(Informer, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Encoder layers с ProbSparse Attention
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': ProbSparseAttention(d_model, nhead, dropout, factor),
                'ff': nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model)
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'dropout': nn.Dropout(dropout)
            })
            for _ in range(num_layers)
        ])
        
        self.decoder = nn.Linear(d_model, horizon)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Encoder
        for layer in self.encoder_layers:
            # Self-attention
            attn_out = layer['attn'](x)
            x = layer['norm1'](x + layer['dropout'](attn_out))
            
            # Feed-forward
            ff_out = layer['ff'](x)
            x = layer['norm2'](x + layer['dropout'](ff_out))
        
        # Decoder: берем последний временной шаг
        x = x[:, -1, :]
        return self.decoder(x)


class AutoCorrelation(nn.Module):
    """Auto-Correlation механизм для Autoformer."""
    
    def __init__(self, d_model, nhead, dropout=0.1):
        super(AutoCorrelation, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.d_k)
        K = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.d_k)
        V = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.d_k)
        
        # Упрощенная auto-correlation: используем FFT для нахождения периодичности
        # В полной версии используется FFT, здесь используем упрощенный подход
        Q_fft = torch.fft.rfft(Q, dim=1, norm='ortho')
        K_fft = torch.fft.rfft(K, dim=1, norm='ortho')
        
        # Auto-correlation в частотной области
        corr = Q_fft * torch.conj(K_fft)
        corr = torch.fft.irfft(corr, n=seq_len, dim=1, norm='ortho')
        
        # Нормализация
        corr = F.softmax(corr / np.sqrt(self.d_k), dim=1)
        corr = self.dropout(corr)
        
        # Применяем к V
        out = torch.einsum('bshd,bshd->bshd', corr, V)
        out = out.contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(out)


class Autoformer(nn.Module):
    """Autoformer: с Auto-Correlation и декомпозицией."""
    
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3,
                 dim_feedforward=512, dropout=0.1, horizon=48, moving_avg=25):
        super(Autoformer, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Decomposition
        self.decomp = nn.AvgPool1d(kernel_size=moving_avg, stride=1, padding=moving_avg//2)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': AutoCorrelation(d_model, nhead, dropout),
                'ff': nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model)
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'dropout': nn.Dropout(dropout)
            })
            for _ in range(num_layers)
        ])
        
        self.decoder = nn.Linear(d_model, horizon)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_projection(x)
        
        # Encoder с Auto-Correlation
        for layer in self.encoder_layers:
            # Auto-Correlation
            attn_out = layer['attn'](x)
            x = layer['norm1'](x + layer['dropout'](attn_out))
            
            # Feed-forward
            ff_out = layer['ff'](x)
            x = layer['norm2'](x + layer['dropout'](ff_out))
        
        # Decoder
        x = x[:, -1, :]
        return self.decoder(x)


class PatchTST(nn.Module):
    """PatchTST: Patching механизм для временных рядов."""
    
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3,
                 dim_feedforward=512, dropout=0.1, horizon=48, patch_len=16, stride=8):
        super(PatchTST, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.input_size = input_size
        
        # Projection для каждого patch
        self.patch_proj = nn.Linear(patch_len * input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder = nn.Linear(d_model, horizon)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        batch_size, seq_len, features = x.size()
        
        # Создаем patches
        # Количество patches
        num_patches = (seq_len - self.patch_len) // self.stride + 1
        
        patches = []
        for i in range(num_patches):
            start = i * self.stride
            end = start + self.patch_len
            if end <= seq_len:
                patch = x[:, start:end, :].reshape(batch_size, -1)  # (batch, patch_len * features)
                patches.append(patch)
        
        if len(patches) == 0:
            # Если не удалось создать patches, используем всю последовательность
            patches = [x.reshape(batch_size, -1)]
            # Добавляем padding, если нужно
            if patches[0].size(1) < self.patch_len * features:
                pad_size = self.patch_len * features - patches[0].size(1)
                patches[0] = F.pad(patches[0], (0, pad_size), mode='constant', value=0)
        
        # Stack patches
        x_patched = torch.stack(patches, dim=1)  # (batch, num_patches, patch_len * features)
        
        # Project patches
        x_patched = self.patch_proj(x_patched)  # (batch, num_patches, d_model)
        
        # Transformer encoding
        x_encoded = self.transformer_encoder(x_patched)
        
        # Берем последний patch
        x_last = x_encoded[:, -1, :]
        
        return self.decoder(x_last)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network для TFT."""
    
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(VariableSelectionNetwork, self).__init__()
        self.hidden_size = hidden_size
        
        # GRN для каждой переменной
        self.grn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Softmax для выбора переменных
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        weights = self.grn(x)
        weights = self.softmax(weights)
        return x * weights


class TFT(nn.Module):
    """Temporal Fusion Transformer (упрощенная версия)."""
    
    def __init__(self, input_size, hidden_size=128, nhead=4, num_layers=2,
                 dropout=0.1, horizon=48):
        super(TFT, self).__init__()
        
        # Variable Selection
        self.vsn = VariableSelectionNetwork(input_size, hidden_size, dropout)
        
        # LSTM для обработки последовательности
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Self-Attention
        self.attention = nn.MultiheadAttention(hidden_size, nhead, dropout=dropout, batch_first=True)
        
        # Gating
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Output
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, horizon)
        )
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        
        # Variable Selection
        x_selected = self.vsn(x)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x_selected)
        
        # Self-Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Gated residual
        gate = self.gate(lstm_out)
        x_gated = self.norm1(lstm_out + gate * attn_out)
        
        # Decoder: берем последний временной шаг
        x_last = x_gated[:, -1, :]
        
        return self.decoder(x_last)


# ==================== Гибридные продвинутые модели ====================

class TCNAttention(nn.Module):
    """TCN с Attention механизмом."""
    
    def __init__(self, input_size, num_channels=[64, 128, 256], kernel_size=2,
                 dropout=0.2, horizon=48, nhead=4):
        super(TCNAttention, self).__init__()
        
        # TCN часть
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size,
                                    padding=(kernel_size - 1) * dilation_size, 
                                    dilation=dilation_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        
        # Attention
        self.attention = nn.MultiheadAttention(num_channels[-1], nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(num_channels[-1])
        
        # Output
        self.linear = nn.Linear(num_channels[-1], horizon)
        
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # TCN
        x = self.tcn(x)
        
        # (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Self-Attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        
        # Берем последний временной шаг
        x = x[:, -1, :]
        
        return self.linear(x)


class LSTMAE(nn.Module):
    """LSTM Autoencoder для временных рядов."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, 
                 dropout=0.2, horizon=48, latent_size=64):
        super(LSTMAE, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Latent space
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        
        # Decoder для прогноза
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Output
        self.output = nn.Linear(hidden_size, horizon)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        
        # Encode
        _, (h_n, c_n) = self.encoder(x)
        
        # Берем последнее скрытое состояние
        encoded = h_n[-1]  # (batch, hidden_size)
        
        # Latent representation
        latent = self.to_latent(encoded)
        decoded = self.from_latent(latent)
        
        # Decode для прогноза
        # Используем decoded как начальное состояние
        decoded = decoded.unsqueeze(1)  # (batch, 1, hidden_size)
        decoder_out, _ = self.decoder(decoded)
        
        # Output
        out = decoder_out[:, -1, :]  # Берем последний шаг
        return self.output(out)


# ==================== Фабрика моделей ====================

def create_model(model_name, input_size, horizon=48, lookback=None, **kwargs):
    """
    Создает модель по имени.
    
    Args:
        input_size: количество признаков (features) в данных
        horizon: горизонт прогнозирования
        lookback: длина последовательности (должна быть задана!)
        **kwargs: дополнительные параметры модели
    """
    if lookback is None:
        raise ValueError(f"lookback должен быть задан для модели {model_name}")
    
    # Для MLP input_size должен быть seq_len * features
    mlp_input_size = lookback * input_size
    
    models = {
        'MLP': lambda: MLP(mlp_input_size, horizon=horizon, **kwargs),
        'TCN': lambda: TCN(input_size, horizon=horizon, **kwargs),
        'N-BEATS': lambda: NBeats(input_size=lookback, horizon=horizon, lookback=lookback, **kwargs),
        'N-HiTS': lambda: NHiTS(input_size=lookback, horizon=horizon, lookback=lookback, **kwargs),
        'RNN': lambda: RNN(input_size, horizon=horizon, **kwargs),
        'LSTM': lambda: LSTM(input_size, horizon=horizon, **kwargs),
        'GRU': lambda: GRU(input_size, horizon=horizon, **kwargs),
        'BiLSTM': lambda: BiLSTM(input_size, horizon=horizon, **kwargs),
        'BiGRU': lambda: BiGRU(input_size, horizon=horizon, **kwargs),
        'Transformer': lambda: Transformer(input_size, horizon=horizon, **kwargs),
        'CNN-LSTM': lambda: CNNLSTM(input_size, horizon=horizon, **kwargs),
        'CNN-GRU': lambda: CNNGRU(input_size, horizon=horizon, **kwargs),
        'DLinear': lambda: DLinear(input_size, horizon=horizon, lookback=lookback, **kwargs),
        'NLinear': lambda: NLinear(input_size, horizon=horizon, lookback=lookback, **kwargs),
        'Naive': lambda: Naive(horizon=horizon),
        'SeasonalNaive': lambda: SeasonalNaive(horizon=horizon, **kwargs),
        # Продвинутые трансформеры
        'Informer': lambda: Informer(input_size, horizon=horizon, **kwargs),
        'Autoformer': lambda: Autoformer(input_size, horizon=horizon, **kwargs),
        'PatchTST': lambda: PatchTST(input_size, horizon=horizon, **kwargs),
        'TFT': lambda: TFT(input_size, horizon=horizon, **kwargs),
        # Гибридные продвинутые
        'TCN-Attention': lambda: TCNAttention(input_size, horizon=horizon, **kwargs),
        'LSTM-AE': lambda: LSTMAE(input_size, horizon=horizon, **kwargs),
    }
    
    if model_name not in models:
        raise ValueError(f"Неизвестная модель: {model_name}")
    
    return models[model_name]()


def create_all_models(input_size, horizon=48, lookback=336, model_configs=None):
    """Создает все модели с заданными конфигурациями."""
    if model_configs is None:
        model_configs = {
            'MLP': {'hidden_sizes': [128, 64, 32], 'dropout': 0.2},
            'TCN': {'num_channels': [64, 128, 256], 'dropout': 0.2},
            'N-BEATS': {'num_blocks': 3, 'layer_size': 512, 'dropout': 0.1},
            'N-HiTS': {'num_stacks': 3, 'num_blocks': 2, 'layer_size': 512, 'dropout': 0.1},
            'RNN': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2},
            'LSTM': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2},
            'GRU': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2},
            'BiLSTM': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2},
            'BiGRU': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2},
            'Transformer': {'d_model': 128, 'nhead': 8, 'num_layers': 3, 'dropout': 0.1},
            'CNN-LSTM': {'cnn_filters': 64, 'lstm_hidden': 128, 'num_layers': 2, 'dropout': 0.2},
            'CNN-GRU': {'cnn_filters': 64, 'gru_hidden': 128, 'num_layers': 2, 'dropout': 0.2},
            'DLinear': {'moving_avg': 25},
            'NLinear': {},
            'Naive': {},
            'SeasonalNaive': {'season_length': 7},
            # Продвинутые трансформеры
            'Informer': {'d_model': 128, 'nhead': 8, 'num_layers': 2, 'dropout': 0.1, 'factor': 5},
            'Autoformer': {'d_model': 128, 'nhead': 8, 'num_layers': 2, 'dropout': 0.1, 'moving_avg': 25},
            'PatchTST': {'d_model': 128, 'nhead': 8, 'num_layers': 2, 'dropout': 0.1, 'patch_len': 16, 'stride': 8},
            'TFT': {'hidden_size': 128, 'nhead': 4, 'num_layers': 2, 'dropout': 0.1},
            # Гибридные продвинутые
            'TCN-Attention': {'num_channels': [64, 128, 256], 'dropout': 0.2, 'nhead': 4},
            'LSTM-AE': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2, 'latent_size': 64},
        }
    
    models = {}
    for model_name, config in model_configs.items():
        try:
            models[model_name] = create_model(model_name, input_size, horizon, lookback=lookback, **config)
        except Exception as e:
            warnings.warn(f"Не удалось создать модель {model_name}: {e}")
    
    return models

