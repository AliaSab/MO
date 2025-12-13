"""
Модуль для обучения моделей глубокого обучения.
Включает стратегии обучения, регуляризацию, early stopping, gradient clipping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesDataset(Dataset):
    """Dataset для временных рядов."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ModelTrainer:
    """Класс для обучения моделей глубокого обучения."""
    
    def __init__(self, model, device='cpu', loss_fn='mse', optimizer='adam',
                 lr=1e-3, weight_decay=1e-4, gradient_clip=1.0,
                 early_stopping_patience=15, reduce_lr_patience=10,
                 label_smoothing=0.0):
        """
        Args:
            model: PyTorch модель
            device: 'cpu' или 'cuda'
            loss_fn: 'mse', 'mae', 'huber' или комбинация
            optimizer: 'adam', 'adamw', 'radam'
            lr: learning rate
            weight_decay: weight decay для регуляризации
            gradient_clip: максимальная норма градиента
            early_stopping_patience: терпение для early stopping
            reduce_lr_patience: терпение для ReduceLROnPlateau
            label_smoothing: параметр label smoothing
        """
        self.model = model.to(device)
        self.device = device
        self.label_smoothing = label_smoothing
        
        # Функция потерь
        if loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_fn == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_fn == 'huber':
            self.criterion = nn.HuberLoss()
        elif loss_fn == 'mse+mae':
            self.criterion = lambda pred, target: nn.MSELoss()(pred, target) + nn.L1Loss()(pred, target)
        elif loss_fn == 'mse+huber':
            self.criterion = lambda pred, target: nn.MSELoss()(pred, target) + nn.HuberLoss()(pred, target)
        else:
            self.criterion = nn.MSELoss()
        
        # Проверяем, есть ли параметры для обучения
        params = list(model.parameters())
        self.has_trainable_params = len(params) > 0
        
        # Оптимизатор (только если есть параметры)
        if self.has_trainable_params:
            if optimizer == 'adam':
                self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer == 'adamw':
                self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer == 'radam':
                try:
                    from torch_optimizer import RAdam
                    self.optimizer = RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
                except ImportError:
                    warnings.warn("RAdam не установлен, используется Adam")
                    self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            # Scheduler
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=reduce_lr_patience
            )
        else:
            # Для моделей без параметров (например, Naive)
            self.optimizer = None
            self.scheduler = None
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        self.best_epoch = 0
        
        # Gradient clipping
        self.gradient_clip = gradient_clip
        
        # История обучения
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader):
        """Одна эпоха обучения."""
        if not self.has_trainable_params:
            # Для моделей без параметров просто вычисляем loss
            self.model.eval()
            total_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    predictions = self.model(X_batch)
                    loss = self.criterion(predictions, y_batch)
                    total_loss += loss.item()
            return total_loss / len(train_loader)
        
        self.model.train()
        total_loss_original = 0  # Loss на оригинальных данных для сравнения с validation
        n_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            
            # Проверка размерностей и значений (только для первого батча)
            if n_batches == 0:
                if predictions.shape != y_batch.shape:
                    import warnings
                    warnings.warn(f"⚠️ Размерности не совпадают: predictions={predictions.shape}, y_batch={y_batch.shape}")
                # Отладочная информация о данных
                print(f"🔍 ОТЛАДКА train_epoch (первый батч):")
                print(f"  X_batch: shape={X_batch.shape}, range=[{X_batch.min():.6f}, {X_batch.max():.6f}]")
                print(f"  y_batch: shape={y_batch.shape}, range=[{y_batch.min():.6f}, {y_batch.max():.6f}]")
                print(f"  predictions: shape={predictions.shape}, range=[{predictions.min():.6f}, {predictions.max():.6f}]")
                print(f"  y_batch mean={y_batch.mean():.6f}, std={y_batch.std():.6f}")
                print(f"  predictions mean={predictions.mean():.6f}, std={predictions.std():.6f}")
            
            # Вычисляем loss на оригинальных данных для отображения (для сравнения с validation)
            loss_original = self.criterion(predictions, y_batch)
            total_loss_original += loss_original.item()
            
            # Label smoothing (только для обучения, не для отображения)
            if self.label_smoothing > 0:
                y_smooth = y_batch * (1 - self.label_smoothing) + y_batch.mean() * self.label_smoothing
                loss = self.criterion(predictions, y_smooth)
            else:
                loss = loss_original
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            n_batches += 1
        
        # Возвращаем loss на оригинальных данных для корректного сравнения с validation
        avg_loss = total_loss_original / n_batches if n_batches > 0 else 0.0
        if n_batches > 0:
            print(f"🔍 train_epoch итог: n_batches={n_batches}, avg_loss={avg_loss:.6f}")
        return avg_loss
    
    def validate(self, val_loader):
        """Валидация модели."""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                
                # Проверка размерностей и значений (только для первого батча)
                if n_batches == 0:
                    if predictions.shape != y_batch.shape:
                        import warnings
                        warnings.warn(f"⚠️ Размерности не совпадают в validation: predictions={predictions.shape}, y_batch={y_batch.shape}")
                    # Отладочная информация о данных
                    print(f"🔍 ОТЛАДКА validate (первый батч):")
                    print(f"  X_batch: shape={X_batch.shape}, range=[{X_batch.min():.6f}, {X_batch.max():.6f}]")
                    print(f"  y_batch: shape={y_batch.shape}, range=[{y_batch.min():.6f}, {y_batch.max():.6f}]")
                    print(f"  predictions: shape={predictions.shape}, range=[{predictions.min():.6f}, {predictions.max():.6f}]")
                    print(f"  y_batch mean={y_batch.mean():.6f}, std={y_batch.std():.6f}")
                    print(f"  predictions mean={predictions.mean():.6f}, std={predictions.std():.6f}")
                
                loss = self.criterion(predictions, y_batch)
                total_loss += loss.item()
                n_batches += 1
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        if n_batches > 0:
            print(f"🔍 validate итог: n_batches={n_batches}, avg_loss={avg_loss:.6f}")
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs=100, verbose=True):
        """Полный цикл обучения."""
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА: вычисляем train loss в режиме eval() для сравнения
            if epoch == 0:
                self.model.eval()
                train_loss_eval = 0
                train_batches_eval = 0
                with torch.no_grad():
                    for X_batch, y_batch in train_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        predictions = self.model(X_batch)
                        loss_eval = self.criterion(predictions, y_batch)
                        train_loss_eval += loss_eval.item()
                        train_batches_eval += 1
                train_loss_eval = train_loss_eval / train_batches_eval if train_batches_eval > 0 else 0.0
                self.model.train()
                
                print(f"🔍 КРИТИЧЕСКАЯ ПРОВЕРКА на эпохе {epoch+1}:")
                print(f"  Train Loss (train mode): {train_loss:.6f}")
                print(f"  Train Loss (eval mode): {train_loss_eval:.6f}")
                print(f"  Val Loss (eval mode): {val_loss:.6f}")
                if abs(train_loss - train_loss_eval) > 0.01:
                    print(f"  ⚠️ РАЗНИЦА между train() и eval() режимами: {abs(train_loss - train_loss_eval):.6f}")
                    print(f"  Это указывает на влияние dropout или batch normalization!")
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА: сравниваем train и val loss
            if epoch == 0 or (epoch + 1) % 10 == 0:
                if train_loss > val_loss * 2:
                    print(f"⚠️ ВНИМАНИЕ на эпохе {epoch+1}: Train Loss ({train_loss:.6f}) >> Val Loss ({val_loss:.6f})")
                    print(f"  Это может указывать на проблему с данными или моделью!")
                    print(f"  Возможные причины:")
                    print(f"    1. Train и val данные имеют разные масштабы (разные std)")
                    print(f"       → Это НОРМАЛЬНО для временных рядов, но влияет на loss")
                    print(f"       → Train данные могут иметь большую вариацию, чем val")
                    print(f"    2. Модель в режиме train() ведет себя по-другому (dropout, batch norm)")
                    print(f"    3. Train данных слишком мало для обучения")
                    print(f"  Решение:")
                    print(f"    - Проверьте статистику train и val данных (mean, std)")
                    print(f"    - Если std сильно отличается, это объясняет разницу в loss")
                    print(f"    - Это нормально для временных рядов, но может быть проблемой")
            
            # Scheduler step (только если есть оптимизатор)
            if self.scheduler is not None:
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_loss)
                new_lr = self.optimizer.param_groups[0]['lr']
                if old_lr != new_lr and verbose:
                    print(f"Learning rate изменен: {old_lr:.6f} -> {new_lr:.6f}")
            
            # Early stopping - сохраняем лучшую модель
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                self.best_epoch = epoch + 1
            else:
                self.patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping - останавливаемся, если нет улучшения
            if self.patience_counter >= self.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1} (лучшая модель была на эпохе {self.best_epoch})")
                break
        
        # Загружаем лучшую модель (с лучшим validation loss)
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if verbose:
                print(f"Загружена лучшая модель с эпохи {self.best_epoch} (val_loss={self.best_val_loss:.4f})")
        else:
            # Если лучшая модель не была сохранена (не должно происходить), сохраняем текущую
            if verbose:
                print("Предупреждение: лучшая модель не была сохранена, используется последняя эпоха")
        
        return self.train_losses, self.val_losses
    
    def predict(self, test_loader):
        """Предсказания на тестовых данных."""
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                pred = self.model(X_batch)
                predictions.append(pred.cpu().numpy())
                targets.append(y_batch.cpu().numpy())
        
        return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def train_model(model, X_train, y_train, X_val, y_val, 
                batch_size=32, epochs=100, device='cpu', verbose=True, **trainer_kwargs):
    """Удобная функция для обучения модели."""
    # Создаем datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    
    # Оптимизация DataLoader для скорости
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=False)
    
    # Создаем trainer
    trainer = ModelTrainer(model, device=device, **trainer_kwargs)
    
    # Обучаем
    train_losses, val_losses = trainer.train(train_loader, val_loader, epochs=epochs, verbose=verbose)
    
    return trainer, train_losses, val_losses

