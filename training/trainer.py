import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import time
import os
from datetime import datetime


class Trainer:
    """Main training class for Protein FT-Transformer"""
    
    def __init__(self, model, train_loader, val_loader, config, device, logger=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger
        
        # Setup training components
        self._setup_loss()
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config['training']['use_mixed_precision'] else None
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_acc = 0
        self.best_model_state = None
        
        # Create output directories
        os.makedirs(config['output']['model_save_dir'], exist_ok=True)
        os.makedirs(config['output']['results_save_dir'], exist_ok=True)
    
    def _setup_loss(self):
        """Setup loss function with class weights"""
        train_labels = []
        for _, labels in self.train_loader:
            train_labels.append(labels.cpu().numpy())
        train_labels = np.concatenate(train_labels)
        
        # Calculate class weights
        if self.config['loss']['use_class_weights']:
            class_counts = np.bincount(train_labels)
            class_weights = 1.0 / (class_counts + 1e-6)
            class_weights = class_weights / class_weights.sum()
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        else:
            class_weights = None
        
        # Create loss function
        from models.losses import MulticlassFocalLoss
        self.criterion = MulticlassFocalLoss(
            alpha=class_weights,
            gamma=self.config['loss']['focal_gamma'],
            label_smoothing=self.config['loss']['label_smoothing']
        )
        
        if self.logger:
            self.logger.info(f"Class weights: {class_weights}")
    
    def _setup_optimizer(self):
        """Setup optimizer with layer-wise learning rates"""
        model_params = []
        
        # Different learning rates for different parts
        lr_config = [
            (self.model.feature_tokenizer.parameters(), self.config['training']['learning_rate'] * 0.5),
            (self.model.transformer.parameters(), self.config['training']['learning_rate']),
            (self.model.classification_heads.parameters(), self.config['training']['learning_rate'] * 0.8),
            (self.model.output_combiner.parameters(), self.config['training']['learning_rate'] * 0.5),
            ([self.model.temperature], self.config['training']['learning_rate'] * 0.1)
        ]
        
        for params, lr in lr_config:
            model_params.append({'params': params, 'lr': lr})
        
        self.optimizer = optim.AdamW(
            model_params,
            weight_decay=self.config['training']['weight_decay']
        )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup"""
        def lr_lambda(epoch):
            warmup_epochs = 10
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            return 0.95 ** (epoch - warmup_epochs)
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]', leave=False)
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * inputs.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total
            })
        
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validation', leave=False)
            
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item() * inputs.size(0)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
                
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': correct / total
                })
        
        val_loss = total_loss / total
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def train(self, num_epochs=None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        early_stopping_counter = 0
        best_val_loss = float('inf')
        
        print(f"\n{'='*60}")
        print(f"Starting Training for {num_epochs} epochs")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                
                if self.config['training']['save_best_only']:
                    self.save_checkpoint(f"best_model_epoch_{epoch+1}.pth")
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"LR: {current_lr:.6f} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss - self.config['training']['early_stopping_min_delta']:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter >= self.config['training']['early_stopping_patience']:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f} seconds")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        return self.history
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        save_path = os.path.join(self.config['output']['model_save_dir'], filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'epoch': len(self.history['train_loss'])
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, save_path)
        
        if self.logger:
            self.logger.info(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.history = checkpoint['history']
        self.best_val_acc = checkpoint['best_val_acc']
        
        if self.logger:
            self.logger.info(f"Checkpoint loaded from {path}")
