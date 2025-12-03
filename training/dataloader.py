import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class ProteinDataset(Dataset):
    """Dataset class for protein classification"""
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DataProcessor:
    """Data preprocessing and loading"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.categorical_columns = config['data']['categorical_columns']
        self.numeric_columns = config['data']['numeric_columns']
    
    def load_data(self, train_path, test_path=None):
        """Load and preprocess data"""
        # Load training data
        df_train = pd.read_csv(train_path, low_memory=False)
        
        # Handle target column
        target_col = self.config['data']['target_column']
        if target_col not in df_train.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Separate features and target
        y_train = df_train[target_col].values
        X_train = df_train.drop(columns=[target_col])
        
        # Load test data if provided
        if test_path:
            df_test = pd.read_csv(test_path, low_memory=False)
            y_test = df_test[target_col].values
            X_test = df_test.drop(columns=[target_col])
        else:
            X_test, y_test = None, None
        
        return self.preprocess_data(X_train, y_train, X_test, y_test)
    
    def preprocess_data(self, X_train, y_train, X_test=None, y_test=None):
        """Preprocess the data"""
        # Encode labels
        y_train = self.label_encoder.fit_transform(y_train)
        if y_test is not None:
            y_test = self.label_encoder.transform(y_test)
        
        # Identify column types
        if not self.categorical_columns:
            self.categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
        if not self.numeric_columns:
            self.numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Encode categorical columns
        categorical_encoders = {}
        for col in self.categorical_columns:
            if col in X_train.columns:
                encoder = LabelEncoder()
                X_train[col] = encoder.fit_transform(X_train[col].astype(str))
                if X_test is not None:
                    X_test[col] = encoder.transform(X_test[col].astype(str))
                categorical_encoders[col] = encoder
        
        # Handle missing values
        for col in self.numeric_columns:
            if col in X_train.columns:
                median_val = X_train[col].median()
                X_train[col].fillna(median_val, inplace=True)
                if X_test is not None:
                    X_test[col].fillna(median_val, inplace=True)
        
        # Scale features
        self.feature_columns = X_train.columns.tolist()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = None
        
        return {
            'X_train': X_train_scaled.astype(np.float32),
            'y_train': y_train.astype(np.int64),
            'X_test': X_test_scaled.astype(np.float32) if X_test_scaled is not None else None,
            'y_test': y_test.astype(np.int64) if y_test is not None else None,
            'feature_names': self.feature_columns,
            'class_names': self.label_encoder.classes_,
            'categorical_encoders': categorical_encoders
        }
    
    def create_datasets(self, data_dict, validation_split=0.1):
        """Create train, validation, and test datasets"""
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        
        # Split training data for validation
        if validation_split > 0:
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train, y_train, 
                test_size=validation_split,
                stratify=y_train,
                random_state=self.config.get('seed', 42)
            )
        else:
            X_train_final, y_train_final = X_train, y_train
            X_val, y_val = None, None
        
        # Create datasets
        train_dataset = ProteinDataset(X_train_final, y_train_final)
        val_dataset = ProteinDataset(X_val, y_val) if X_val is not None else None
        test_dataset = ProteinDataset(X_test, y_test) if X_test is not None else None
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self, train_dataset, val_dataset=None, test_dataset=None, batch_size=64):
        """Create data loaders"""
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
        
        test_loader = None
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
        
        return train_loader, val_loader, test_loader
