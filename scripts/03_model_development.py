"""
Advanced Model Development for Real Estate Prediction

This module implements state-of-the-art ML models including:
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Neural Networks (TabNet, deep feedforward)
- Ensemble methods (stacking, voting)
- Hyperparameter optimization with Optuna
- Cross-validation and evaluation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge

# Gradient Boosting
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Hyperparameter Optimization
import optuna

# MLOps and Tracking
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.tensorflow

class AdvancedModelDevelopment:
    """
    Advanced ML model development with modern techniques.

    Implements multiple model types, hyperparameter optimization,
    ensemble methods, and comprehensive evaluation.
    """

    def __init__(self, features_dir: str = "data/features", models_dir: str = "models"):
        """
        Initialize the model development system.

        Args:
            features_dir: Directory containing feature-engineered data
            models_dir: Directory to save trained models
        """
        # Set up paths
        current_dir = Path.cwd()
        project_root = current_dir
        while not (project_root / "data").exists() and project_root.parent != project_root:
            project_root = project_root.parent

        self.features_dir = project_root / features_dir
        self.models_dir = project_root / models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = self._setup_logging()

        # Initialize MLflow
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        mlflow.set_experiment("real_estate_prediction")

        # Model configurations
        self.target_cols = [
            'target_return_1m', 'target_return_3m',
            'target_return_6m', 'target_return_12m'
        ]

        # Data containers
        self.data: Optional[pd.DataFrame] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[Dict[str, pd.Series]] = None
        self.y_val: Optional[Dict[str, pd.Series]] = None
        self.y_test: Optional[Dict[str, pd.Series]] = None

        # Trained models
        self.models: Dict[str, Dict[str, Any]] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def load_feature_data(self) -> pd.DataFrame:
        """
        Load feature-engineered dataset.

        Returns:
            Feature-engineered DataFrame
        """
        self.logger.info("Loading feature-engineered dataset...")

        # Load training dataset
        training_file = self.features_dir / "training_dataset.csv"
        if not training_file.exists():
            raise FileNotFoundError(f"Training dataset not found: {training_file}")

        # Load data
        self.data = pd.read_csv(training_file)
        self.logger.info(f"Loaded dataset: {self.data.shape}")

        # Convert Date column
        self.data['Date'] = pd.to_datetime(self.data['Date'])

        return self.data

    def prepare_data_splits(self, test_size: float = 0.2, val_size: float = 0.15) -> None:
        """
        Prepare time-series aware data splits.

        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of remaining data for validation
        """
        self.logger.info("Preparing data splits...")

        if self.data is None:
            raise ValueError("Data not loaded. Call load_feature_data() first.")

        # Remove rows with missing targets
        data_clean = self.data.dropna(subset=self.target_cols).copy()
        self.logger.info(f"Clean dataset: {data_clean.shape}")

        # Sort by date for time-series split
        data_clean = data_clean.sort_values('Date')

        # Identify feature columns (exclude metadata and targets)
        exclude_cols = {
            'Date', 'RegionID', 'RegionName', 'Value', 'Dataset', 'GeographyLevel'
        }
        exclude_cols.update({col for col in data_clean.columns if col.startswith('target_')})

        feature_cols = [col for col in data_clean.columns if col not in exclude_cols]
        self.logger.info(f"Feature columns: {len(feature_cols)}")

        # Handle categorical columns
        categorical_cols = []
        for col in feature_cols:
            if data_clean[col].dtype == 'object':
                categorical_cols.append(col)

        self.logger.info(f"Categorical columns to encode: {categorical_cols}")

        # Encode categorical variables
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            data_clean[col] = le.fit_transform(data_clean[col].astype(str))
            label_encoders[col] = le

        # Store encoders for later use
        self.label_encoders = label_encoders

        # Time-series split
        n_samples = len(data_clean)
        test_start = int(n_samples * (1 - test_size))
        val_start = int(test_start * (1 - val_size))

        # Split data
        train_data = data_clean.iloc[:val_start]
        val_data = data_clean.iloc[val_start:test_start]
        test_data = data_clean.iloc[test_start:]

        self.logger.info(f"Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")

        # Prepare features
        self.X_train = train_data[feature_cols]
        self.X_val = val_data[feature_cols]
        self.X_test = test_data[feature_cols]

        # Prepare targets
        self.y_train = {col: train_data[col] for col in self.target_cols}
        self.y_val = {col: val_data[col] for col in self.target_cols}
        self.y_test = {col: test_data[col] for col in self.target_cols}

        # Handle missing values in features
        self.X_train = self.X_train.fillna(0)
        self.X_val = self.X_val.fillna(0)
        self.X_test = self.X_test.fillna(0)

        # Ensure all features are numeric
        for col in self.X_train.columns:
            self.X_train[col] = pd.to_numeric(self.X_train[col], errors='coerce').fillna(0)
            self.X_val[col] = pd.to_numeric(self.X_val[col], errors='coerce').fillna(0)
            self.X_test[col] = pd.to_numeric(self.X_test[col], errors='coerce').fillna(0)

        self.logger.info("Data splits prepared successfully")
        self.logger.info(f"Final feature dtypes: {self.X_train.dtypes.value_counts().to_dict()}")

    def optimize_xgboost(self, target_col: str, n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters using Optuna.

        Args:
            target_col: Target column name
            n_trials: Number of optimization trials

        Returns:
            Best parameters and score
        """
        self.logger.info(f"Optimizing XGBoost for {target_col}...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'n_jobs': -1
            }

            model = xgb.XGBRegressor(**params)
            model.fit(
                self.X_train, self.y_train[target_col],
                eval_set=[(self.X_val, self.y_val[target_col])],
                verbose=False
            )

            y_pred = model.predict(self.X_val)
            rmse = np.sqrt(mean_squared_error(self.y_val[target_col], y_pred))
            return rmse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study
        }

    def optimize_lightgbm(self, target_col: str, n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize LightGBM hyperparameters using Optuna.

        Args:
            target_col: Target column name
            n_trials: Number of optimization trials

        Returns:
            Best parameters and score
        """
        self.logger.info(f"Optimizing LightGBM for {target_col}...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }

            model = lgb.LGBMRegressor(**params)
            model.fit(
                self.X_train, self.y_train[target_col],
                eval_set=[(self.X_val, self.y_val[target_col])],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )

            y_pred = model.predict(self.X_val)
            rmse = np.sqrt(mean_squared_error(self.y_val[target_col], y_pred))
            return rmse

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study
        }

    def create_neural_network(self, input_dim: int, target_col: str) -> keras.Model:
        """
        Create TabNet-inspired neural network.

        Args:
            input_dim: Number of input features
            target_col: Target column name

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Dense(512, activation='relu', input_dim=input_dim),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),

            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def train_models(self, target_col: str = 'target_return_3m') -> Dict[str, Any]:
        """
        Train all model types for a specific target.

        Args:
            target_col: Target column to predict

        Returns:
            Dictionary containing trained models and metrics
        """
        self.logger.info(f"Training models for {target_col}...")

        with mlflow.start_run(run_name=f"training_{target_col}"):
            models = {}

            # 1. XGBoost with optimization
            self.logger.info("Training XGBoost...")
            xgb_results = self.optimize_xgboost(target_col, n_trials=2)

            xgb_model = xgb.XGBRegressor(**xgb_results['best_params'])
            xgb_model.fit(self.X_train, self.y_train[target_col])

            models['xgboost'] = {
                'model': xgb_model,
                'params': xgb_results['best_params'],
                'validation_score': xgb_results['best_score']
            }

            # 2. LightGBM with optimization
            self.logger.info("Training LightGBM...")
            lgb_results = self.optimize_lightgbm(target_col, n_trials=2)

            lgb_model = lgb.LGBMRegressor(**lgb_results['best_params'])
            lgb_model.fit(self.X_train, self.y_train[target_col])

            models['lightgbm'] = {
                'model': lgb_model,
                'params': lgb_results['best_params'],
                'validation_score': lgb_results['best_score']
            }

            # 3. CatBoost
            self.logger.info("Training CatBoost...")
            cat_model = cb.CatBoostRegressor(
                iterations=500,
                learning_rate=0.1,
                depth=6,
                random_state=42,
                verbose=False
            )
            cat_model.fit(self.X_train, self.y_train[target_col])

            models['catboost'] = {
                'model': cat_model,
                'params': cat_model.get_params(),
                'validation_score': None
            }

            # 4. Neural Network
            self.logger.info("Training Neural Network...")

            # Scale features for neural network
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(self.X_train)
            X_val_scaled = scaler.transform(self.X_val)

            nn_model = self.create_neural_network(self.X_train.shape[1], target_col)

            # Train with early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=20, restore_best_weights=True
            )

            history = nn_model.fit(
                X_train_scaled, self.y_train[target_col],
                validation_data=(X_val_scaled, self.y_val[target_col]),
                epochs=200,
                batch_size=1024,
                callbacks=[early_stopping],
                verbose=0
            )

            models['neural_network'] = {
                'model': nn_model,
                'scaler': scaler,
                'history': history.history,
                'validation_score': min(history.history['val_loss'])
            }

            # 5. Ensemble
            self.logger.info("Creating ensemble...")

            ensemble_models = [
                ('xgb', models['xgboost']['model']),
                ('lgb', models['lightgbm']['model']),
                ('cat', models['catboost']['model'])
            ]

            voting_model = VotingRegressor(ensemble_models)
            voting_model.fit(self.X_train, self.y_train[target_col])

            models['ensemble_voting'] = {
                'model': voting_model,
                'params': {},
                'validation_score': None
            }

            # Log metrics to MLflow
            mlflow.log_param("target_column", target_col)
            mlflow.log_param("n_features", self.X_train.shape[1])
            mlflow.log_param("train_samples", len(self.X_train))

            # Store models
            self.models[target_col] = models

            self.logger.info(f"Completed training for {target_col}")
            return models

    def evaluate_models(self, target_col: str) -> pd.DataFrame:
        """
        Evaluate all models on test set.

        Args:
            target_col: Target column name

        Returns:
            DataFrame with evaluation metrics
        """
        self.logger.info(f"Evaluating models for {target_col}...")

        if target_col not in self.models:
            raise ValueError(f"Models not trained for {target_col}")

        results = []
        models = self.models[target_col]

        for model_name, model_info in models.items():
            model = model_info['model']

            # Make predictions
            if model_name == 'neural_network':
                # Use scaled features for neural network
                scaler = model_info['scaler']
                X_test_scaled = scaler.transform(self.X_test)
                y_pred = model.predict(X_test_scaled).flatten()
            else:
                y_pred = model.predict(self.X_test)

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(self.y_test[target_col], y_pred))
            mae = mean_absolute_error(self.y_test[target_col], y_pred)
            r2 = r2_score(self.y_test[target_col], y_pred)

            results.append({
                'Model': model_name,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'Validation_Score': model_info.get('validation_score', 'N/A')
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('RMSE')

        return results_df

    def run_full_training_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline for all targets.

        Returns:
            Dictionary with all results
        """
        self.logger.info("🚀 STARTING ADVANCED MODEL DEVELOPMENT")
        self.logger.info("=" * 60)

        # Load data and prepare splits
        self.load_feature_data()
        self.prepare_data_splits()

        # Train models for each target
        all_results = {}

        for target_col in self.target_cols:
            self.logger.info(f"Processing target: {target_col}")

            # Train models
            models = self.train_models(target_col)

            # Evaluate models
            evaluation = self.evaluate_models(target_col)

            all_results[target_col] = {
                'models': models,
                'evaluation': evaluation
            }

            # Save best model
            best_model_name = evaluation.iloc[0]['Model']
            best_model = models[best_model_name]['model']

            model_file = self.models_dir / f"best_model_{target_col}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': best_model,
                    'model_name': best_model_name,
                    'scaler': models[best_model_name].get('scaler'),
                    'feature_names': list(self.X_train.columns)
                }, f)

            self.logger.info(f"Saved best model for {target_col}: {best_model_name}")
            print(f"\n📊 {target_col.upper()} EVALUATION:")
            print("=" * 40)
            print(evaluation.to_string(index=False))
            print()

        # Overall summary
        print("\n🎯 MODEL DEVELOPMENT COMPLETE!")
        print("=" * 50)
        print(f"📊 Models trained for {len(self.target_cols)} targets")
        print(f"🔧 Model types: XGBoost, LightGBM, CatBoost, Neural Network, Ensemble")
        print(f"📈 Dataset: {len(self.X_train):,} training samples")
        print(f"🎯 Features: {self.X_train.shape[1]} engineered features")
        print(f"📂 Models saved to: {self.models_dir}")
        print()
        print("🚀 Ready for Phase 5: Production Deployment!")

        return all_results

def main():
    """Main execution function."""
    model_dev = AdvancedModelDevelopment()
    results = model_dev.run_full_training_pipeline()
    return results

if __name__ == "__main__":
    results = main()