"""
Stress Prediction Model Training Module

This module contains training scripts and utilities for building stress prediction models.

The training pipeline uses data from:
    src/stress_prediction/data/stress_training.db

Output:
    Trained models in src/stress_prediction/models/

Usage:
    # Future: Train LightGBM model
    python -m src.stress_prediction.training.train_lightgbm
    
    # Future: Evaluate model
    python -m src.stress_prediction.training.evaluate_model
"""

from pathlib import Path

TRAINING_MODULE_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
