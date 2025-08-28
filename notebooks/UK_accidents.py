import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_recall_curve, roc_auc_score, f1_score,
                           recall_score, precision_score, make_scorer, accuracy_score)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class AccidentSeverityClassifier:
    def __init__(self, data_path=None):
        """
        Initialize the classifier for UK accident severity prediction.
        Focus on maximizing recall for severe accidents (class 1).
        """
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.best_severe_model = None
        self.best_balanced_model = None
        self.calibrated_models = {}
        self.feature_names = []
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """Load and display basic information about the dataset"""
        try:
            self.data = pd.read_csv(data_path)
            print("Dataset loaded successfully!")
            print(f"Dataset shape: {self.data.shape}")
            print(f"\nAccident severity distribution:")
            print(self.data['accident_severity'].value_counts().sort_index())
            print(f"\nClass distribution percentages:")
            dist = self.data['accident_severity'].value_counts(normalize=True).sort_index() * 100
            for severity, pct in dist.items():
                print(f"Severity {severity}: {pct:.2f}%")
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self):
        """Basic data exploration and visualization"""
        if self.data is None:
            print("Please load data first!")
            return
        
        print("="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        # Basic info
        print(f"Dataset shape: {self.data.shape}")
        print(f"Missing values per column:")
        print(self.data.isnull().sum().sort_values(ascending=False).head(10))
        
        # Visualize class distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        self.data['accident_severity'].value_counts().sort_index().plot(kind='bar')
        plt.title('Accident Severity Distribution (Count)')
        plt.xlabel('Severity Level')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        self.data['accident_severity'].value_counts(normalize=True).sort_index().plot(kind='bar')
        plt.title('Accident Severity Distribution (Proportion)')
        plt.xlabel('Severity Level')
        plt.ylabel('Proportion')
        
        plt.tight_layout()
        plt.show()
        
        return self.data.describe()
    
    def preprocess_data(self, target_col='accident_severity', test_size=0.2, random_state=42):
        """
        Preprocess the data including encoding and scaling
        """
        if self.data is None:
            print("Please load data first!")
            return
        
        print("="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Separate features and target
        X = self.data.drop(columns=[target_col])
        y = self.data[target_col]
        
        # Handle missing values
        # For numerical columns, fill with median
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())
        
        # For categorical columns, fill with mode
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
        
        # Encode categorical variables
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        self.feature_names = X.columns.tolist()
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Training set class distribution:")
        print(pd.Series(self.y_train).value_counts().sort_index())
        
        return self.X_train, self.X_test, self.y_train, self.y_test
