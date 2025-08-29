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
import lightgbm as lgb

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
    def apply_sampling_techniques(self):
        """
        Apply various sampling techniques to handle imbalanced data
        """
        if self.X_train is None:
            print("Please preprocess data first!")
            return
        
        print("="*50)
        print("APPLYING SAMPLING TECHNIQUES")
        print("="*50)
        
        sampling_techniques = {}
        
        # Original data
        sampling_techniques['Original'] = (self.X_train, self.y_train)
        
        # SMOTE
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(self.X_train, self.y_train)
        sampling_techniques['SMOTE'] = (X_smote, y_smote)
        
        # ADASYN
        try:
            adasyn = ADASYN(random_state=42)
            X_adasyn, y_adasyn = adasyn.fit_resample(self.X_train, self.y_train)
            sampling_techniques['ADASYN'] = (X_adasyn, y_adasyn)
        except:
            print("ADASYN failed, skipping...")
        
        # SMOTE + Tomek
        smote_tomek = SMOTETomek(random_state=42)
        X_smote_tomek, y_smote_tomek = smote_tomek.fit_resample(self.X_train, self.y_train)
        sampling_techniques['SMOTE+Tomek'] = (X_smote_tomek, y_smote_tomek)
        
        # Random Under Sampling (to balance severe cases)
        rus = RandomUnderSampler(random_state=42, sampling_strategy={3: 10000, 2: 5000, 1: 1218})
        X_rus, y_rus = rus.fit_resample(self.X_train, self.y_train)
        sampling_techniques['Random Under Sampling'] = (X_rus, y_rus)
        
        # Display distributions
        for name, (X_samp, y_samp) in sampling_techniques.items():
            print(f"\n{name} distribution:")
            print(pd.Series(y_samp).value_counts().sort_index())
        
        return sampling_techniques
    
    def train_models(self, sampling_techniques):
        """
        Train multiple models with different sampling techniques
        Focus on two optimization strategies:
        1. Maximum severe case recall
        2. Best overall balance (macro recall)
        """
        print("="*50)
        print("TRAINING MODELS - DUAL OPTIMIZATION STRATEGY")
        print("="*50)
        
        # Define models with different optimization focuses
        severe_optimized_models = {
            'RF_SevereOptim': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                class_weight={1: 50, 2: 5, 3: 1},  # Heavy weight for severe cases
                random_state=42
            ),
            'LR_SevereOptim': LogisticRegression(
                class_weight={1: 50, 2: 5, 3: 1},
                random_state=42,
                max_iter=1000
            ),
            'GB_SevereOptim': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        }
        
        balanced_models = {
            'RF_Balanced': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                class_weight='balanced',
                random_state=42
            ),
            'LR_Balanced': LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            ),
            'BalancedRF': BalancedRandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        }
        
        
        # Calculate class weights for LightGBM
        class_counts = pd.Series(self.y_train).value_counts()
        total_samples = len(self.y_train)
            
        balanced_models['LightGBM_Balanced'] = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=3,
                class_weight='balanced',
                n_estimators=100,
                random_state=42,
                verbosity=-1
            )
            
        severe_optimized_models['LightGBM_SevereOptim'] = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=3,
                class_weight={1: 50, 2: 5, 3: 1},
                n_estimators=100,
                random_state=42,
                verbosity=-1
            )
        
        all_models = {**severe_optimized_models, **balanced_models}
        results = []
        
        for sampling_name, (X_samp, y_samp) in sampling_techniques.items():
            print(f"\nTraining models with {sampling_name}...")
            
            # Scale the sampled data
            if sampling_name == 'Original':
                X_samp_scaled = self.X_train_scaled
            else:
                X_samp_scaled = self.scaler.fit_transform(X_samp)
            
            for model_name, model in all_models.items():
                # Skip BalancedRandomForest for non-original data as it handles imbalance internally
                if model_name == 'BalancedRF' and sampling_name != 'Original':
                    continue
                
                try:
                    # Train model
                    if 'LR_' in model_name:
                        model.fit(X_samp_scaled, y_samp)
                        y_pred = model.predict(self.X_test_scaled)
                        y_pred_proba = model.predict_proba(self.X_test_scaled)
                    else:
                        model.fit(X_samp, y_samp)
                        y_pred = model.predict(self.X_test)
                        y_pred_proba = model.predict_proba(self.X_test)
                    
                    # Calculate comprehensive metrics
                    recall_severe = recall_score(self.y_test, y_pred, labels=[1], average='macro', zero_division=0)
                    precision_severe = precision_score(self.y_test, y_pred, labels=[1], average='macro', zero_division=0)
                    f1_severe = f1_score(self.y_test, y_pred, labels=[1], average='macro', zero_division=0)
                    
                    # Overall metrics
                    recall_macro = recall_score(self.y_test, y_pred, average='macro')
                    precision_macro = precision_score(self.y_test, y_pred, average='macro')
                    f1_macro = f1_score(self.y_test, y_pred, average='macro')
                    accuracy = accuracy_score(self.y_test, y_pred)
                    
                    # Per-class metrics
                    recall_per_class = recall_score(self.y_test, y_pred, average=None)
                    precision_per_class = precision_score(self.y_test, y_pred, average=None, zero_division=0)
                    
                    # Calculate severe case statistics from confusion matrix
                    cm = confusion_matrix(self.y_test, y_pred, labels=[1, 2, 3])
                    if cm.shape[0] > 0:  # Ensure severe cases exist in test set
                        severe_true_pos = cm[0, 0] if cm.shape[0] > 0 else 0
                        severe_false_pos = cm[1:, 0].sum() if cm.shape[0] > 0 else 0
                        severe_precision_actual = severe_true_pos / (severe_true_pos + severe_false_pos) if (severe_true_pos + severe_false_pos) > 0 else 0
                    else:
                        severe_precision_actual = 0
                    
                    # Model type classification
                    model_type = "Severe-Optimized" if "SevereOptim" in model_name else "Balanced"
                    
                    results.append({
                        'Sampling': sampling_name,
                        'Model': model_name,
                        'Model_Type': model_type,
                        'Recall_Severe': recall_severe,
                        'Precision_Severe': precision_severe,
                        'F1_Severe': f1_severe,
                        'Severe_Precision_Actual': severe_precision_actual,
                        'Recall_Macro': recall_macro,
                        'Precision_Macro': precision_macro,
                        'F1_Macro': f1_macro,
                        'Accuracy': accuracy,
                        'Recall_Class1': recall_per_class[0] if len(recall_per_class) > 0 else 0,
                        'Recall_Class2': recall_per_class[1] if len(recall_per_class) > 1 else 0,
                        'Recall_Class3': recall_per_class[2] if len(recall_per_class) > 2 else 0,
                        'Model_Object': model,
                        'Predictions': y_pred,
                        'Probabilities': y_pred_proba,
                        'Sampling_Data': (X_samp, y_samp)
                    })
                    
                except Exception as e:
                    print(f"Error training {model_name} with {sampling_name}: {e}")
        
        self.results_df = pd.DataFrame(results)
        return self.results_df
    