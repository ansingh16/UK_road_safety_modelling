**UK Road Accident Severity Classification - Dual Strategy Approach (2023)**

**📌 Project Overview**
This project leverages the **UK Department for Transport Road Safety Data (2023)** to develop a comprehensive accident severity prediction system with two optimization strategies:
* **Emergency Response Model** - Achieves 92.4% recall for severe accidents to ensure no critical cases are missed
* **Traffic Management Model** - Provides balanced classification across all severity levels for resource allocation
* **Advanced Imbalanced Learning** - Handles extreme class imbalance (severe cases: 1.4% of data) using SMOTE, ADASYN, and custom threshold optimization

The system addresses the critical challenge of **detecting life-threatening accidents** while maintaining practical utility for general traffic management, supporting **data-driven emergency response** and road safety improvements.

**📊 Data Sources**
Data for 2023 is publicly available from the **UK Department for Transport**:
* **Collisions 2023 CSV** - Accident details, location, conditions, timing
* **Vehicles 2023 CSV** - Vehicle characteristics, maneuvers, damage
* **Casualties 2023 CSV** - Injury severity, demographics, roles
* **Dataset Statistics**: 151,852 total accidents (117K slight, 32K serious, 2K severe)

**🎯 Key Features**
* **Dual Model Architecture** - Separate models optimized for different use cases
* **Imbalanced Learning Techniques** - SMOTE+Tomek, ADASYN, custom class weighting
* **Probability Calibration** - Improved threshold optimization for false alarm reduction
* **Business Impact Analysis** - Quantifies trade-offs between recall and precision
* **Production-Ready Pipeline** - Model saving/loading, comprehensive evaluation metrics

**🛠 Tech Stack**
* **Python** - Core development and machine learning pipeline
* **Scikit-learn / Imbalanced-learn** - Classification algorithms and sampling techniques
* **LightGBM** - Advanced gradient boosting for balanced performance
* **Pandas / NumPy** - Data preprocessing and feature engineering
* **Matplotlib / Seaborn** - Model evaluation and performance visualization
* **Joblib** - Model serialization for production deployment

**📈 Model Performance**
| Strategy | Severe Recall | Severe Precision | Macro Recall | Use Case |
|----------|---------------|------------------|--------------|----------|
| **Emergency Response** | **92.4%** | 3.6% | 65% | Critical case detection |
| **Traffic Management** | 78% | **15%** | **72%** | Balanced classification |

**🚨 Business Impact**
* **Emergency Services**: 92.4% severe case detection ensures rapid response to life-threatening accidents
* **Resource Allocation**: Balanced model reduces false alarms while maintaining good overall performance  
* **Cost-Benefit Analysis**: Trade-off quantification helps authorities choose optimal deployment strategy
* **Scalable Solution**: Production-ready models with comprehensive monitoring and evaluation frameworks