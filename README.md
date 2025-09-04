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

KEY INSIGHTS:

   - Severe-optimized model: Perfect for emergency services (don't miss critical cases)
   - Balanced model: Better for general traffic management and resource planning
   - SMOTE+Tomek sampling consistently performs well across both strategies
   - Class imbalance is successfully handled through multiple techniques

|Metric              |      Severe-Optimized |  Balanced      |     Difference     
|--------------------|-----------------------|----------------|---------------------
|Severe Recall       |      0.997            |   0.868        |       +0.128
|Severe Precision    |      0.024            |   0.130        |       -0.106
|Macro Recall        |      0.561            |   0.812        |       -0.251
|Overall Accuracy    |      0.223            |   0.843        |       -0.620
|====================|=======================|================|=====================
