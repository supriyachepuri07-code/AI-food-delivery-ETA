# System Architecture

## 1. System Overview

The AI Food Delivery ETA Prediction System is designed as a modular, scalable, and production-ready Machine Learning platform that predicts food delivery times using historical delivery records and real-time operational information.

The system consists of multiple independent components responsible for data ingestion, data validation, preprocessing, feature engineering, model training, model serving, workflow orchestration, monitoring, and cloud deployment.

Each component has a clearly defined responsibility, making the platform easier to maintain, scale, test, and extend as the business grows.

The architecture follows an end-to-end MLOps approach, enabling automated data pipelines, reproducible model training, API-based prediction services, continuous integration and deployment (CI/CD), and cloud-native deployment on Google Cloud Platform.
## 2. High-Level Architecture

The AI Food Delivery ETA Prediction System is divided into two major workflows:

1. **Offline Training Pipeline**
   - Responsible for collecting historical delivery data, validating it, preprocessing it, engineering features, training machine learning models, evaluating their performance, and storing the best-performing model for deployment.

2. **Online Prediction Pipeline**
   - Responsible for receiving incoming prediction requests, collecting the required operational data, preparing input features, loading the trained model, generating ETA predictions, and returning the estimated delivery time to the application through a REST API.

Separating the training and prediction pipelines allows the system to retrain models independently without affecting the real-time prediction service. This architecture improves maintainability, scalability, and reliability while following production-grade MLOps practices.