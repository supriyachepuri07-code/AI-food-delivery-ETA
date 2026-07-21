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
## 3. Major Components

The AI Food Delivery ETA Prediction System consists of the following major components, each responsible for a specific part of the machine learning lifecycle.

| Component | Responsibility |
|-----------|----------------|
| Data Ingestion | Collects historical and operational data from different sources. |
| Data Validation | Validates data quality, schema, and completeness before processing. |
| Data Preprocessing | Cleans raw data, handles missing values, removes duplicates, and prepares the dataset for feature engineering. |
| Feature Engineering | Creates meaningful input features for machine learning models. |
| Model Training | Trains regression models using the processed dataset. |
| Model Evaluation | Compares trained models using predefined evaluation metrics and selects the best-performing model. |
| Model Registry | Stores versioned machine learning models for deployment and rollback. |
| FastAPI Prediction Service | Provides REST API endpoints for real-time ETA prediction requests. |
| Apache Airflow | Automates data pipelines, model training, and scheduled workflows. |
| MLflow | Tracks machine learning experiments, metrics, parameters, and model versions. |
| PostgreSQL | Stores structured operational data such as restaurant, driver, and order information. |
| Logging & Monitoring | Tracks application logs, pipeline execution, API health, and model performance. |
| Docker | Packages the application and its dependencies into portable containers. |
| GitHub Actions | Automates testing, building, and deployment using CI/CD pipelines. |
| Google Cloud Platform (GCP) | Hosts the application, APIs, and machine learning services in the cloud. |
## 4. Data Flow

The system contains two independent data flows: the Offline Training Data Flow and the Online Prediction Data Flow.

### 4.1 Offline Training Data Flow

The offline training pipeline is responsible for preparing historical delivery data and training machine learning models.

**Flow:**

1. Collect historical delivery data.
2. Validate the data for quality and schema consistency.
3. Clean and preprocess the validated dataset.
4. Perform feature engineering to create machine learning features.
5. Train regression models using the processed dataset.
6. Evaluate multiple models using predefined metrics.
7. Register the best-performing model in the Model Registry (MLflow).

### 4.2 Online Prediction Data Flow

The online prediction pipeline is responsible for generating ETA predictions for incoming customer orders.

**Flow:**

1. Receive prediction requests through the FastAPI service.
2. Collect the required operational data, including restaurant, driver, customer, traffic, and weather information.
3. Apply the same feature engineering transformations used during training.
4. Load the latest production-ready model.
5. Generate the ETA prediction.
6. Return the prediction to the requesting application as a JSON response.