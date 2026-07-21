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
## 5. Training Pipeline

The Training Pipeline is responsible for building, evaluating, and preparing machine learning models for production deployment. The pipeline is executed through Apache Airflow and follows a sequence of well-defined stages.

### Training Pipeline Stages

#### 1. Data Ingestion
Collect historical delivery data and supporting datasets from configured data sources.

#### 2. Data Validation
Validate schema, data quality, missing values, duplicate records, and business rules before processing.

#### 3. Data Preprocessing
Clean the validated dataset by handling missing values, removing duplicates, standardizing formats, and preparing the data for feature engineering.

#### 4. Feature Engineering
Transform raw attributes into machine learning features such as delivery distance, peak-hour indicators, meal-time categories, and encoded categorical variables.

#### 5. Model Training
Train multiple regression algorithms using the engineered feature dataset.

#### 6. Model Evaluation
Evaluate the trained models using predefined regression metrics and select the best-performing model.

#### 7. Model Registry
Register the selected model in MLflow with version information, evaluation metrics, and metadata for future deployment.

#### 8. Deployment Preparation
Prepare the approved model artifact for use by the online prediction service.
## 6. Prediction Pipeline

The Prediction Pipeline is responsible for generating real-time ETA predictions for incoming customer orders. Unlike the training pipeline, this workflow does not retrain the model. Instead, it loads the latest approved model and uses it to serve prediction requests through the FastAPI service.

### Prediction Pipeline Stages

#### 1. Prediction Request
Receive a prediction request from the application after the restaurant accepts the order and a delivery partner is assigned.

#### 2. Input Validation
Validate that all required input fields are present and contain valid values before processing the request.

#### 3. Feature Collection
Collect the operational information required for prediction, including restaurant details, driver information, customer location, traffic conditions, weather information, and delivery distance.

#### 4. Feature Engineering
Apply the same preprocessing and feature engineering transformations that were used during model training.

#### 5. Model Loading
Load the latest production-ready model from the Model Registry.

#### 6. ETA Prediction
Generate the estimated delivery time using the trained regression model.

#### 7. API Response
Return the predicted ETA to the requesting application as a JSON response.
## 7. External Integrations

The AI Food Delivery ETA Prediction System integrates with several external services to enrich prediction data, automate workflows, and support production deployment.

| External Service | Purpose | Data Provided |
|------------------|---------|---------------|
| Google Maps API | Calculate delivery distance, route information, and traffic conditions | Distance, travel time, traffic |
| Weather API | Retrieve current weather information affecting delivery | Temperature, rainfall, weather condition, wind speed |
| PostgreSQL | Store operational and historical business data | Orders, restaurants, drivers, delivery records |
| MLflow | Track machine learning experiments and store model versions | Model artifacts, metrics, parameters |
| GitHub | Source code version control | Project source code |
| GitHub Actions | Continuous Integration and Continuous Deployment (CI/CD) | Automated testing and deployment |
| Google Cloud Platform (GCP) | Host the production environment | APIs, containers, databases, workflow services |

### Integration Principles

- External services are accessed through dedicated service layers rather than directly by the machine learning model.
- The prediction service validates responses from external services before using them.
- The system should log integration failures and implement fallback strategies where possible to maintain service availability.
## 8. Deployment Architecture

The AI Food Delivery ETA Prediction System follows a containerized and cloud-native deployment architecture to ensure scalability, portability, and maintainability.

### Deployment Workflow

1. Developers push code changes to the GitHub repository.
2. GitHub Actions automatically executes the CI/CD pipeline.
3. Automated tests and code quality checks are performed.
4. A Docker image is built for the application.
5. The Docker image is pushed to a container registry.
6. The application is deployed to Google Cloud Platform (GCP).
7. FastAPI serves prediction requests, while Airflow manages scheduled workflows.
8. PostgreSQL stores operational data, and MLflow manages model versions and experiment tracking.

### Deployment Components

| Component | Purpose |
|-----------|---------|
| GitHub | Source code management |
| GitHub Actions | CI/CD automation |
| Docker | Containerization |
| Container Registry | Store Docker images |
| Google Cloud Platform | Production hosting |
| FastAPI | Real-time prediction service |
| Airflow | Workflow orchestration |
| PostgreSQL | Data storage |
| MLflow | Model management |
## 9. Monitoring and Logging

The system continuously monitors application health, machine learning performance, and infrastructure to ensure reliable operation.

### Monitoring Objectives

- Monitor API availability and response times.
- Monitor Airflow workflow execution.
- Monitor model prediction performance.
- Track data quality issues.
- Detect pipeline failures.
- Monitor cloud resource utilization.

### Logging

The following events are logged:

- API requests and responses
- Model prediction events
- Training pipeline execution
- Data validation errors
- Application exceptions
- Deployment events

Logs help developers identify issues, troubleshoot failures, and improve system reliability.