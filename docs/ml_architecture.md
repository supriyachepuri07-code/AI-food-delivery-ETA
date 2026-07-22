# Machine Learning Architecture

## 1. ML System Overview

The AI Food Delivery ETA Prediction platform uses a complete machine learning lifecycle to generate accurate delivery time predictions. The system transforms raw operational data into machine learning features, trains predictive models, deploys the best-performing model, and continuously monitors prediction quality in production.

The ML system consists of the following major stages:

- Data Collection
- Data Validation
- Data Preprocessing
- Feature Engineering
- Model Training
- Model Evaluation
- Model Registration
- Model Deployment
- Real-Time Inference
- Model Monitoring
- Continuous Retraining

The architecture is designed to ensure reproducibility, scalability, maintainability, and continuous improvement of prediction accuracy.
## 2.1 Business Problem

Food delivery platforms strive to provide customers with accurate Estimated Time of Arrival (ETA) predictions to improve user experience and operational efficiency. However, traditional ETA estimation methods often rely on static calculations or limited real-time information, resulting in inaccurate delivery estimates. These inaccuracies are caused by dynamic factors such as restaurant preparation delays, driver availability and arrival time, traffic congestion, weather conditions, route changes, and delivery distance.

Inaccurate ETA predictions lead to customer dissatisfaction, increased order cancellations, refund requests, negative reviews, inefficient driver utilization, and reduced trust in the platform. They also make it difficult for restaurants to manage food preparation and delivery coordination effectively.

The business requires an intelligent, data-driven ETA prediction system capable of continuously analyzing historical delivery patterns and real-time operational data to generate more accurate delivery time estimates. By improving ETA accuracy, the platform aims to enhance customer satisfaction, optimize delivery operations, reduce operational costs, and build long-term customer trust.
## 2.2 Machine Learning Problem

The objective of the machine learning system is to develop a supervised regression model that accurately predicts the Estimated Time of Arrival (ETA) for food deliveries before the delivery process is completed.

The model learns from historical delivery data and utilizes real-time operational information, including order details, restaurant characteristics, driver information, traffic conditions, weather conditions, and geographical distances, to estimate the expected delivery duration.

The machine learning system must generate predictions using only the information available at the time of prediction, ensuring that no future information is used during inference. This prevents data leakage and allows the model to make reliable real-time predictions.

The trained model should continuously improve as new delivery data becomes available. By leveraging periodic retraining, model monitoring, and version control, the system aims to maintain high prediction accuracy despite changing traffic patterns, seasonal variations, restaurant performance, and driver behavior.

The machine learning solution is designed to operate as a production-grade AI service that supports scalable, low-latency inference through FastAPI while maintaining reproducibility using MLflow, Airflow, Docker, GitHub Actions, and GCP.
## 2.3 Problem Type

The AI Food Delivery ETA Prediction system is designed as a supervised machine learning regression problem. The model learns from historical delivery records containing input features and the corresponding actual delivery times.

### Problem Classification

| Category | Description |
|----------|-------------|
| Learning Type | Supervised Learning |
| Problem Type | Regression |
| Prediction Target | Estimated Time of Arrival (ETA) in minutes |
| Target Variable | Continuous Numerical Value |
| Prediction Mode | Real-Time Online Prediction |
| Inference Type | Single Prediction per Delivery Request |
| Learning Approach | Batch Training with Periodic Retraining |
| Model Output | Predicted ETA (minutes) |

### Characteristics

- The model predicts a continuous numerical value representing the expected delivery time.
- Predictions are generated when a driver accepts a delivery request.
- The model is retrained periodically using newly collected historical delivery data.
- Training and inference use identical preprocessing and feature engineering pipelines to ensure consistency.
- The system is designed for low-latency predictions suitable for production deployment.
## 2.4 Target Variable

The target variable represents the value that the machine learning model is trained to predict. In this project, the target is the **Actual Estimated Time of Arrival (Actual ETA)** expressed in minutes.

### Target Definition

The Actual ETA is calculated as the total time elapsed between the moment an order is successfully placed and the moment it is delivered to the customer.

**Formula**

Actual ETA = Delivered Time − Order Time

### Target Characteristics

| Property | Description |
|----------|-------------|
| Target Name | Actual ETA |
| Data Type | Continuous Numerical Value |
| Unit | Minutes |
| Problem Type | Regression |
| Source | Delivery Service |
| Availability | Available only after delivery completion |

### Business Significance

Accurately predicting the Actual ETA enables the platform to:

- Provide reliable delivery estimates to customers.
- Improve customer satisfaction and trust.
- Reduce order cancellations and refund requests.
- Optimize restaurant preparation and driver scheduling.
- Improve overall delivery efficiency.

### Data Leakage Consideration

The Actual ETA is used **only during model training and evaluation**.

During real-time prediction, the model cannot access the actual delivery time because it has not yet occurred. Therefore, the model uses only the information available at the prediction timestamp, such as order details, driver information, traffic conditions, weather conditions, and restaurant preparation estimates.

This approach prevents data leakage and ensures realistic production predictions.
## 2.5 Input Features

The ETA prediction model uses a combination of historical delivery data and real-time operational information available at the moment of prediction. These features represent different aspects of the delivery process and are grouped into logical categories.

### 2.5.1 Feature Categories

#### Order Features
- Order ID
- Order Timestamp
- Day of Week
- Hour of Day
- Order Value
- Number of Items

#### Restaurant Features
- Restaurant ID
- Restaurant Location
- Restaurant Rating
- Average Food Preparation Time
- Cuisine Type

#### Driver Features
- Driver ID
- Driver Location
- Driver Rating
- Driver Experience
- Driver Availability Status

#### Traffic Features
- Traffic Congestion Level
- Estimated Travel Time
- Road Condition
- Traffic Incidents

#### Weather Features
- Weather Condition
- Temperature
- Rainfall
- Humidity
- Wind Speed

#### Distance Features
- Driver to Restaurant Distance
- Restaurant to Customer Distance
- Total Estimated Route Distance

### 2.5.2 Raw Features

Raw features are collected directly from operational systems or external APIs without modification.

Examples:
- Driver GPS Coordinates
- Restaurant Coordinates
- Customer Coordinates
- Weather API Response
- Traffic API Response
- Order Timestamp

### 2.5.3 Engineered Features

Engineered features are derived from raw data to improve model performance.

Examples:
- Peak Hour Indicator
- Weekend Indicator
- Meal Time Category
- Estimated Restaurant Delay
- Driver Arrival Time
- Total Estimated Travel Time
- Historical Average Delivery Time
- Distance Buckets

### 2.5.4 Excluded Features

The following information must not be used during prediction because it is unavailable before delivery completion or would introduce data leakage.

- Delivered Time
- Actual ETA
- Customer Rating
- Delivery Completion Status
- Final Delivery Duration

### 2.5.5 Feature Availability

Only features available at the prediction timestamp are used during inference. Training and inference pipelines use identical preprocessing and feature engineering logic to ensure consistency and prevent training-serving skew.
# 3. Dataset Strategy

## 3.1 Dataset Overview

The performance of the ETA prediction model depends on the quality, diversity, and completeness of the training dataset. The dataset is designed to represent the complete food delivery lifecycle by combining historical delivery records with operational and environmental information collected during each delivery.

Each record in the dataset represents a single completed delivery and contains all the information available before or during the delivery process, along with the actual delivery time used as the target variable for supervised learning.

The dataset is continuously updated as new deliveries are completed, enabling the model to learn from recent delivery patterns and adapt to changing business conditions.

### Dataset Characteristics

| Property | Description |
|----------|-------------|
| Dataset Type | Structured Tabular Dataset |
| Learning Type | Supervised Learning |
| Problem Type | Regression |
| Target Variable | Actual ETA (Minutes) |
| Record Level | One Record per Completed Delivery |
| Data Sources | Internal Databases + External APIs |
| Update Frequency | Continuous Data Collection |
| Usage | Model Training, Validation, Testing, and Monitoring |

### Objectives of the Dataset

The dataset is designed to:

- Capture historical delivery patterns.
- Represent real-world delivery scenarios.
- Support accurate ETA prediction.
- Enable feature engineering for machine learning.
- Provide sufficient diversity for model generalization.
- Support continuous model retraining as new delivery data becomes available.

### Dataset Lifecycle

The dataset follows a structured lifecycle throughout the machine learning pipeline.

```
Historical Delivery Data
        │
        ▼
Data Collection
        │
        ▼
Data Validation
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Training Dataset
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Production Deployment
```

### Design Principles

The dataset is designed according to the following principles:

- Ensure data quality through validation and preprocessing.
- Include only features available at prediction time to prevent data leakage.
- Maintain dataset versioning for reproducibility.
- Store historical datasets for future retraining.
- Preserve consistency between training and inference data.
- Support scalability for increasing delivery volumes.
- Enable integration with Airflow, MLflow, and cloud storage.
## 3.2 Dataset Sources

The ETA prediction model is trained using data collected from multiple internal systems and external services. Each data source contributes unique information required to accurately estimate food delivery time.

The machine learning dataset is created by integrating operational business data with real-time environmental data and historical delivery records.

### Internal Data Sources

| Source | Description | Example Features |
|---------|-------------|------------------|
| Customer Service | Customer order information | Customer ID, Order Value, Number of Items |
| Order Management System | Order lifecycle events | Order Time, Order Status, Payment Status |
| Restaurant Management System | Restaurant information | Restaurant ID, Cuisine Type, Restaurant Rating, Average Preparation Time |
| Driver Management System | Driver information | Driver ID, Driver Rating, Driver Availability, Experience |
| Delivery Tracking System | Historical delivery information | Pickup Time, Delivery Time, Actual ETA |
| GPS Tracking System | Driver location updates | Driver Latitude, Driver Longitude |

### External Data Sources

| Source | Description | Example Features |
|---------|-------------|------------------|
| Google Maps API | Route and distance information | Route Distance, Estimated Travel Time |
| Traffic API | Live traffic conditions | Congestion Level, Road Incidents |
| Weather API | Current weather conditions | Temperature, Rainfall, Humidity, Weather Condition |
| Calendar Service | Date-related information | Weekend Indicator, Public Holiday, Festival |

### Future Data Sources

The platform is designed to support additional data sources as the business grows.

Potential future integrations include:

- Live restaurant kitchen status
- Road closure notifications
- City event schedules
- Fuel price information
- Driver behavior analytics
- Customer feedback
- Demand forecasting systems

### Data Integration Strategy

The training dataset is created by joining data from multiple sources using common business identifiers such as:

- Order ID
- Driver ID
- Restaurant ID
- Customer ID
- Delivery ID

External data is synchronized using timestamps and geographical coordinates to ensure that traffic and weather conditions correspond to the actual delivery period.

### Source Reliability

To maintain data quality, every data source is monitored for:

- Availability
- Data completeness
- Data freshness
- Schema consistency
- API response status
- Missing or delayed records

Any unavailable external service should trigger fallback mechanisms or data quality alerts before the data is used for model training or inference.
## 3.3 Dataset Composition

The training dataset is a structured tabular dataset where each row represents a single completed food delivery. Every record combines historical delivery information with operational and environmental data available during the delivery process.

The dataset is organized into multiple feature groups, with each group representing a specific aspect of the delivery lifecycle. These features are combined to create a comprehensive representation of each delivery before being used for machine learning model training.

### Dataset Structure

| Feature Group | Description |
|--------------|-------------|
| Order Features | Information related to customer orders |
| Restaurant Features | Restaurant details and food preparation information |
| Driver Features | Driver profile, availability, and location information |
| Traffic Features | Real-time traffic conditions and travel estimates |
| Weather Features | Environmental conditions during delivery |
| Distance Features | Geographic distances between delivery points |
| Time Features | Date and time-based information |
| Engineered Features | Features created during feature engineering |
| Target Variable | Actual ETA (Minutes) |

### Example Dataset Schema

| Feature Category | Example Columns |
|------------------|-----------------|
| Order Features | Order ID, Order Time, Order Value, Number of Items |
| Restaurant Features | Restaurant ID, Cuisine Type, Restaurant Rating, Average Preparation Time |
| Driver Features | Driver ID, Driver Rating, Driver Experience, Driver Availability |
| Traffic Features | Congestion Level, Estimated Travel Time |
| Weather Features | Temperature, Humidity, Rainfall, Weather Condition |
| Distance Features | Driver-to-Restaurant Distance, Restaurant-to-Customer Distance |
| Time Features | Hour of Day, Day of Week, Weekend Indicator |
| Engineered Features | Peak Hour Indicator, Estimated Restaurant Delay, Total Estimated Route Time |
| Target Variable | Actual ETA |

### Dataset Characteristics

| Property | Value |
|----------|-------|
| Record Granularity | One Completed Delivery |
| Feature Types | Numerical, Categorical, Boolean, Datetime |
| Target Type | Continuous Numerical |
| Missing Values | Handled during preprocessing |
| Duplicate Records | Removed during validation |
| Feature Engineering | Applied before model training |

### Dataset Design Principles

The dataset is designed to:

- Represent the complete delivery process.
- Capture operational, geographical, and environmental factors.
- Include only features available at prediction time.
- Support scalable feature engineering.
- Maintain consistency between training and inference datasets.
- Prevent data leakage by excluding post-delivery information except for the target variable.
## 3.4 Data Collection Strategy

The AI Food Delivery ETA Prediction system collects data from multiple internal services and external providers to build a comprehensive machine learning dataset. Data collection is designed to ensure that every completed delivery contributes to improving future ETA predictions.

The data collection process combines batch and real-time ingestion mechanisms. Operational data generated during the delivery lifecycle is continuously captured and stored in the operational database, while external APIs provide dynamic information such as traffic conditions, weather, and route details.

### Data Collection Workflow

1. A customer places a food order.
2. The order is stored in the Order Management System.
3. A driver is assigned to the delivery.
4. Driver location, restaurant information, and order details are recorded.
5. Real-time traffic and weather information are retrieved using external APIs.
6. Delivery events are tracked until the order is completed.
7. The completed delivery record is stored in the historical delivery database.
8. Airflow periodically extracts newly completed delivery records.
9. The extracted data is validated, cleaned, and transformed into a training-ready dataset.
10. The processed dataset is versioned and stored for future model training.

### Data Collection Frequency

| Data Source | Collection Frequency |
|-------------|----------------------|
| Order Data | Real-Time |
| Driver Data | Real-Time |
| GPS Location | Real-Time |
| Restaurant Data | Real-Time / Event-Based |
| Traffic Information | Real-Time |
| Weather Information | Real-Time |
| Historical Deliveries | Batch (Periodic) |

### Data Collection Principles

The data collection process follows these principles:

- Capture only reliable and validated data.
- Preserve historical delivery records for future retraining.
- Synchronize data from multiple sources using timestamps and unique identifiers.
- Maintain data consistency across all systems.
- Handle missing or delayed data gracefully.
- Log all ingestion activities for auditing and troubleshooting.

### Data Storage Flow

Raw operational data is first stored in the operational database. After validation and preprocessing, the data is transformed into structured machine learning datasets and stored in cloud storage for model training. Each dataset version is tracked to ensure reproducibility and support continuous model improvement.
## 3.5 Dataset Splitting Strategy

The machine learning dataset is divided into separate subsets for training, validation, and testing. This ensures that the model is trained, tuned, and evaluated on different data, allowing an unbiased assessment of its performance before deployment.

Since food delivery data is generated continuously over time, the dataset is split chronologically rather than randomly. This approach better reflects real-world production scenarios, where the model is trained on historical deliveries and used to predict future deliveries.

### Dataset Split

| Dataset | Purpose | Percentage |
|----------|----------|------------|
| Training Dataset | Model Training | 70% |
| Validation Dataset | Hyperparameter Tuning and Model Selection | 15% |
| Test Dataset | Final Performance Evaluation | 15% |

### Splitting Strategy

The dataset is divided based on delivery timestamps:

- The oldest historical records are used for training.
- More recent records are used for validation.
- The latest records are reserved for final testing.

This chronological approach prevents future information from leaking into the training process and provides a realistic estimate of production performance.

### Dataset Splitting Workflow

1. Collect completed delivery records.
2. Sort the dataset by delivery timestamp.
3. Create the training dataset from the earliest records.
4. Create the validation dataset from the next set of records.
5. Reserve the most recent records as the test dataset.
6. Store each dataset version separately for reproducibility.

### Data Leakage Prevention

The following practices are followed to prevent data leakage:

- No future delivery information is included in the training dataset.
- The target variable (Actual ETA) is never used as an input feature.
- Training and inference use the same preprocessing and feature engineering pipeline.
- Validation and test datasets remain unseen during model training.

### Benefits

This strategy:

- Simulates real-world production conditions.
- Produces reliable model evaluation results.
- Supports reproducible experiments.
- Reduces the risk of overly optimistic performance estimates.
- Enables fair comparison between different model versions.