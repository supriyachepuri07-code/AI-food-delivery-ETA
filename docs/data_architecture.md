# Data Architecture

## 1. Data Overview

The AI Food Delivery ETA Prediction System relies on a combination of operational business data, historical delivery records, external data sources, and machine learning generated data to accurately predict delivery times.

The data architecture is designed to support two primary objectives:

1. Efficient day-to-day business operations for the food delivery platform.
2. Reliable machine learning model training and real-time ETA prediction.

The platform continuously collects and processes data from multiple sources, including customers, restaurants, delivery partners, historical orders, traffic services, and weather services. This data is validated, cleaned, transformed, and enriched before being used for machine learning model training and real-time inference.

The system distinguishes between operational data used for business transactions and analytical data used for machine learning. Operational data supports order management, driver assignment, and delivery tracking, while analytical data supports feature engineering, model training, performance evaluation, and prediction services.

The overall data architecture is designed to ensure data quality, consistency, scalability, and reproducibility across both the training and prediction pipelines.
## 2. Data Sources

The ETA prediction platform collects data from multiple internal and external sources. Each source provides specific information required for business operations, machine learning training, or real-time prediction.

| Data Source | Category | Data Provided | Access Method | Update Frequency |
|-------------|----------|---------------|---------------|------------------|
| Customer Service | Internal | Customer profile and delivery address | PostgreSQL | Real-time |
| Restaurant Service | Internal | Restaurant information, preparation time, location | PostgreSQL | Real-time |
| Driver Service | Internal | Driver information, availability, GPS location | PostgreSQL | Real-time |
| Order Service | Internal | Order status, timestamps, order details | PostgreSQL | Real-time |
| Historical Delivery Dataset | Internal | Historical delivery records for model training | CSV / Data Lake | Batch |
| Google Maps API | External | Distance, route information, traffic conditions | REST API | Real-time |
| Weather API | External | Weather conditions, temperature, rainfall | REST API | Real-time |
| MLflow | Internal | Model metadata, experiment tracking, model versions | MLflow API | On-demand |
## 3. Core Entities

The ETA prediction platform is built around a set of core business entities that represent the primary objects managed by the system. These entities capture operational information required for business transactions and machine learning.

| Entity | Description |
|---------|-------------|
| Customer | Represents users who place food delivery orders. |
| Restaurant | Represents restaurants that prepare customer orders. |
| Driver | Represents delivery partners responsible for delivering orders. |
| Order | Represents a food order placed by a customer and accepted by a restaurant. |
| Delivery | Represents the complete delivery process from restaurant pickup to customer delivery. |
| Weather | Represents weather conditions associated with a delivery request. |
| Traffic | Represents traffic conditions affecting travel time. |
| ETA Prediction | Represents machine learning generated delivery time predictions and related metadata. |

These entities form the foundation of the operational database and provide the information required for machine learning feature engineering and ETA prediction.
## 4. Entity Relationships

The core business entities are interconnected to support order processing, delivery operations, and machine learning predictions.

### Relationship Summary

| Parent Entity | Child Entity | Relationship |
|---------------|--------------|--------------|
| Customer | Order | One Customer can place many Orders |
| Restaurant | Order | One Restaurant can receive many Orders |
| Driver | Delivery | One Driver can complete many Deliveries |
| Order | Delivery | One Order has one Delivery |
| Delivery | ETA Prediction | One Delivery has one ETA Prediction record |
| Weather | Delivery | One Weather record can be associated with multiple Deliveries |
| Traffic | Delivery | One Traffic record can be associated with multiple Deliveries |

These relationships ensure that business operations remain consistent while providing complete historical information for machine learning model training and performance analysis.
## 5. Logical Database Design

The ETA prediction platform uses a relational database to store operational business data. Each table has a single responsibility and represents one core business entity.

### Database Tables

| Table | Purpose | Data Owner |
|--------|---------|------------|
| customers | Stores customer profile and delivery address information. | Customer Service |
| restaurants | Stores restaurant details, location, and preparation-related information. | Restaurant Service |
| drivers | Stores delivery partner details, vehicle information, and availability. | Driver Service |
| orders | Stores customer order information and order lifecycle events. | Order Service |
| deliveries | Stores pickup, transit, and delivery execution details. | Delivery Service |
| weather | Stores weather snapshots associated with delivery locations and times. | Weather Service |
| traffic | Stores traffic conditions and route-related information. | Google Maps Service |
| eta_predictions | Stores ETA predictions, prediction timestamps, confidence scores, model versions, and actual delivery outcomes for monitoring and analysis. | AI Prediction Service |

### Design Principles

- Each table represents a single business entity.
- Data duplication is minimized through normalization.
- Relationships between tables are maintained using primary and foreign keys.
- Operational data and machine learning prediction data are logically separated.
- The schema is designed to support both transactional operations and analytical workloads.
## 6. Physical Database Design

The AI Food Delivery ETA Prediction System uses PostgreSQL as the primary relational database for storing operational data. The database schema is designed to support business operations, machine learning workflows, and future scalability.

---

### 6.1 Customers Table

**Purpose**

Stores customer profile information and delivery location details.

| Column | Data Type | Constraints | Description |
|---------|-----------|------------|-------------|
| customer_id | UUID | Primary Key | Unique customer identifier |
| full_name | VARCHAR(100) | NOT NULL | Customer full name |
| phone_number | VARCHAR(15) | UNIQUE, NOT NULL | Customer mobile number |
| email | VARCHAR(100) | UNIQUE | Customer email address |
| latitude | DECIMAL(9,6) | NOT NULL | Current latitude |
| longitude | DECIMAL(9,6) | NOT NULL | Current longitude |
| created_at | TIMESTAMP | NOT NULL | Registration timestamp |

---

### 6.2 Restaurants Table

**Purpose**

Stores restaurant information and preparation-related details.

| Column | Data Type | Constraints | Description |
|---------|-----------|------------|-------------|
| restaurant_id | UUID | Primary Key | Unique restaurant identifier |
| restaurant_name | VARCHAR(150) | NOT NULL | Restaurant name |
| cuisine_type | VARCHAR(50) | NOT NULL | Cuisine category |
| latitude | DECIMAL(9,6) | NOT NULL | Restaurant latitude |
| longitude | DECIMAL(9,6) | NOT NULL | Restaurant longitude |
| average_preparation_time | INTEGER | NOT NULL | Average preparation time (minutes) |
| restaurant_rating | DECIMAL(2,1) | | Average customer rating |
| created_at | TIMESTAMP | NOT NULL | Record creation timestamp |

---

### 6.3 Drivers Table

**Purpose**

Stores delivery partner information.

| Column | Data Type | Constraints | Description |
|---------|-----------|------------|-------------|
| driver_id | UUID | Primary Key | Unique driver identifier |
| full_name | VARCHAR(100) | NOT NULL | Driver name |
| phone_number | VARCHAR(15) | UNIQUE | Driver phone number |
| vehicle_type | VARCHAR(30) | NOT NULL | Bike, Scooter, Car |
| driver_rating | DECIMAL(2,1) | | Driver rating |
| current_latitude | DECIMAL(9,6) | | Current latitude |
| current_longitude | DECIMAL(9,6) | | Current longitude |
| availability_status | BOOLEAN | DEFAULT TRUE | Driver availability |
| created_at | TIMESTAMP | NOT NULL | Registration timestamp |

---

### 6.4 Orders Table

**Purpose**

Stores customer order information.

| Column | Data Type | Constraints | Description |
|---------|-----------|------------|-------------|
| order_id | UUID | Primary Key | Unique order identifier |
| customer_id | UUID | Foreign Key | References Customers |
| restaurant_id | UUID | Foreign Key | References Restaurants |
| order_amount | DECIMAL(10,2) | NOT NULL | Total order value |
| payment_method | VARCHAR(30) | | Payment type |
| payment_status | VARCHAR(30) | | Paid / Pending |
| order_status | VARCHAR(30) | | Placed / Preparing / Cancelled |
| order_time | TIMESTAMP | NOT NULL | Order placement time |

---

### 6.5 Deliveries Table

**Purpose**

Tracks the delivery lifecycle.

| Column | Data Type | Constraints | Description |
|---------|-----------|------------|-------------|
| delivery_id | UUID | Primary Key | Unique delivery identifier |
| order_id | UUID | Foreign Key | References Orders |
| driver_id | UUID | Foreign Key | References Drivers |
| assigned_time | TIMESTAMP | | Driver assignment time |
| driver_arrival_time | TIMESTAMP | | Driver reaches restaurant |
| pickup_time | TIMESTAMP | | Food pickup time |
| delivered_time | TIMESTAMP | | Delivery completion time |
| delivery_status | VARCHAR(30) | | Assigned / Picked / Delivered |

---

### 6.6 Weather Table

**Purpose**

Stores weather conditions used for ETA prediction.

| Column | Data Type | Constraints | Description |
|---------|-----------|------------|-------------|
| weather_id | UUID | Primary Key | Weather record identifier |
| temperature | DECIMAL(5,2) | | Temperature (°C) |
| humidity | DECIMAL(5,2) | | Humidity (%) |
| rainfall | DECIMAL(5,2) | | Rainfall (mm) |
| wind_speed | DECIMAL(5,2) | | Wind speed |
| weather_condition | VARCHAR(50) | | Sunny, Rainy, Cloudy |
| observation_time | TIMESTAMP | NOT NULL | Weather observation time |

---

### 6.7 Traffic Table

**Purpose**

Stores traffic conditions for delivery routes.

| Column | Data Type | Constraints | Description |
|---------|-----------|------------|-------------|
| traffic_id | UUID | Primary Key | Traffic record identifier |
| congestion_level | VARCHAR(20) | | Low / Medium / High |
| estimated_travel_time | INTEGER | | Travel time (minutes) |
| distance_km | DECIMAL(6,2) | | Route distance |
| observation_time | TIMESTAMP | NOT NULL | Observation timestamp |

---

### 6.8 ETA Predictions Table

**Purpose**

Stores machine learning predictions and actual outcomes.

| Column | Data Type | Constraints | Description |
|---------|-----------|------------|-------------|
| prediction_id | UUID | Primary Key | Prediction identifier |
| delivery_id | UUID | Foreign Key | References Deliveries |
| model_version | VARCHAR(30) | NOT NULL | ML model version |
| predicted_eta_minutes | INTEGER | NOT NULL | Predicted delivery time |
| actual_eta_minutes | INTEGER | | Actual delivery time |
| prediction_error_minutes | INTEGER | | Difference between prediction and actual |
| prediction_timestamp | TIMESTAMP | NOT NULL | Prediction generation time |

---

### Design Standards

- All primary keys use UUIDs.
- Foreign keys maintain referential integrity.
- Timestamps are stored in UTC.
- Nullable fields are used only when information is unavailable during earlier stages of the delivery lifecycle.
- Business entities are normalized to reduce data duplication.
- The schema is designed to support both operational workloads and machine learning analytics.
## 7. Data Dictionary

The Data Dictionary defines the meaning, source, format, and intended usage of the key data fields used throughout the ETA prediction platform.

Each field should include:

- Business Definition
- Data Type
- Source System
- Example Value
- Validation Rules
- Usage (Operational / Machine Learning / API)

### Example Fields

| Field Name | Description | Source | Example | Used In |
|------------|-------------|--------|---------|---------|
| order_time | Time when the customer successfully places an order | Order Service | 2026-07-22 12:30:45 | Feature Engineering |
| assigned_time | Time when a driver accepts the delivery | Delivery Service | 2026-07-22 12:34:10 | ETA Calculation |
| average_preparation_time | Average time taken by a restaurant to prepare an order | Restaurant Service | 18 minutes | ETA Prediction |
| congestion_level | Current traffic level on the delivery route | Google Maps API | High | Feature Engineering |
| weather_condition | Current weather at the delivery location | Weather API | Rainy | Feature Engineering |
| predicted_eta_minutes | Predicted delivery duration generated by the AI model | Prediction Service | 28 | Customer Application |
| actual_eta_minutes | Actual delivery duration after order completion | Delivery Service | 31 | Model Evaluation |
## 8. Feature Mapping

The machine learning model does not consume raw database fields directly. Instead, selected fields are transformed into meaningful features through preprocessing and feature engineering.

### Feature Categories

| Category | Example Features |
|----------|------------------|
| Order Features | Order hour, day of week, meal period, order amount |
| Restaurant Features | Average preparation time, restaurant rating, cuisine type |
| Driver Features | Driver rating, distance to restaurant, availability status |
| Environmental Features | Traffic level, weather condition, temperature, rainfall, delivery distance |

### Example Feature Mapping

| Database Field | Engineered Feature | Purpose |
|----------------|--------------------|---------|
| order_time | Hour of Day | Capture time-based demand patterns |
| order_time | Day of Week | Capture weekly trends |
| order_time | Peak Hour Flag | Identify busy delivery periods |
| order_time | Meal Period | Breakfast, Lunch, Dinner classification |
| restaurant_latitude + restaurant_longitude | Restaurant Location | Used to calculate delivery distance |
| customer_latitude + customer_longitude | Customer Location | Used to calculate delivery distance |
| driver_latitude + driver_longitude | Driver-to-Restaurant Distance | Estimate driver arrival time |
| average_preparation_time | Preparation Time | Estimate restaurant readiness |
| congestion_level | Traffic Level | Measure road congestion impact |
| weather_condition | Weather Category | Capture weather effects on delivery |
| delivered_time - order_time | Actual ETA | Target variable used during model training |

### Feature Engineering Principles

- Maintain identical feature engineering logic for both training and prediction.
- Normalize or encode features where appropriate.
- Avoid data leakage by excluding information unavailable at prediction time.
- Version feature engineering logic alongside model versions.
## 9. Data Validation Rules

To ensure high-quality machine learning predictions and reliable business operations, all incoming data must pass validation before being stored or used for model training and inference.

### Validation Principles

- Mandatory fields must not be null.
- Numeric values must fall within acceptable ranges.
- Timestamp fields must follow logical chronological order.
- Categorical values must match predefined allowed values.
- Geographic coordinates must be valid.
- Invalid records should be logged for investigation.

### Example Validation Rules

| Data Field | Validation Rule |
|------------|-----------------|
| Latitude | Must be between -90 and 90 |
| Longitude | Must be between -180 and 180 |
| Restaurant Rating | Must be between 0 and 5 |
| Driver Rating | Must be between 0 and 5 |
| Order Amount | Must be greater than 0 |
| Average Preparation Time | Must be greater than 0 |
| Temperature | Must be within a realistic range |
| Humidity | Must be between 0 and 100 |
| Delivery Time | Must occur after Pickup Time |
| Actual ETA | Must be greater than 0 |

### Validation Failure Handling

- Reject invalid records from the prediction pipeline.
- Log validation failures for monitoring.
- Notify upstream services if critical validation errors occur.
- Maintain validation reports for auditing and troubleshooting.