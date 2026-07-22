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