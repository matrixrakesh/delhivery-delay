# Last-Mile Delivery Delay Prediction & Hub Analytics

A Python machine learning project that analyzes shipment data, estimates delivery time, identifies shipments at risk of delay, and segments delivery hubs to support operational decisions.

> **Data note:** This project uses synthetic shipment data generated within the repository. It does not use or represent actual Delhivery shipment records. The Delhivery reference is used only to describe the last-mile delivery use case.

## Project Overview

Last-mile delivery performance can be affected by factors such as shipment distance, promised delivery time, number of stops, hub capacity, service type, weather, and sale periods.

This project brings together data preparation, exploratory data analysis, regression, classification, and clustering in one workflow.

The pipeline addresses three main questions:

1. **Delivery time estimation:** How many hours might a shipment take to be delivered?
2. **Delay risk prediction:** Is a shipment likely to miss its promised delivery time?
3. **Hub performance analysis:** Which hubs may need additional operational attention?

A shared preparation module, `src/shipment_prep.py`, is used for data cleaning and feature creation so that training and prediction follow the same preprocessing steps.

## Key Features

- Generates a reproducible synthetic shipment dataset.
- Cleans shipment records, handles missing values and duplicates, and standardizes service categories.
- Performs exploratory data analysis to identify patterns related to delivery delays.
- Trains and compares regression models for delivery-time estimation.
- Trains classification models to identify shipments at risk of delay.
- Selects useful features and evaluates different alert thresholds.
- Uses K-Means clustering to group hubs by delivery performance.
- Scores new shipments and assigns suggested follow-up actions.
- Saves model files and analysis outputs for review.

## Technology Stack

- **Language:** Python 3.11+
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn
- **Model persistence:** joblib

The project uses free Python libraries and runs locally.

## Project Workflow

| Stage | Work performed | Main script |
|---|---|---|
| Data generation | Creates the synthetic shipment and hub data | `generate_data.py` |
| Data preparation and EDA | Cleans data, checks quality, and explores patterns | `part_a_clean_eda.py` |
| Regression | Compares models to estimate delivery hours | `part_b_regression.py` |
| Classification | Builds and evaluates shipment delay classifiers | `part_c_classification.py` |
| Hub segmentation | Groups hubs and prepares operational segments | `part_d_segments.py` |
| Model packaging | Saves the selected prediction model | `package_model.py` |
| Prediction | Scores new shipments and creates suggested actions | `predict_delay.py` |

## Getting Started

### Requirements

- Python 3.11 or newer
- pip

### Installation

Clone the repository and move into the project directory:

```bash
git clone <your-repository-url>
cd <repository-folder>
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Run the project

Run the scripts in this order:

```bash
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

Each stage creates files used by later stages, so the order matters.

The model is saved at `models/delay_model.pkl`. The project pins scikit-learn 1.8.0 in `requirements.txt`. If you use a different version, regenerate the model by running `python package_model.py` before running prediction.

The scripts use fixed random states to make the generated data and model experiments reproducible.

## Data Preparation and Exploratory Analysis

The generated raw dataset contains 1,224 rows. After removing 24 duplicate records, 1,200 shipments remain. There are 278 delayed shipments, giving a delay rate of approximately **23.17%**.

Data preparation includes:

- Handling missing distance, parcel weight, declared value, and payment mode values.
- Treating the sentinel weight value of 999 kg as missing.
- Standardizing inconsistent service-type values.
- Combining shipment records with hub information.
- Creating model features from the cleaned data.

After cleaning, there are no missing values in the 17 model features, `delivery_hours`, and `is_delayed`.

### Some findings from the data

- Express shipments have a delay rate of approximately **48.97%**.
- Standard shipments have a delay rate of approximately **20.56%**.
- Economy shipments have a delay rate of approximately **8.57%**.
- Delay rates increase across the hub tiers in this generated dataset: 4% for Tier 1, 21.5% for Tier 2, and 44% for Tier 3.
- Longer shipment distances are associated with higher delay rates. The rate increases from approximately 2.19% for 0–200 km to 85.23% for shipments over 600 km.
- Promised delivery hours, distance, and number of stops have the strongest positive correlations with delivery time among the analyzed features.

Detailed checks and EDA summaries are saved in the `outputs` directory.

## Model Development and Results

### 1. Delivery Time Estimation — Regression

The data was split into 840 training rows, 180 validation rows, and 180 test rows. The scaler was fitted on the training data only to avoid data leakage.

The following models were compared on the validation set:

| Model | Validation RMSE (hours) | Validation R² |
|---|---:|---:|
| Linear Regression | 4.8912 | 0.9313 |
| Gradient Boosting | 4.9165 | 0.9306 |
| Random Forest | 5.5095 | 0.9128 |
| Decision Tree | 6.2796 | 0.8867 |

Linear Regression had the lowest validation RMSE, with Gradient Boosting performing very closely. In five-fold cross-validation, Gradient Boosting achieved a mean R² of 0.9353, compared with 0.9261 for Linear Regression.

Following the lowest-validation-RMSE selection rule, Linear Regression was evaluated on the test set:

| Metric | Test result |
|---|---:|
| RMSE | 4.5235 hours |
| MAE | 3.5671 hours |
| R² | 0.9493 |

The difference between the validation ranking and cross-validation ranking is worth noting. Further evaluation on a larger and more representative dataset would help confirm which model is the better choice.

### 2. Shipment Delay Prediction — Classification

Logistic Regression, Decision Tree, and Random Forest were evaluated. The final Random Forest experiment used these seven selected features:

- `distance_km`
- `hub_tier`
- `hub_daily_capacity`
- `n_stops`
- `promised_hours`
- `rain_flag`
- `sale_event_flag`

A delay probability threshold of **0.5** was selected because it was the highest tested threshold that achieved the validation recall requirement of at least 0.80.

Test-set results at this threshold:

| Metric | Test result |
|---|---:|
| Accuracy | 0.8667 |
| Precision | 0.6889 |
| Recall | 0.7561 |
| F1 score | 0.7209 |

The test recall is below the validation target of 0.80. This result is reported as measured rather than treating the validation target as achieved on unseen data.

### 3. Hub Segmentation — K-Means

K-Means was used to group hubs based on their performance profiles. Three segments were selected:

| Segment | Number of hubs | Segment delay rate |
|---|---:|---:|
| Escalate | 9 | 46.00% |
| Watch | 8 | 15.50% |
| Standard | 7 | 2.57% |

The analysis also identified cases where the hub tier did not match the observed segment. For example, Nagpur and Kochi are Tier 2 hubs in the Escalate segment, while Kolkata is a Tier 1 hub in the Watch segment. This shows why hub performance data can provide additional information beyond a static tier label.

## New Shipment Scoring

The prediction workflow processes 48 generated new shipments:

- 8 shipments are flagged by the selected delay-risk threshold.
- 6 flagged shipments associated with Escalate hubs receive a suggested hub-manager escalation.
- 2 other flagged shipments receive a suggested proactive customer call.
- The remaining 40 shipments do not receive an action under the current rules.

These are example outputs from the synthetic dataset, not live operational recommendations.

## Monitoring Considerations

For a real deployment, the following measures could be monitored:

- Weekly shipment delay rate.
- Percentage of shipments flagged by the model.
- Recall after actual delivery outcomes become available.
- Missing-value rates for important incoming fields such as distance and parcel weight.
- Changes in shipment distance, number of stops, service-type mix, and hub performance.

The current project proposes monthly model retraining and quarterly hub-segment refreshes. These are suggested monitoring intervals and would need to be validated against real operational requirements.

## Repository Outputs

The scripts generate analysis and evaluation files in the `outputs` directory, including:

- Data quality checks and EDA summaries.
- Regression and classification metrics.
- Feature-selection and threshold comparison results.
- Hub segment profiles.
- Predictions and suggested actions for new shipments.

The `models` directory contains the packaged model and supporting files used by the prediction workflow.

## Limitations

- The dataset is synthetic and may not reflect the complexity or distribution of real shipment operations.
- The test set contains 41 delayed shipments, so classification metrics may vary with a different sample.
- The classification threshold and model configuration were selected using one validation split.
- Regression model rankings differ slightly between validation RMSE and cross-validation R².
- Hub profiles are based on 50 generated shipments per hub, so the segments may change with more data.
- The saved model depends on the scikit-learn version used to create it.

## Summary

This project demonstrates an end-to-end machine learning workflow for a last-mile delivery use case, from synthetic data generation and preprocessing to model evaluation, hub segmentation, and shipment scoring. It combines multiple ML techniques and includes evaluation results, reproducible scripts, and practical monitoring considerations.