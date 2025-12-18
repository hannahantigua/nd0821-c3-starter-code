# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model type: Logistic Regression

Implementation: scikit-learn (LogisticRegression with max_iter=1000)

Input features: 14 features from the U.S. Census Adult dataset, including categorical features such as workclass, education, and occupation

## Intended Use
Purpose: Predict whether an individual earns more than $50K per year based on census data

Users: Researchers, students, or practitioners exploring structured data classification and model evaluation

Scope: Intended for educational and experimental purposes only

## Training Data
Dataset: U.S. Census Adult dataset

Source: UCI Machine Learning Repository

Preprocessing: Categorical features were one-hot encoded; label column cleaned to remove leading/trailing spaces

## Evaluation Data
Test set: 20% split of the cleaned dataset

Slice-based evaluation: Model performance was also evaluated on slices of categorical features to detect subgroup differences

## Metrics
Overall Performance:
Precision: 0.728
Recall:    0.553
F1 Score:  0.629

## Ethical Considerations
The model may reflect historical biases in income based on race, sex, or occupation

Predictions should not be used for real-world hiring, lending, or legal decisions

## Caveats and Recommendations
The dataset is limited to U.S. adults and may not generalize globally

Categorical features are limited; model does not capture all factors influencing income

Slice-based evaluation is recommended to check fairness across subgroups

Use this model for educational purposes and experimentation only
