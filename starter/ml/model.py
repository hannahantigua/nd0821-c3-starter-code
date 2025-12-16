import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from .data import process_data

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)


def compute_model_metrics_on_slices(df, cat_features, model, encoder, lb, label="salary"):
    """
    Computes and prints model performance metrics for slices of categorical features.

    Inputs
    ------
    df : pd.DataFrame
        Full dataset
    cat_features : list[str]
        List of categorical features to slice on
    model : trained ML model
    encoder : trained OneHotEncoder
    lb : trained LabelBinarizer
    label : str
        Name of the target column
    """
    for feature in cat_features:
        print(f"\nFeature: {feature}")
        for value in df[feature].unique():
            # Filter for this slice
            slice_df = df[df[feature] == value]
            if slice_df.empty:
                continue

            # Process the data
            X_slice, y_slice, _, _ = process_data(
                slice_df,
                categorical_features=cat_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb
            )

            # Make predictions
            preds = inference(model, X_slice)

            # Compute metrics
            precision, recall, f1 = compute_model_metrics(y_slice, preds)

            print(f"  Value: {value} | Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

def compute_model_metrics_overall(df, cat_features, model, encoder, lb, label="salary"):
    """
    Computes and prints overall model performance metrics on a dataset.

    Inputs
    ------
    df : pd.DataFrame
        Dataset to evaluate
    cat_features : list[str]
        List of categorical features
    model : trained ML model
    encoder : trained OneHotEncoder
    lb : trained LabelBinarizer
    label : str
        Name of the target column
    """
    # Process the full dataset
    X, y, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Run inference
    preds = inference(model, X)

    # Compute metrics
    precision, recall, f1 = compute_model_metrics(y, preds)

    print("\nOverall Performance:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
