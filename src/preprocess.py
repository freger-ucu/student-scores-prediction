"""
This module contains functions to preprocess student performance data.
"""
import pandas as pd
import numpy as np


def load_data(zip_path: str) -> pd.DataFrame:
    """
    Load data from a zip file
    :param zip_path: str, path to a zip file
    :return: pd.DataFrame
    """
    return pd.read_csv(zip_path, compression='zip')


def cap_scores(df: pd.DataFrame, max_score: int = 100) -> pd.DataFrame:
    """
    Caps exam scores in the DataFrame to a maximum value.
    :param df: pd.DataFrame, DataFrame containing exam scores
    :param max_score: int, maximum allowed score (default is 100)
    :return: pd.DataFrame, DataFrame with capped scores
    """
    df = df.copy()
    df.loc[df["Exam_Score"] > max_score, "Exam_Score"] = max_score
    return df


def fill_null_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills null values, the mode in categorical columns and
    the mean in numerical columns.
    :param df: pd.DataFrame, DataFrame with null values
    :return: pd.DataFrame, DataFrame with null values filled
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())
    return df


def select_features(df: pd.DataFrame, cat_cols: list, num_cols: list) -> pd.DataFrame:
    """
    Selects features from the DataFrame, encodes categorical columns,
    and returns the feature matrix.
    :param df: pd.DataFrame, DataFrame with features
    :param cat_cols: list, list of categorical column names to encode
    :param num_cols: list, list of numerical column names
    :return: pd.DataFrame, feature matrix
    """
    df_encoded = pd.get_dummies(
        df,
        columns=cat_cols,
        drop_first=True
    )

    dummy_cols = [
        col for col in df_encoded.columns
        if any(col.startswith(f"{cat}_") for cat in cat_cols)
    ]

    feature_cols = num_cols + dummy_cols

    return df_encoded[feature_cols]


def select_target(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Selects the target variable from the DataFrame.
    :param df: pd.DataFrame, DataFrame with features and target
    :param target_col: str, name of the target column
    :return: pd.Series, target variable
    """
    return df[target_col]


def scale_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Scales features using Min-Max scaling.
    :param X: pd.DataFrame, feature matrix
    :return: pd.DataFrame, scaled feature matrix
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    num_cols = X.select_dtypes(include=['float64', 'int64']).columns

    X_scaled = X.copy()
    X_scaled[num_cols] = scaler.fit_transform(X_scaled[num_cols])

    return X_scaled


def preprocess_data(df,
                    cat_cols: list | tuple = (
                                        'Access_to_Resources',
                                        'Parental_Involvement',
                                        'Parental_Education_Level',
                                        'Peer_Influence',
                                        'Learning_Disabilities'
                    ),
                    num_cols: list | tuple = (
                                        'Attendance',
                                        'Hours_Studied'
                    ),
                    target_col: str = 'Exam_Score'
                    ) -> tuple:
    """
    Preprocesses the data by loading, capping scores, filling null values,
    selecting features and target, and scaling features.
    :param df: pd.Dataframe, initial data
    :param cat_cols: list, list of categorical column names to encode
    :param num_cols: list, list of numerical column names
    :param target_col: str, name of the target column
    :return: tuple, (X_scaled, y) where X_scaled is the scaled feature matrix and y is the target variable
    """
    df = cap_scores(df)
    df = fill_null_values(df)

    X = select_features(df, list(cat_cols), list(num_cols))
    y = select_target(df, target_col)

    X_scaled = scale_features(X)

    return X_scaled.astype(np.float64), y
