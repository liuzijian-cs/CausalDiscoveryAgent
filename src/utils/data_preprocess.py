# src/utils/data_preprocess.py
# 作者: LiuZijian(liuzj109@163.com & liuzijian-cs@shu.edu.cn)
# 时间: 2025-09-18 23:00
# 描述: 数据预处理工具 (Data Preprocessing Utilities)

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

def data_preprocess(df: pd.DataFrame):
    """预处理数据 (Preprocess data)
    column_type：dict[str, str]，每列名 → "Continuous" 或 "Category"
    overall_type：dict，形如 {"Data Type": "Continuous" | "Category" | "Mixture"}
    """
    # Data Type Classification
    column_type = {}
    overall_type = {}

    for column in df.columns:
        col_data = df[column]
        # Exclude NaN values for type determination
        non_nan_data = col_data.dropna()

        if pd.api.types.is_numeric_dtype(non_nan_data):
            is_effective_integer = np.all(np.floor(non_nan_data) == non_nan_data)
            # Check if numeric
            if is_effective_integer and non_nan_data.nunique() < 6:
                column_type[column] = "Category"
            else:
                column_type[column] = "Continuous"
        else:
            # Non-numeric data types
            column_type[column] = "Category"

    all_type = list(column_type.values())
    unique_type = list(set(all_type))

    
    if len(unique_type) == 1:
        if unique_type[0] == "Continuous":
            overall_type["Data Type"] = "Continuous"
        elif unique_type[0] == "Category":
            overall_type["Data Type"] = "Category"
    else:
        overall_type["Data Type"] = "Mixture"
    # Convert category data to numeric data
    categorical_features = [key for key, value in column_type.items() if value == "Category"]
    continuous_features = [key for key, value in column_type.items() if value == "Continuous"]

    for column in categorical_features:
        df[column] = pd.Categorical(df[column])
        df[column] = df[column].cat.codes.replace(-1, np.nan) # Keep NaN while converting    

    # imputation
    imputer_cat = SimpleImputer(strategy='most_frequent')
    imputer_cont = IterativeImputer(random_state=42)

    # Imputation for continuous data
    df[continuous_features] = imputer_cont.fit_transform(df[continuous_features])

    # Imputation for categorical data
    for column in categorical_features:
        df[column] = imputer_cat.fit_transform(df[[column]]).ravel()

    # Feature selection
    df = df.select_dtypes(include=['float64', 'int64'])

    # Z-score normalization
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return column_type, overall_type, scaled_df