import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_and_clean_data(file_path):
    dataset = pd.read_csv(file_path)

    # Limpieza inicial
    threshold = 0.6
    dataset_cleaned = dataset.loc[:, dataset.isnull().mean() < threshold]
    dataset_cleaned = dataset_cleaned[(dataset_cleaned['price'] > 1000) & (dataset_cleaned['price'] < 100000)]
    dataset_cleaned = dataset_cleaned[(dataset_cleaned['odometer'] > 1000) & (dataset_cleaned['odometer'] < 300000)]
    dataset_cleaned = dataset_cleaned.drop_duplicates()

    return dataset_cleaned


def preprocess_data(dataset):
    features = dataset.drop(
        columns=['price', 'id', 'url', 'region_url', 'image_url', 'description', 'VIN', 'posting_date'])
    target = dataset['price']

    # Identificamos las columnas
    categorical_columns = features.select_dtypes(include=['object']).columns
    numerical_columns = features.select_dtypes(include=['float64', 'int64']).columns

    # Realizamos el preprocesamiento
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])
    X_preprocessed = preprocessor.fit_transform(features)
    return X_preprocessed, target
