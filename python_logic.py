import pandas as pd
import numpy as np
import joblib
import io
import traceback

# --- Feature engineering functions ---
def map_ordinal_features(df):
    """Ordinális jellemzők átalakítása numerikus értékekké."""
    df_mapped = df.copy()
    qual_cond_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0, np.nan: 0}
    
    ordinal_cols_map_dict = {
        'LotShape': {'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1, 'NA': 0, np.nan: 0},
        'Utilities': {'AllPub': 4, 'NoSewr': 3, 'NoSeWa': 2, 'ELO': 1, 'NA': 0, np.nan: 0},
        'LandSlope': {'Gtl': 3, 'Mod': 2, 'Sev': 1, 'NA': 0, np.nan: 0},
        'ExterQual': qual_cond_map, 'ExterCond': qual_cond_map,
        'BsmtQual': qual_cond_map, 'BsmtCond': qual_cond_map,
        'BsmtExposure': {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0, np.nan: 0},
        'BsmtFinType1': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0, np.nan: 0},
        'BsmtFinType2': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0, np.nan: 0},
        'HeatingQC': qual_cond_map,
        'KitchenQual': qual_cond_map,
        'FireplaceQu': qual_cond_map,
        'GarageFinish': {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0, np.nan: 0},
        'GarageQual': qual_cond_map, 'GarageCond': qual_cond_map,
        'PoolQC': qual_cond_map,
        'Fence': {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0, np.nan: 0}
    }
    
    for col, mapping in ordinal_cols_map_dict.items():
        if col in df_mapped.columns:
            df_mapped[col] = pd.to_numeric(df_mapped[col].map(mapping), errors='coerce').fillna(0)
            
    return df_mapped

def engineer_features(df, training_medians=None, training_quantiles=None):
    """Új jellemzők létrehozása és meglévők átalakítása."""
    df_engineered = df.copy()
    df_engineered = map_ordinal_features(df_engineered)

    # LotFrontage: Különös figyelemmel kell kezelni, ha hiányzik
    if 'LotFrontage' in df_engineered.columns:
        # Simple median imputation for this prediction case
        if pd.isna(df_engineered['LotFrontage']).any():
            median_lot_frontage = 60.0  # Default global median if missing
            df_engineered['LotFrontage'] = df_engineered['LotFrontage'].fillna(median_lot_frontage)

    # House Age calculation
    if 'YrSold' in df_engineered.columns and 'YearBuilt' in df_engineered.columns:
        df_engineered['House Age'] = df_engineered['YrSold'] - df_engineered['YearBuilt']
    
    # Remodeling features
    if 'YrSold' in df_engineered.columns and 'YearRemodAdd' in df_engineered.columns:
        df_engineered['Remod Age'] = df_engineered['YrSold'] - df_engineered['YearRemodAdd']
        if 'YearBuilt' in df_engineered.columns:
            df_engineered['Is Remodeled'] = (df_engineered['YearRemodAdd'] > df_engineered['YearBuilt']).astype(int)
    
    # New build feature
    if 'YearBuilt' in df_engineered.columns and 'YrSold' in df_engineered.columns:
        df_engineered['Is New Build'] = (df_engineered['YearBuilt'] == df_engineered['YrSold']).astype(int)

    # Fill NA for area columns
    area_cols_to_fill_na = ['GrLivArea', 'TotalBsmtSF', 'GarageArea', 'PoolArea', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MasVnrArea']
    for col in area_cols_to_fill_na:
        if col in df_engineered.columns:
            df_engineered[col] = df_engineered[col].fillna(0) 

    # Total square footage
    if 'GrLivArea' in df_engineered.columns and 'TotalBsmtSF' in df_engineered.columns:
        df_engineered['Total SF'] = df_engineered['GrLivArea'] + df_engineered['TotalBsmtSF']
    
    # Bathroom calculations
    numeric_cols_for_bathroom = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    if all(col in df_engineered.columns for col in numeric_cols_for_bathroom):
        for col in numeric_cols_for_bathroom: 
            df_engineered[col] = df_engineered[col].fillna(0)
        df_engineered['Total Bathroom'] = df_engineered['FullBath'] + 0.5 * df_engineered['HalfBath'] + \
                                        df_engineered['BsmtFullBath'] + 0.5 * df_engineered['BsmtHalfBath']

    # Garage features
    if 'GarageArea' in df_engineered.columns:
        df_engineered['Has Garage'] = df_engineered['GarageArea'].apply(lambda x: 0 if x == 0 else 1)

    if 'GarageYrBlt' in df_engineered.columns and 'YrSold' in df_engineered.columns:
        # Fill missing garage year built with house year built if available
        if 'YearBuilt' in df_engineered.columns:
            df_engineered['GarageYrBlt'] = df_engineered['GarageYrBlt'].fillna(df_engineered['YearBuilt'])
        # Calculate garage age
        df_engineered['Garage Age'] = df_engineered['YrSold'] - df_engineered['GarageYrBlt']
        df_engineered['Garage Age'] = df_engineered['Garage Age'].apply(lambda x: max(0, x if pd.notna(x) else 0))

    # Porch area
    porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    df_engineered['Total Porch SF'] = 0
    for col in porch_cols:
        if col in df_engineered.columns:
            df_engineered['Total Porch SF'] += df_engineered[col].fillna(0)

    # Pool feature
    if 'PoolArea' in df_engineered.columns:
        df_engineered['Has Pool'] = df_engineered['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    
    # Fireplace feature
    if 'Fireplaces' in df_engineered.columns:
        df_engineered['Fireplaces'] = df_engineered['Fireplaces'].fillna(0)
        df_engineered['Has Fireplace'] = df_engineered['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    # Quality interaction features
    quality_interaction_pairs = {
        'Overall Grade': ('OverallQual', 'OverallCond'),
        'Garage Grade': ('GarageQual', 'GarageCond'), 
        'Exter Grade': ('ExterQual', 'ExterCond')    
    }
    
    for new_col, (col1, col2) in quality_interaction_pairs.items():
        if col1 in df_engineered.columns and col2 in df_engineered.columns:
            df_engineered[col1] = pd.to_numeric(df_engineered[col1], errors='coerce').fillna(0)
            df_engineered[col2] = pd.to_numeric(df_engineered[col2], errors='coerce').fillna(0)
            df_engineered[new_col] = df_engineered[col1] * df_engineered[col2]

    # Age and quality interaction
    if 'House Age' in df_engineered.columns and 'OverallQual' in df_engineered.columns:
        df_engineered['Age*OverallQual'] = df_engineered['House Age'] * pd.to_numeric(df_engineered['OverallQual'], errors='coerce').fillna(0)
    
    # Total SF and quality interaction
    if 'Total SF' in df_engineered.columns and 'OverallQual' in df_engineered.columns:
        df_engineered['TotalSF*OverallQual'] = df_engineered['Total SF'] * pd.to_numeric(df_engineered['OverallQual'], errors='coerce').fillna(0)
    
    return df_engineered

# Define the input columns we expect from the form
FORM_COLUMNS = [
    'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 
    'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 
    'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 
    'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
    'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 
    'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 
    '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 
    'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 
    'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 
    'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 
    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 
    'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition'
]

# Create a complete mapping between form column names and model column names
COLUMN_MAPPING = {
    # Numeric columns
    'MSSubClass': 'MS SubClass', 
    'LotFrontage': 'Lot Frontage', 
    'LotArea': 'Lot Area', 
    'LotShape': 'Lot Shape', 
    'LandSlope': 'Land Slope', 
    'OverallQual': 'Overall Qual', 
    'OverallCond': 'Overall Cond', 
    'YearBuilt': 'Year Built', 
    'YearRemodAdd': 'Year Remod/Add', 
    'MasVnrArea': 'Mas Vnr Area', 
    'ExterQual': 'Exter Qual', 
    'ExterCond': 'Exter Cond', 
    'BsmtQual': 'Bsmt Qual', 
    'BsmtCond': 'Bsmt Cond', 
    'BsmtExposure': 'Bsmt Exposure', 
    'BsmtFinType1': 'BsmtFin Type 1', 
    'BsmtFinSF1': 'BsmtFin SF 1', 
    'BsmtFinType2': 'BsmtFin Type 2', 
    'BsmtFinSF2': 'BsmtFin SF 2', 
    'BsmtUnfSF': 'Bsmt Unf SF', 
    'TotalBsmtSF': 'Total Bsmt SF', 
    'HeatingQC': 'Heating QC', 
    '1stFlrSF': '1st Flr SF', 
    '2ndFlrSF': '2nd Flr SF', 
    'LowQualFinSF': 'Low Qual Fin SF', 
    'GrLivArea': 'Gr Liv Area', 
    'BsmtFullBath': 'Bsmt Full Bath', 
    'BsmtHalfBath': 'Bsmt Half Bath', 
    'FullBath': 'Full Bath', 
    'HalfBath': 'Half Bath', 
    'BedroomAbvGr': 'Bedroom AbvGr', 
    'KitchenAbvGr': 'Kitchen AbvGr', 
    'KitchenQual': 'Kitchen Qual', 
    'TotRmsAbvGrd': 'TotRms AbvGrd', 
    'FireplaceQu': 'Fireplace Qu', 
    'GarageYrBlt': 'Garage Yr Blt', 
    'GarageFinish': 'Garage Finish', 
    'GarageCars': 'Garage Cars', 
    'GarageArea': 'Garage Area', 
    'GarageQual': 'Garage Qual', 
    'GarageCond': 'Garage Cond', 
    'WoodDeckSF': 'Wood Deck SF', 
    'OpenPorchSF': 'Open Porch SF', 
    'EnclosedPorch': 'Enclosed Porch', 
    '3SsnPorch': '3Ssn Porch', 
    'ScreenPorch': 'Screen Porch', 
    'PoolArea': 'Pool Area', 
    'PoolQC': 'Pool QC', 
    'MiscVal': 'Misc Val', 
    'MoSold': 'Mo Sold', 
    'YrSold': 'Yr Sold',
    
    # Categorical columns
    'MSZoning': 'MS Zoning',
    'LandContour': 'Land Contour',
    'LotConfig': 'Lot Config',
    'Condition1': 'Condition 1',
    'Condition2': 'Condition 2',
    'BldgType': 'Bldg Type',
    'HouseStyle': 'House Style',
    'RoofStyle': 'Roof Style',
    'RoofMatl': 'Roof Matl',
    'Exterior1st': 'Exterior 1st',
    'Exterior2nd': 'Exterior 2nd',
    'MasVnrType': 'Mas Vnr Type',
    'CentralAir': 'Central Air',
    'GarageType': 'Garage Type',
    'PavedDrive': 'Paved Drive',
    'MiscFeature': 'Misc Feature',
    'SaleType': 'Sale Type',
    'SaleCondition': 'Sale Condition'
}

def get_prediction(input_data_dict, model_bytes):
    try:
        # Debug print the input data
        print("Input data dict:", input_data_dict)
        
        # Convert model_bytes to bytes if needed
        if hasattr(model_bytes, 'to_py'):
            model_bytes = model_bytes.to_py()
        
        # Create a DataFrame with one row
        df = pd.DataFrame({k: [v] for k, v in input_data_dict.items() if k in FORM_COLUMNS})
        
        # Add any missing columns that our model expects
        for col in FORM_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        
        # Apply feature engineering to create the additional columns
        df_engineered = engineer_features(df)
        
        print("Engineered DataFrame columns:", list(df_engineered.columns))
        
        # Load the model from bytes
        model = joblib.load(io.BytesIO(model_bytes))
        
        # Rename the columns to match what the model expects
        rename_dict = {k: v for k, v in COLUMN_MAPPING.items() if k in df_engineered.columns}
        df_engineered = df_engineered.rename(columns=rename_dict)
        
        print("After renaming, DataFrame columns:", list(df_engineered.columns))
        
        # Add dummy columns for categorical variables
        # This is critical if the model was trained with OneHotEncoder
        categorical_cols = [
            'MS Zoning', 'Street', 'Alley', 'Land Contour', 'Lot Config', 
            'Neighborhood', 'Condition 1', 'Condition 2', 'Bldg Type', 'House Style',
            'Roof Style', 'Roof Matl', 'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type',
            'Foundation', 'Heating', 'Central Air', 'Electrical', 'Functional',
            'Garage Type', 'Paved Drive', 'Fence', 'Misc Feature', 'Sale Type', 'Sale Condition'
        ]
        
        # Ensure categorical columns exist in the dataframe
        for col in categorical_cols:
            if col not in df_engineered.columns:
                df_engineered[col] = 'NA'  # Default value for missing categorical columns
        
        # Make the prediction
        prediction_log = model.predict(df_engineered)
        prediction = np.expm1(prediction_log[0])
        
        # Return a primitive float, not a numpy float
        return float(prediction)
    
    except Exception as e:
        print(f"Error in get_prediction: {str(e)}")
        print(traceback.format_exc())
        raise