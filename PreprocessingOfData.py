import pandas as pd
import re
import ast
from pandas import json_normalize

def convert_presence_list_to_dict(lst):
    return {item.strip(): 1 for item in lst if isinstance(item, str)}

def safe_parse(x):
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return x

def safe_parse_list(x):
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    if isinstance(x, list):
        return x
    return []

def parse_features_with_count(feature_list):
    feature_dict = {}
    for item in feature_list:
        if isinstance(item, str):
            if item.startswith('No '):
                name = item.replace('No ', '').strip()
                feature_dict[name] = 0
            else:
                match = re.match(r'(\d+)\s+(.*)', item)
                if match:
                    count = int(match.group(1))
                    name = match.group(2).strip()
                    feature_dict[name] = count
    return feature_dict

def parse_rating_list(rating_list):
    rating_dict = {}
    for item in rating_list:
        if isinstance(item, str):
            match = re.match(r'([A-Za-z\s]+)(\d+)\s*out of\s*5', item)
            if match:
                category = match.group(1).strip()
                rating = int(match.group(2))
                rating_dict[category] = rating
    return rating_dict

d = pd.read_csv("E:\placement\HouseRegression\houses.csv")
d['price'] = d['price'].str.extract(r'(\d+(?:\.\d+)?)').astype(float)
y = d['price']
d.drop(['link', 'property_id', 'description', 'nearbyLocations', 'property_name'], axis=1, inplace=True)
d['bedRoom'] = d['bedRoom'].str.extract(r'(\d+)').astype(float)
d['rate'] = d['rate'].str.extract(r'(\d+)').astype(float)
d['area'] = d['area'].str.extract(r'(\d+)').astype(float)
d['bathroom'] = d['bathroom'].str.extract(r'(\d+)').astype(float)
d['balcony'] = d['balcony'].str.extract(r'(\d+)').astype(float)
d['noOfFloor'] = d['noOfFloor'].str.extract(r'(\d+)').astype(float)
d[['area_number', 'area_sq_m']] = d['areaWithType'].str.extract(r'(\d+)[^\d]+(\d+\.\d+)?')
d['area_number'] = d['area_number'].astype(float)
d['area_sq_m'] = d['area_sq_m'].astype(float)
d.drop('areaWithType', axis=1, inplace=True)
d = pd.get_dummies(d, columns=['additionalRoom', 'agePossession', 'facing', 'society', 'address'])
d['furnishDetails'] = d['furnishDetails'].apply(safe_parse)
parsed_features = d['furnishDetails'].apply(parse_features_with_count)
features_df = pd.json_normalize(parsed_features).fillna(0).astype(int)
d = pd.concat([d.drop(columns=['furnishDetails']), features_df], axis=1)
d['rating'] = d['rating'].apply(safe_parse_list)
parsed_ratings = d['rating'].apply(parse_rating_list)
ratings_df = pd.json_normalize(parsed_ratings).fillna(0).astype(int)
d = pd.concat([d.drop(columns=['rating']), ratings_df], axis=1)
d['features'] = d['features'].apply(safe_parse)
parsed_amenities = d['features'].apply(convert_presence_list_to_dict)
amenities_df = pd.json_normalize(parsed_amenities).fillna(0).astype(int)
df = pd.concat([d.drop(columns=['features']), amenities_df], axis=1)
df['bedRoom'] = df['bedRoom'].fillna(0)
df['bathroom'] = df['bathroom'].fillna(0)
df['balcony'] = df['balcony'].fillna(0)
df['noOfFloor'] = df['noOfFloor'].fillna(0)
df['area_number']=df['area_number'].fillna(df['area_number'].median())
df['area_sq_m'] = df['area_sq_m'].fillna(df['area_sq_m'].median())
df['rate'] = df['rate'].fillna(df['rate'].median())
df['price'] = df['price'].fillna(df['price'].median())
df = df.drop(columns=['bathroom', 'bedRoom', 'Natural Light', 'Airy Rooms'])
df.to_csv("E:/placement/HouseRegression/processed_houses.csv", index=False)