import pandas


_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]
df=pandas.read_csv('/home/zxk/AI/data/census_data/adult.data',delimiter=',',names=_CSV_COLUMNS)
print(df['race'].unique())