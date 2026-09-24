import pandas as pd

csv_filepath = r'data\SEIP_data.csv'
csv_dataframe = pd.read_csv(csv_filepath)

print(csv_dataframe.head())
print(csv_dataframe.info())
column = 'm1_f_psf'
print(csv_dataframe[column].describe())