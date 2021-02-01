import numpy as np
import pandas as pd
import re

def load_flat_data(filename,study='Study',vals='IC50 (ug/mL)',limit_factor=1.5):
	x = pd.read_csv(filename)
	year = [re.findall(r'[0-9]+', s) for s in x[study]]
	year = [s[0] if len(s)>0 else '' for s in year]
	x['Year'] = year
	new_vals = []
	for v in x[vals]:
		if '>' in v:
			new_vals.append(float(v[1:])*limit_factor)
		elif '<' in v:
			new_vals.append(float(v[1:])/limit_factor)
		else:
			new_vals.append(float(v))
	x[vals] = new_vals
	return x

def get_value_matrix(flat_data,rows='Antibody',cols='Virus', vals='IC50 (ug/mL)', duplicate_resolution_mode='most_recent', min_count=5, year=None):
	if duplicate_resolution_mode == 'most_recent':
		df = flat_data.sort_values('Year', ascending=False).drop_duplicates(subset=[rows, cols])
	elif duplicate_resolution_mode == 'max_year':
		df = flat_data[flat_data.Year <= year].sort_values('Year', ascending=False).drop_duplicates(subset=[rows, cols])
	elif duplicate_resolution_mode == 'min_year':
		df = flat_data[flat_data.Year >= year].sort_values('Year', ascending=False).drop_duplicates(subset=[rows, cols])
	df = df.pivot(index=rows, columns=cols, values=vals).astype(np.float)
	include_rows = np.where(np.invert(np.isnan(df.values)).sum(1) > min_count)[0]
	df = df.loc[df.index[include_rows]]
	include_cols = np.where(np.invert(np.isnan(df.values)).sum(0) > min_count)[0]
	df = df[df.columns[include_cols]]
	return df

def get_common_subset(X1,X2,min_count=5):
	common_rows = [i for i in X1.index if i in X2.index]
	common_cols = [j for j in X1.columns if j in X2.columns]
	X1_common = X1.loc[common_rows][common_cols]
	X2_common = X2.loc[common_rows][common_cols]
	include_rows = np.where(np.invert(np.isnan(X1_common.values)).sum(1) > min_count)[0]
	X1_common = X1_common.loc[X1_common.index[include_rows]]
	X2_common = X2_common.loc[X2_common.index[include_rows]]
	include_cols = np.where(np.invert(np.isnan(X1_common.values)).sum(0) > min_count)[0]
	X1_common = X1_common[X1_common.columns[include_cols]]
	X2_common = X2_common[X2_common.columns[include_cols]]
	mask = np.invert(np.isnan(X1_common.values))
	return X1_common, X2_common, mask

