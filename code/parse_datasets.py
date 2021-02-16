import numpy as np
import pandas as pd
import re

def load_fonville_table(filename,min_titer_value=5,max_titer_value=2560,multiindex=False,index='Subject Number',level1_labels=['Unnamed: 0_level_1','PRE','POST'],add_index=None):
	# 1: df = load_table('Fonville2014_TableS1.csv',index='Sample')
	# 3: df = load_table('Fonville2014_TableS3.csv', add_index='Sample Year')
	# 13,14: df = load_table('Fonville2014_TableS13.csv',multiindex=True)
	# 5,6: df = load_table('Fonville2014_TableS5.csv',add_index='Age',multiindex=True,level1_labels=['Unnamed: 0_level_1','Unnamed: 1_level_1','PRE','POST'])
	if 'TableS1.csv' in filename:
		index = 'Sample'
	elif 'TableS3.csv' in filename:
		add_index = 'Sample Year'
	elif ('TableS13.csv' in filename) or ('TableS14.csv' in filename):
		multiindex = True
	elif ('TableS5.csv' in filename) or ('TableS6.csv' in filename):
		add_index = 'Age'
		multiindex = True
		level1_labels = ['Unnamed: 0_level_1','Unnamed: 1_level_1','PRE','POST']
	if multiindex:
		df = pd.read_csv(filename, header=[0,1],na_values=['*']).replace('<10',min_titer_value).replace('>=1280',max_titer_value)
		df_index = df.xs(level1_labels[0],level=1, drop_level=True, axis=1)
		if add_index is not None:
			df_index2 = df.xs(level1_labels[1],level=1, drop_level=True, axis=1)
		df_pre = df.xs(level1_labels[-2],level=1, drop_level=True, axis=1)
		df_post = df.xs(level1_labels[-1],level=1, drop_level=True, axis=1)
		df_pre[index] = df_index
		df_post[index] = df_index
		if add_index is not None:
			df_pre[add_index] = df_index2
			df_post[add_index] = df_index2
		df_pre = df_pre.set_index(index)
		df_post = df_post.set_index(index)
		if add_index is not None:
			df_pre.set_index(add_index,append=True,inplace=True)
			df_post.set_index(add_index,append=True,inplace=True)
		if not np.all(df_pre.columns == df_post.columns):
			df_post.columns = df_pre.columns
		return df_pre.astype(float), df_post.astype(float)
	else:
		df = pd.read_csv(filename,index_col=0,na_values=['*']).replace('<10',min_titer_value).replace('>=1280',max_titer_value)
		if add_index is not None:
			df.set_index(add_index,append=True,inplace=True)
		return df.astype(float)

def load_flat_data(filename,study='Study',vals='IC50 (ug/mL)',limit_factor=2):
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

def get_value_matrix(flat_data,rows='Antibody',cols='Virus',vals='IC50 (ug/mL)',duplicate_resolution_mode='geom_mean', min_count=5, min_year=None, max_year=None):
	if min_year is not None:
		flat_data = flat_data[flat_data.Year >= min_year]
	if max_year is not None:
		flat_data = flat_data[flat_data.Year <= max_year]
	if duplicate_resolution_mode == 'most_recent':
		df = flat_data.sort_values('Year', ascending=False).drop_duplicates(subset=[rows, cols])
	elif duplicate_resolution_mode == 'geom_mean':
		flat_data[vals] = np.log(flat_data[vals])
		df = flat_data.groupby([rows,cols]).mean().reset_index()
		df[vals] = np.exp(df[vals])	
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

