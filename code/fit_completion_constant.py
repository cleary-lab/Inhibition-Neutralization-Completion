import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
mpl.style.use('ggplot')
import seaborn as sns
import argparse
import glob,os
from analyze_individual_trials import get_results, interpolate_values
from parse_datasets import load_flat_data, get_value_matrix, load_fonville_table
from matrix_completion_analysis import get_mask, get_specific_mask, complete_matrix

def transform_raw(x):
	return x

def transform_neglog10(x):
	return -np.log10(x)

def transform_log10(x):
	return np.log10(x)

def find_inflection(df,slope_thresh=-0.2,statistic='rmse'):
	available = np.invert(np.isnan(X.values))
	available_fraction = np.average(available)
	df_new = interpolate_values(df,available_fraction)
	max_frac = df_new.loc[(df_new['extrapolated'] == 'available data')]['observed fraction'].max()
	df_extrapolated = df_new.loc[(df_new['statistic'] == statistic) & (df_new['extrapolated'] == 'extrapolated') & (df_new['observed fraction'] <= max_frac)].sort_values('observed fraction')
	grad = np.gradient(df_extrapolated['value'], df_extrapolated['observed fraction'])
	if grad.max() > slope_thresh:
		return df_extrapolated['observed fraction'].values[(grad > slope_thresh)][0]
	else:
		return False

def get_rank(X,coh_thresh=0.5,rank_thresh=0.95,eps=0.3):
	x = X.values
	x = x[np.invert(np.isnan(x))]
	# rank_thresh = 1 - np.linalg.norm(np.random.randn(x.shape[0])*eps)**2/np.linalg.norm(x)**2
	# print(rank_thresh)
	mask = get_mask(X,1)
	X_hat = complete_matrix(X,mask,offset=True)
	u,s,vt = np.linalg.svd(X_hat - X_hat.mean())
	r = np.where(np.cumsum(s**2) > (s**2).sum()*rank_thresh)[0][0]
	idx_u = np.where(mask.mean(1) > coh_thresh)[0]
	idx_vt = np.where(mask.mean(0) > coh_thresh)[0]
	Xsub = X_hat[:,idx_vt]
	u,s,vt = np.linalg.svd(Xsub - Xsub.mean())
	q = min(r, np.where(np.cumsum(s**2) > (s**2).sum()*rank_thresh)[0][0])
	coh_u = np.linalg.norm(u[:,:q].dot(u[:,:q].T),axis=1).max()**2*X.shape[0]/r
	Xsub = X_hat[idx_u].T
	u,s,vt = np.linalg.svd(Xsub - Xsub.mean())
	q = min(r, np.where(np.cumsum(s**2) > (s**2).sum()*rank_thresh)[0][0])
	coh_v = np.linalg.norm(u[:,:q].dot(u[:,:q].T),axis=1).max()**2*X.shape[1]/r
	# coh_u = np.linalg.norm(u[:r].dot(u[:r].T),axis=1).max()**2*X.shape[0]/r
	# coh_v = np.linalg.norm(vt.T[:r].dot(vt.T[:r].T),axis=1).max()**2*X.shape[1]/r
	coh = max(coh_u,coh_v)
	return r, coh

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--resultspath', help='Path to individual trial results')
	parser.add_argument('--dataset', help='Path to dataset csv, comma separated if multiple')
	parser.add_argument('--savepath', help='Path to save results')
	parser.add_argument('--antibody-col-name', help='Column name for antibodies', default='Antibody')
	parser.add_argument('--value-name', help='Column name for values (IC50 or titer)', default='IC50 (ug/mL)')
	parser.add_argument('--data-transform', help='Type of transform to apply to raw data',default='neglog10', choices=('raw','neglog10','log10'))
	parser.add_argument('--antibody-col-name-list', help='List of column names for antibodies')
	parser.add_argument('--value-name-list', help='List of column names for values (IC50 or titer)')
	parser.add_argument('--data-transform-list', help='List of type of transforms to apply to raw data')
	parser.add_argument('--concat-option', help='Choose the 1st (eg PRE), 2nd (eg POST), or concatenate matrices',default='concat', choices=('concat','pre','post'))
	parser.add_argument('--min-titer-value', help='Replace eg <10 with min value', default=5,type=float)
	parser.add_argument('--max-titer-value', help='Replace eg >=1280 with max value', default=2560,type=float)
	parser.add_argument('--flat-file',dest='flat_file',help='Load dataset as flat file', action='store_true')
	parser.set_defaults(flat_file=False)
	args,_ = parser.parse_known_args()
	for key,value in vars(args).items():
		print('%s\t%s' % (key,str(value)))
	transform = {'raw': transform_raw, 'neglog10': transform_neglog10, 'log10': transform_log10}
	args.dataset = args.dataset.split(',')
	args.antibody_col_name_list = [args.antibody_col_name] if args.antibody_col_name_list is None else args.antibody_col_name_list.split(',')
	args.value_name_list = [args.value_name] if args.value_name_list is None else args.value_name_list.split(',')
	args.data_transform_list = [args.data_transform] if args.data_transform_list is None else args.data_transform_list.split(',')
	f = open(args.savepath,'w')
	_=f.write('\t'.join(['Dataset','Rank', 'Coherence','Inflection (observed fraction)', 'Inferred constant']) + '\n')
	for i,dataset in enumerate(args.dataset):
		dataset_prefix = dataset.split('/')[-1]
		dataset_prefix = dataset_prefix[:dataset_prefix.rfind('.')]
		if args.flat_file:
			flat_data = load_flat_data(dataset, vals=args.value_name_list[i])
			X = get_value_matrix(flat_data, rows=args.antibody_col_name_list[i], vals=args.value_name_list[i])
			X = transform[args.data_transform_list[i]](X)
		else:
			X = load_fonville_table(dataset, min_titer_value=args.min_titer_value, max_titer_value=args.max_titer_value)
			if isinstance(X,tuple):
				if args.concat_option == 'concat':
					X = pd.concat(X,keys=['PRE','POST'])
					dataset_prefix += '_concatenated'
				elif args.concat_option == 'pre':
					X = X[0]
					dataset_prefix += '_pre'
				elif args.concat_option == 'post':
					X = X[1]
					dataset_prefix += '_post'
			X.columns = [x.upper().replace('_','/') for x in X.columns]
			X = transform[args.data_transform](X)
		df_results = get_results(X, args.resultspath, dataset_prefix)
		inflection = find_inflection(df_results,slope_thresh=-0.2)
		rank, coh = get_rank(X)
		if inflection:
			completion_constant = inflection/(coh*max(X.shape)*rank*np.log(max(X.shape))/np.product(X.shape))
			_=f.write('%s\t%d\t%.4f\t%.2f\t%.4f\n' % (dataset_prefix, rank, coh, inflection, completion_constant))
		else:
			_=f.write('%s\t%d\t%.4f\tNaN\tNaN\n' % (dataset_prefix, rank, coh))
		print(dataset,rank,coh,inflection)
	f.close()
		