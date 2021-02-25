import numpy as np
from parse_datasets import load_flat_data, get_value_matrix, load_fonville_table
from matrix_completion_analysis import get_mask, complete_matrix, calc_unobserved_r2, calc_unobserved_rmse
import pandas as pd
import argparse
import os

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', help='Dataset (flatfile) to analyze')
	parser.add_argument('--job-id', help='Task array job id', default=None)
	parser.add_argument('--savepath', help='Path to save results')
	parser.add_argument('--antibody-col-name', help='Column name for antibodies', default='Antibody')
	parser.add_argument('--value-name', help='Column name for values (IC50 or titer)', default='IC50 (ug/mL)')
	parser.add_argument('--data-transform', help='Type of transform to apply to raw data',default='neglog10', choices=('raw','neglog10','log10'))
	parser.add_argument('--obs-frac', help='Observed fraction (of available entries)',default=0.3,type=float)
	parser.add_argument('--min-titer-value', help='Replace eg <10 with min value', default=5,type=float)
	parser.add_argument('--max-titer-value', help='Replace eg >=1280 with max value', default=2560,type=float)
	parser.add_argument('--concat-option', help='Choose the 1st (eg PRE), 2nd (eg POST), or concatenate matrices',default='concat', choices=('concat','pre','post'))
	parser.add_argument('--flat-file',dest='flat_file',help='Load dataset as flat file', action='store_true')
	parser.set_defaults(flat_file=False)
	args,_ = parser.parse_known_args()
	for key,value in vars(args).items():
		print('%s\t%s' % (key,str(value)))
	if args.flat_file:
		flat_data = load_flat_data(args.dataset, vals=args.value_name)
		X = get_value_matrix(flat_data, rows=args.antibody_col_name, vals=args.value_name)
	else:
		X = load_fonville_table(args.dataset, min_titer_value=args.min_titer_value, max_titer_value=args.max_titer_value)
		if isinstance(X,tuple):
			if args.savepath[-1] == '/':
				args.savepath = args.savepath[:-1]
			if args.concat_option == 'concat':
				X = pd.concat(X,keys=['PRE','POST'])
				args.savepath += '_concatenated'
			elif args.concat_option == 'pre':
				X = X[0]
				args.savepath += '_pre'
			elif args.concat_option == 'post':
				X = X[1]
				args.savepath += '_post'
	if args.data_transform == 'raw':
		def transform(x):
			return x
	elif args.data_transform == 'neglog10':
		def transform(x):
			return -np.log10(x)
	elif args.data_transform == 'log10':
		def transform(x):
			return np.log10(x)
	X = transform(X)
	mask = get_mask(X,args.obs_frac)
	X_hat = complete_matrix(X,mask,offset=True)
	rmse = calc_unobserved_rmse(X, X_hat, mask)
	r2 = calc_unobserved_r2(X, X_hat, mask)
	print('RMSE: %.4f' % rmse)
	print('r^2: %.4f' % r2)
	if not os.path.exists(args.savepath):
		os.makedirs(args.savepath)
	if args.job_id is not None:
		np.save('%s/completed.job-%s.obs-%.4f.npy' % (args.savepath, args.job_id, args.obs_frac), X_hat)
		np.save('%s/mask.job-%s.obs-%.4f.npy' % (args.savepath, args.job_id, args.obs_frac), mask)
	else:
		X_hat = pd.DataFrame(X_hat, index=X.index, columns=X.columns)
		X_hat.to_csv('%s/completed.obs-%.4f.%s.csv' % (args.savepath, args.obs_frac, args.data_transform))
		np.save('%s/mask.obs-%.4f.npy' % (args.savepath, args.obs_frac), mask)