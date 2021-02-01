import numpy as np
from parse_datasets import load_flat_data, get_value_matrix
from matrix_completion_analysis import get_mask, complete_matrix, calc_unobserved_r2, calc_unobserved_rmse
import pandas as pd
import argparse
import os

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', help='Dataset (flatfile) to analyze')
	parser.add_argument('--job-id', help='Task array job id')
	parser.add_argument('--savepath', help='Path to save results')
	parser.add_argument('--antibody-col-name', help='Column name for antibodies', default='Antibody')
	parser.add_argument('--value-name', help='Column name for values (IC50 or titer)', default='IC50 (ug/mL)')
	parser.add_argument('--data-transform', help='Type of transform to apply to raw data',default='neglog10', choices=('raw','neglog10','log10'))
	parser.add_argument('--obs-frac', help='Observed fraction (of available entries)',default=0.3,type=float)
	parser.add_argument('--min-titer-value', help='Replace eg <10 with min value', default=5,type=float)
	parser.add_argument('--flat-file',dest='flat_file',help='Load dataset as flat file', action='store_true')
	parser.set_defaults(flat_file=False)
	args,_ = parser.parse_known_args()
	for key,value in vars(args).items():
		print('%s\t%s' % (key,str(value)))
	if args.flat_file:
		flat_data = load_flat_data(args.dataset, vals=args.value_name)
		X = get_value_matrix(flat_data, rows=args.antibody_col_name, vals=args.value_name)
	else:
		X = pd.read_csv(args.dataset,index_col=0,na_values=['*']).replace('<10',args.min_titer_value).astype(float)
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
	np.save('%s/completed.job-%s.obs-%.4f.npy' % (args.savepath, args.job_id, args.obs_frac), X_hat)
	np.save('%s/mask.job-%s.obs-%.4f.npy' % (args.savepath, args.job_id, args.obs_frac), mask)