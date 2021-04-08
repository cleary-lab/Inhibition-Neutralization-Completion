import numpy as np
from parse_datasets import load_flat_data, get_value_matrix, load_fonville_table
from matrix_completion_analysis import calc_unobserved_r2, calc_unobserved_rmse
import pandas as pd
from scipy.spatial import distance
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
mpl.style.use('ggplot')
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib.ticker import FormatStrFormatter
import argparse
import glob,os

def get_results(X, resultspath, dataset_prefix, virus_list, table_list):
	RMSE = np.zeros((len(virus_list), len(table_list)))
	RMSE[:] = np.nan
	R2 = np.zeros((len(virus_list), len(table_list)))
	R2[:] = np.nan
	FP = glob.glob(os.path.join(resultspath, dataset_prefix, 'completed.*'))
	for fp in FP:
		if '.obs-1.0' not in fp:
			X_hat = np.load(fp)
			mask = np.load(fp.replace('completed','mask'))
			virus = fp.split('.')[-3].replace('-','/')
			table = fp.split('.')[-2]
			rmse = calc_unobserved_rmse(X, X_hat, mask)
			r2 = calc_unobserved_r2(X, X_hat, mask)
			i = np.where(virus_list == virus)[0][0]
			j = np.where(table_list == table)[0][0]
			RMSE[i,j] = rmse
			R2[i,j] = r2
	RMSE = pd.DataFrame(data=RMSE, index=virus_list, columns=table_list)
	R2 = pd.DataFrame(data=R2, index=virus_list, columns=table_list)
	return RMSE, R2

def plot_heatmap(virus_table, rmse, r2, savepath):
	fig,axes = plt.subplots(1, 3, sharey=True, figsize=(8,6))
	_=sns.heatmap(virus_table, ax=axes[0], cmap=sns.cm.rocket_r, cbar=False)
	_=axes[0].set_title('Table')
	_=sns.heatmap(rmse, ax=axes[1])
	_=axes[1].set_title('RMSE')
	_=sns.heatmap(r2, ax=axes[2])
	_=axes[2].set_title('R^2')
	_=plt.tight_layout()
	plt.savefig(savepath)
	plt.close()

def plot_violins(virus_table, values, value_label, savepath):
	counts = virus_table.values.sum(1)
	df = {value_label: [], 'Count': []}
	for i,j in zip(*np.where(virus_table.values)):
		df[value_label].append(values[i,j])
		df['Count'].append(counts[i])
	df = pd.DataFrame(df)
	_=sns.swarmplot(data=df, x='Count', y=value_label)
	_=plt.title('Performance on virus withheld in single table')
	_=plt.xlabel('Number of tables in which virus is present')
	plt.savefig(savepath)
	plt.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--resultspath', help='Path to individual trial results')
	parser.add_argument('--dataset', help='Path to dataset csv, comma separated if multiple')
	parser.add_argument('--savepath', help='Path to save results')
	parser.add_argument('--virus-table-path', help='Path to csv with viruses present in each Fonville dataset')
	parser.add_argument('--antibody-col-name', help='Column name for antibodies', default='Antibody')
	parser.add_argument('--value-name', help='Column name for values (IC50 or titer)', default='IC50 (ug/mL)')
	parser.add_argument('--data-transform', help='Type of transform to apply to raw data',default='neglog10', choices=('raw','neglog10','log10'))
	parser.add_argument('--high-titer-thresh', help='Threshold for high titer values (log10 units)',default=1,type=float)
	parser.add_argument('--example-obs-frac', help='Observed fraction (of available entries) in example plots (comma separated)',default='0.1,0.3,0.5')
	parser.add_argument('--concat-option', help='Choose the 1st (eg PRE), 2nd (eg POST), or concatenate matrices',default='concat', choices=('concat','pre','post'))
	parser.add_argument('--min-titer-value', help='Replace eg <10 with min value', default=5,type=float)
	parser.add_argument('--max-titer-value', help='Replace eg >=1280 with max value', default=2560,type=float)
	parser.add_argument('--flat-file',dest='flat_file',help='Load dataset as flat file', action='store_true')
	parser.set_defaults(flat_file=False)
	args,_ = parser.parse_known_args()
	for key,value in vars(args).items():
		print('%s\t%s' % (key,str(value)))
	dataset_prefix = args.dataset.split('/')[-1]
	dataset_prefix = dataset_prefix[:dataset_prefix.rfind('.')]
	if args.flat_file:
		flat_data = load_flat_data(args.dataset, vals=args.value_name)
		X = get_value_matrix(flat_data, rows=args.antibody_col_name, vals=args.value_name)
	else:
		update_savepath = False
		args.dataset = args.dataset.split(',')
		datasets = []
		for dataset in args.dataset:
			X = load_fonville_table(dataset, min_titer_value=args.min_titer_value, max_titer_value=args.max_titer_value)
			if isinstance(X,tuple):
				if args.concat_option == 'concat':
					X = pd.concat(X,keys=['PRE','POST'])
				elif args.concat_option == 'pre':
					X = X[0]
				elif args.concat_option == 'post':
					X = X[1]
				update_savepath = True
			X.columns = [x.upper().replace('_','/') for x in X.columns]
			datasets.append(X)
		X = datasets[0]
		for x in datasets[1:]:
			X = pd.concat([X,x])
		if update_savepath and (len(datasets) == 1):
			if args.concat_option == 'concat':
				args.savepath += '_concatenated'
			elif args.concat_option == 'pre':
				args.savepath += '_pre'
			elif args.concat_option == 'post':
				args.savepath += '_post'
		else:
			dataset_prefix = ''
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
	virus_table = pd.read_csv(args.virus_table_path,index_col=0)
	savepath_full = '%s/%s' % (args.savepath, dataset_prefix)
	rmse, r2 = get_results(X, args.resultspath, dataset_prefix, virus_table.index.values, virus_table.columns.values)
	xs = np.argsort(virus_table.values.sum(1))
	virus_table = virus_table.reindex(index = virus_table.index.values[xs])
	rmse = rmse.reindex(index = rmse.index.values[xs])
	r2 = r2.reindex(index = r2.index.values[xs])
	if not os.path.exists(savepath_full):
		os.makedirs(savepath_full)
	plot_heatmap(virus_table, rmse, r2, '%s/heatmaps.png' % args.savepath)
	plot_violins(virus_table, rmse.values, 'RMSE', '%s/rmse_by_count.png' % args.savepath)
	plot_violins(virus_table, r2.values, 'R^2', '%s/r2_by_count.png' % args.savepath)
	virus_table.to_csv('%s/virus_table.csv' % args.savepath)
	rmse.to_csv('%s/rmse.csv' % args.savepath)
	r2.to_csv('%s/r2.csv' % args.savepath)
