import numpy as np
from parse_datasets import load_flat_data, get_value_matrix
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

def get_results(resultspath, dataset_prefix):
	df = {'value': [], 'observed fraction': [], 'statistic': [], 'X_hat': [], 'mask': []}
	FP = glob.glob(os.path.join(resultspath, dataset_prefix, 'completed.*.npy'))
	for fp in FP:
		X_hat = np.load(fp)
		mask = np.load(fp.replace('completed','mask'))
		p = float(fp.split('obs-')[-1].split('.npy')[0])
		rmse = calc_unobserved_rmse(X, X_hat, mask)
		r2 = calc_unobserved_r2(X, X_hat, mask)
		df['observed fraction'].append(p)
		df['statistic'].append('r^2')
		df['value'].append(r2)
		df['X_hat'].append(X_hat)
		df['mask'].append(mask)
		df['observed fraction'].append(p)
		df['statistic'].append('rmse')
		df['value'].append(rmse)
		df['X_hat'].append(X_hat)
		df['mask'].append(mask)
	return pd.DataFrame(df)

def median_performer(df_results,obs_frac):
	df_obs_frac = df_results[(df_results['observed fraction'] == obs_frac) & (df_results['statistic'] == 'rmse')].sort_values('value')
	median_performer_id = df_obs_frac.iloc[int(df_obs_frac.shape[0]/2)].name
	X_hat = df_results.loc[median_performer_id]['X_hat']
	mask = df_results.loc[median_performer_id]['mask']
	X_hat = pd.DataFrame(X_hat, index=X.index, columns=X.columns)
	return X_hat, mask

def plot_results_curves(df,filename):
	g = sns.FacetGrid(df, col="statistic", col_wrap=1, hue='statistic', sharey=False)
	g = (g.map_dataframe(sns.lineplot, "observed fraction", "value"))
	plt.savefig(filename)
	plt.close()

def plot_heatmap(X,X_hat,mask,filename):
	available = np.invert(np.isnan(X.values))
	rmse = calc_unobserved_rmse(X, X_hat.values, mask)
	r2 = calc_unobserved_r2(X, X_hat.values, mask)
	u,s,vt = np.linalg.svd(X_hat - X_hat.values.mean())
	approx_rank = (s > s.max()/100).sum()
	correlations = np.asarray(X.corr())
	correlations[np.isnan(correlations)] = 0
	col_linkage = linkage(distance.pdist(correlations), method='average')
	col_order = leaves_list(col_linkage)
	correlations = np.asarray(X.T.corr())
	correlations[np.isnan(correlations)] = 0
	row_linkage = linkage(distance.pdist(correlations), method='average')
	row_order = leaves_list(row_linkage)
	X_reorder = X.reindex(X.index[row_order])[X.columns[col_order]]
	Xhat_reorder = X_hat.reindex(X.index[row_order])[X.columns[col_order]]
	df = pd.concat([X_reorder,Xhat_reorder],keys=['original','inferred'])
	df = df.rename_axis(['Unobserved','Neutralization[Titers]'])
	df_mask = np.vstack([mask,mask])
	fig, ax = plt.subplots(figsize=(8,12))
	_=plt.title('%.1f%% of entries available; %.1f%% observed; RMSE=%.3f; r^2=%.1f%%\nsize: %d x %d; approx. rank: %d' % (np.average(available)*100, np.average(mask)*100, rmse, r2*100, X.shape[0], X.shape[1], approx_rank), fontsize=10)
	ax=sns.heatmap(df,mask=df_mask,ax=ax,cmap=sns.cm.rocket_r)
	_=ax.axhline(X.shape[0],color='blue')
	_=plt.tight_layout()
	bottom, top = ax.get_ylim()
	ax.set_ylim(bottom + 0.5, top - 0.5) # sorry, this may cut off the bottom row...some sort of matplotlib bug
	plt.savefig(filename)
	plt.close()

def plot_scatter(X,X_hat,mask,filename):
	available = np.invert(np.isnan(X.values))
	rmse = calc_unobserved_rmse(X, X_hat.values, mask)
	r2 = calc_unobserved_r2(X, X_hat.values, mask)
	_=sns.regplot(x=X.values[np.where((1-mask)*available)], y = X_hat.values[np.where((1-mask)*available)], x_jitter=.1, scatter_kws={'alpha': 0.35})
	_=plt.xlabel('True titer (log10)')
	_=plt.ylabel('Predicted titer (log10)')
	_=plt.title('%.1f%% of entries available; %.1f%% observed; RMSE=%.3f; r^2=%.1f%%' % (100*np.average(available), 100*np.average(mask), rmse, 100*r2), fontsize=8)
	plt.savefig(filename)
	plt.close()

def plot_recall(X,df_results,obs_frac,filename,thresh,n=25):
	available = np.invert(np.isnan(X.values))
	df = {'Total fraction of available entries observed (random sample + validation)': [], 'Total fraction of high titers > %.2f identified' % (thresh): []}
	df_obs_frac = df_results[(df_results['observed fraction'] == obs_frac) & (df_results['statistic'] == 'rmse')]
	for index,row in df_obs_frac.iterrows():
		mask = row['mask']
		X_hat = pd.DataFrame(row['X_hat'], index=X.index, columns=X.columns)
		x = X.values[np.where((1-mask)*available)].flatten()
		x_hat = X_hat.values[np.where((1-mask)*available)].flatten()
		xs = np.argsort(-x_hat)
		x_hat_frac = np.arange(1,len(x)+1)/len(x)*(1-obs_frac) + obs_frac
		x_frac = np.cumsum((x[xs] >= thresh)/(X.values >= thresh).sum()) + (X.values[np.where(mask)] >= thresh).sum()/(X.values >= thresh).sum()
		df['Total fraction of available entries observed (random sample + validation)'].append(x_hat_frac)
		df['Total fraction of high titers > %.2f identified' % (thresh)].append(x_frac)
	x_frac_avg = np.average(df['Total fraction of high titers > %.2f identified' % (thresh)],axis=0)
	df['Total fraction of available entries observed (random sample + validation)'] = np.hstack(df['Total fraction of available entries observed (random sample + validation)'])
	df['Total fraction of high titers > %.2f identified' % (thresh)] = np.hstack(df['Total fraction of high titers > %.2f identified' % (thresh)])
	df = pd.DataFrame(df)
	ax=sns.lineplot(data=df,x='Total fraction of available entries observed (random sample + validation)', y='Total fraction of high titers > %.2f identified' % (thresh), ci='sd')
	_=ax.axhline(0.8,ls='--',c='grey')
	_=ax.axhline(0.9,ls='--',c='grey')
	_=ax.axhline(0.95,ls='--',c='grey')
	_=ax.axvline(x_hat_frac[np.where(x_frac_avg > 0.8)[0][0]],ls='--',c='grey')
	_=ax.axvline(x_hat_frac[np.where(x_frac_avg > 0.9)[0][0]],ls='--',c='grey')
	_=ax.axvline(x_hat_frac[np.where(x_frac_avg > 0.95)[0][0]],ls='--',c='grey')
	_=ax.set_yticks([x_frac_avg.min(),0.8,0.9,0.95,1])
	_=ax.set_xticks([obs_frac, x_hat_frac[np.where(x_frac_avg > 0.8)[0][0]], x_hat_frac[np.where(x_frac_avg > 0.9)[0][0]], x_hat_frac[np.where(x_frac_avg > 0.95)[0][0]],1])
	_=ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	_=ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	_=plt.title('Recall of high titer entries from matrix completion (%.1f%% available; %.1f%% observed) + validation' % (100*np.average(available), 100*np.average(mask)), fontsize=8)
	_=plt.tight_layout()
	plt.savefig(filename)
	plt.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--resultspath', help='Path to individual trial results')
	parser.add_argument('--dataset', help='Path to dataset csv')
	parser.add_argument('--savepath', help='Path to save results')
	parser.add_argument('--antibody-col-name', help='Column name for antibodies', default='Antibody')
	parser.add_argument('--value-name', help='Column name for values (IC50 or titer)', default='IC50 (ug/mL)')
	parser.add_argument('--data-transform', help='Type of transform to apply to raw data',default='neglog10', choices=('raw','neglog10','log10'))
	parser.add_argument('--high-titer-thresh', help='Threshold for high titer values (log10 units)',default=2,type=float)
	parser.add_argument('--example-obs-frac', help='Observed fraction (of available entries) in example plots',default=0.3,type=float)
	parser.add_argument('--rmse-r2-curves',dest='rmse_r2_curves',help='Plot RMSE and r^2 vs fraction observed', action='store_true')
	parser.add_argument('--heatmap',dest='heatmap',help='Plot heatmap from an example of matrix completion', action='store_true')
	parser.add_argument('--scatter',dest='scatter',help='Scatter plot from an example of matrix completion', action='store_true')
	parser.add_argument('--recall-plot',dest='recall_plot',help='Plot recall curve', action='store_true')
	parser.set_defaults(rmse_r2_curves=False)
	parser.set_defaults(heatmap=False)
	parser.set_defaults(scatter=False)
	parser.set_defaults(recall_plot=False)
	args,_ = parser.parse_known_args()
	for key,value in vars(args).items():
		print('%s\t%s' % (key,str(value)))
	flat_data = load_flat_data(args.dataset, vals=args.value_name)
	X = get_value_matrix(flat_data, rows=args.antibody_col_name, vals=args.value_name)
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
	dataset_prefix = args.dataset.split('/')[-1]
	dataset_prefix = dataset_prefix[:dataset_prefix.rfind('.')]
	savepath_full = '%s/%s' % (args.savepath, dataset_prefix)
	if not os.path.exists(savepath_full):
		os.makedirs(savepath_full)
	df_results = get_results(args.resultspath, dataset_prefix)
	X_hat, mask = median_performer(df_results, args.example_obs_frac)
	if args.rmse_r2_curves:
		plot_results_curves(df_results,'%s/rmse_r2.png' % savepath_full)
	if args.heatmap:
		plot_heatmap(X, X_hat, mask,'%s/heatmap.png' % savepath_full)
	if args.scatter:
		plot_scatter(X, X_hat, mask,'%s/scatter.png' % savepath_full)
	if args.recall_plot:
		plot_recall(X, df_results, args.example_obs_frac,'%s/recall.png' % savepath_full, args.high_titer_thresh)
