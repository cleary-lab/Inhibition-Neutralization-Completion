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
from scipy.optimize import curve_fit

def get_results(X, resultspath, dataset_prefix):
	df = {'value': [], 'observed fraction': [], 'statistic': [], 'X_hat': [], 'mask': []}
	FP = glob.glob(os.path.join(resultspath, dataset_prefix, 'completed.*'))
	for fp in FP:
		if '.npy' in fp:
			X_hat = np.load(fp)
		elif '.csv' in fp:
			X_hat = pd.read_csv(fp,index_col=0).values
		mask = np.load(fp.replace('completed','mask').replace('.csv','.npy'))
		p = fp.split('obs-')[-1].split('.npy')[0]
		while p.count('.') > 1:
			p = p[:p.rfind('.')]
		p = float(p)
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

def median_performer(df_results,obs_frac, index, columns):
	df_obs_frac = df_results[(df_results['observed fraction'] == obs_frac) & (df_results['statistic'] == 'rmse')].sort_values('value')
	median_performer_id = df_obs_frac.iloc[int(df_obs_frac.shape[0]/2)].name
	X_hat = df_results.loc[median_performer_id]['X_hat']
	mask = df_results.loc[median_performer_id]['mask']
	X_hat = pd.DataFrame(X_hat, index=index, columns=columns)
	return X_hat, mask

def fit_func(x, *coeffs):
	# multi-exponential model
	y = coeffs[0]
	for i in range(1,len(coeffs),2):
		y += coeffs[i]*np.exp(-coeffs[i+1]*x)
	return y

def fit_sensitivity_curve(x,y,x1,order=5):
	original_order = order
	while order > 0:
		try:
			popt, pcov = curve_fit(fit_func, x, y, p0=np.ones(order))
			break
		except:
			order -= 2
	if order >= 3:
		return fit_func(x1,*popt)
	else:
		# try fitting the inverse
		order = original_order
		while order > 0:
			try:
				popt, pcov = curve_fit(fit_func, x, 1/y, p0=np.ones(order))
				break
			except:
				order -= 2
		return 1/fit_func(x1,*popt)

def interpolate_values(df,available_fraction):
	df_new = df[['observed fraction','value','statistic']]
	df_new['observed fraction'] = df_new['observed fraction']*available_fraction
	df_new['extrapolated'] = 'available data'
	df_r2 = df_new.loc[df_new['statistic'] == 'r^2']
	x = df_r2['observed fraction']
	y = df_r2['value']
	x1 = np.linspace(x.min(),1,50)
	y1 = fit_sensitivity_curve(x,y,x1,order=9)
	df_r2 = pd.DataFrame({'observed fraction': x1, 'value': y1, 'statistic': ['r^2']*len(x1), 'extrapolated': ['extrapolated']*len(x1)})
	df_rmse = df_new.loc[df_new['statistic'] == 'rmse']
	x = df_rmse['observed fraction']
	y = df_rmse['value']
	x1 = np.linspace(x.min(),1,50)
	y1 = fit_sensitivity_curve(x,y,x1,order=5)
	df_rmse = pd.DataFrame({'observed fraction': x1, 'value': y1, 'statistic': ['rmse']*len(x1), 'extrapolated': ['extrapolated']*len(x1)})
	return pd.concat([df_new,df_r2,df_rmse])

def plot_results_curves(df,filename,available_fraction):
	df_new = interpolate_values(df,available_fraction)
	g = sns.FacetGrid(df_new, col="statistic", col_wrap=1, hue='statistic', sharey=False)
	g = (g.map_dataframe(sns.lineplot, "observed fraction", "value", style="extrapolated"))
	plt.savefig(filename)
	plt.close()

def plot_heatmap(X,X_hat,mask,filename,data_transform,value_name):
	available = np.invert(np.isnan(X.values))
	rmse = calc_unobserved_rmse(X, X_hat.values, mask)
	r2 = calc_unobserved_r2(X, X_hat.values, mask)
	u,s,vt = np.linalg.svd(X_hat - X_hat.values.mean())
	approx_rank = np.where(np.cumsum(s**2) > (s**2).sum()*0.95)[0][0] + 1
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
	try:
		df = df.rename_axis(['Unobserved','%s %s' % (data_transform,value_name)])
	except:
		pass
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

def plot_scatter(X,X_hat,mask,filename,data_transform,value_name):
	available = np.invert(np.isnan(X.values))
	rmse = calc_unobserved_rmse(X, X_hat.values, mask)
	r2 = calc_unobserved_r2(X, X_hat.values, mask)
	_=sns.regplot(x=X.values[np.where((1-mask)*available)], y = X_hat.values[np.where((1-mask)*available)], x_jitter=.1, scatter_kws={'alpha': 0.25})
	_=plt.xlabel('True titer (%s %s)' % (data_transform, value_name))
	_=plt.ylabel('Predicted titer (%s %s)' % (data_transform, value_name))
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

def plot_rank(X_hat,filename):
	u,s,vt = np.linalg.svd(X_hat - X_hat.values.mean())
	x = np.arange(len(s))+1
	y = np.cumsum(s**2)/(s**2).sum()
	ax=sns.lineplot(x=x, y=y)
	_=ax.axhline(0.95,ls='--',c='grey')
	_=ax.axvline(x[np.where(y > 0.95)[0][0]],ls='--',c='grey')
	_=plt.xlabel('sorted eigenvalues')
	_=plt.ylabel('cumulative variance')
	_=plt.tight_layout()
	plt.savefig(filename)
	plt.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--resultspath', help='Path to individual trial results')
	parser.add_argument('--dataset', help='Path to dataset csv, comma separated if multiple')
	parser.add_argument('--savepath', help='Path to save results')
	parser.add_argument('--antibody-col-name', help='Column name for antibodies', default='Antibody')
	parser.add_argument('--value-name', help='Column name for values (IC50 or titer)', default='IC50 (ug/mL)')
	parser.add_argument('--data-transform', help='Type of transform to apply to raw data',default='neglog10', choices=('raw','neglog10','log10'))
	parser.add_argument('--high-titer-thresh', help='Threshold for high titer values (log10 units)',default=1,type=float)
	parser.add_argument('--example-obs-frac', help='Observed fraction (of available entries) in example plots (comma separated)',default='0.1,0.3,0.5')
	parser.add_argument('--concat-option', help='Choose the 1st (eg PRE), 2nd (eg POST), or concatenate matrices',default='concat', choices=('concat','pre','post'))
	parser.add_argument('--min-titer-value', help='Replace eg <10 with min value', default=5,type=float)
	parser.add_argument('--max-titer-value', help='Replace eg >=1280 with max value', default=2560,type=float)
	parser.add_argument('--rmse-r2-curves',dest='rmse_r2_curves',help='Plot RMSE and r^2 vs fraction observed', action='store_true')
	parser.add_argument('--heatmap',dest='heatmap',help='Plot heatmap from an example of matrix completion', action='store_true')
	parser.add_argument('--scatter',dest='scatter',help='Scatter plot from an example of matrix completion', action='store_true')
	parser.add_argument('--recall-plot',dest='recall_plot',help='Plot recall curve', action='store_true')
	parser.add_argument('--rank-plot',dest='rank_plot',help='Plot rank curve', action='store_true')
	parser.add_argument('--flat-file',dest='flat_file',help='Load dataset as flat file', action='store_true')
	parser.set_defaults(flat_file=False)
	parser.set_defaults(rmse_r2_curves=False)
	parser.set_defaults(heatmap=False)
	parser.set_defaults(scatter=False)
	parser.set_defaults(recall_plot=False)
	parser.set_defaults(rank_plot=False)
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
			X.columns = [x.upper() for x in X.columns]
			datasets.append(X)
		X = datasets[0]
		for x in datasets[1:]:
			X = X.merge(x,how='outer')
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
	savepath_full = '%s/%s' % (args.savepath, dataset_prefix)
	df_results = get_results(X, args.resultspath, dataset_prefix)
	if not os.path.exists(savepath_full):
		os.makedirs(savepath_full)
	if args.rmse_r2_curves:
		available = np.invert(np.isnan(X.values))
		available_fraction = np.average(available)
		plot_results_curves(df_results,'%s/rmse_r2.png' % savepath_full, available_fraction)
	for o in args.example_obs_frac.split(','):
		obs_frac = float(o)
		X_hat, mask = median_performer(df_results, obs_frac, X.index, X.columns)
		if args.recall_plot:
			plot_recall(X, df_results, obs_frac,'%s/%d_pct_obs.recall.png' % (savepath_full,np.round(obs_frac*100)), args.high_titer_thresh)
		if args.heatmap:
			plot_heatmap(X, X_hat, mask,'%s/%d_pct_obs.heatmap.png' % (savepath_full,np.round(obs_frac*100)), args.data_transform, args.value_name)
		if args.scatter:
			plot_scatter(X, X_hat, mask,'%s/%d_pct_obs.scatter.png' % (savepath_full,np.round(obs_frac*100)), args.data_transform, args.value_name)
		if args.rank_plot:
			plot_rank(X_hat,'%s/%d_pct_obs.rank.png' % (savepath_full,np.round(obs_frac*100)))
