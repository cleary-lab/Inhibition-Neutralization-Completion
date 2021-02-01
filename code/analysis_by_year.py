import numpy as np
from parse_datasets import load_flat_data, get_value_matrix, get_common_subset
from matrix_completion_analysis import get_mask, complete_matrix, calc_unobserved_r2, calc_unobserved_rmse, calc_observed_rmse, calc_observed_r2
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

def plot_recall(X,X_hat,mask,thresh,savepath,max_year,min_year,obs_frac=0):
	x_label = 'Total fraction of withheld entries (%s and later) observed' % min_year
	y_label = 'Total fraction of high titers > %.2f identified' % (thresh)
	available = np.invert(np.isnan(X.values))
	df = {x_label: [], y_label: []}
	x = X.values[np.where((1-mask)*available)].flatten()
	x_hat = X_hat.values[np.where((1-mask)*available)].flatten()
	xs = np.argsort(-x_hat)
	x_hat_frac = np.arange(1,len(x)+1)/len(x)*(1-obs_frac) + obs_frac
	x_frac = np.cumsum((x[xs] >= thresh)/(x >= thresh).sum())
	df[x_label] = x_hat_frac
	df[y_label] = x_frac
	df = pd.DataFrame(df)
	ax=sns.lineplot(data=df,x=x_label, y=y_label, ci=None)
	_=ax.axhline(0.8,ls='--',c='grey')
	_=ax.axhline(0.9,ls='--',c='grey')
	_=ax.axhline(0.95,ls='--',c='grey')
	_=ax.axvline(x_hat_frac[np.where(x_frac > 0.8)[0][0]],ls='--',c='grey')
	_=ax.axvline(x_hat_frac[np.where(x_frac > 0.9)[0][0]],ls='--',c='grey')
	_=ax.axvline(x_hat_frac[np.where(x_frac > 0.95)[0][0]],ls='--',c='grey')
	_=ax.set_yticks([x_frac.min(),0.8,0.9,0.95,1])
	_=ax.set_xticks([obs_frac, x_hat_frac[np.where(x_frac > 0.8)[0][0]], x_hat_frac[np.where(x_frac > 0.9)[0][0]], x_hat_frac[np.where(x_frac > 0.95)[0][0]],1])
	_=ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	_=ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	_=plt.title('Recall of high titer entries from matrix completion (%s and earlier; ~%.1f%% unobserved)' % (max_year, 100-100*np.average(mask)), fontsize=8)
	_=plt.tight_layout()
	plt.savefig(savepath)
	plt.close()

def plot_scatter(X,X_hat,mask,savepath,rmse,r2,transform_units,max_year,min_year):
	available = np.invert(np.isnan(X.values))
	_=sns.regplot(x=X.values[np.where((1-mask)*available)], y = X_hat.values[np.where((1-mask)*available)], x_jitter=.1, scatter_kws={'alpha': 0.35})
	_=plt.xlabel('True titer (%s) in entries from %s and later' % (transform_units, min_year))
	_=plt.ylabel('Predicted titer (%s) from matrix completion of %s and earlier data' % (transform_units, max_year), fontsize=8)
	_=plt.title('Accuracy of matrix completion (%.1f%% of entries unobserved; RMSE=%.3f; r^2=%.1f%%)' % (100-100*np.average(mask), rmse, 100*r2), fontsize=8)
	plt.savefig(savepath)
	plt.close()

def plot_heatmap(X,X_hat,mask,savepath,rmse,r2):
	available = np.invert(np.isnan(X.values))
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
	_=plt.title('%.1f%% of available entries unobserved; RMSE=%.3f; r^2=%.1f%%' % (100 - np.average(mask)*100, rmse, r2*100))
	ax=sns.heatmap(df,mask=df_mask,ax=ax,cmap=sns.cm.rocket_r)
	_=ax.axhline(X.shape[0],color='blue')
	_=plt.tight_layout()
	bottom, top = ax.get_ylim()
	ax.set_ylim(bottom + 0.5, top - 0.5) # sorry, this may cut off the bottom row...some sort of matplotlib bug
	plt.savefig(savepath)
	plt.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', help='Path to data (csv flatfile)')
	parser.add_argument('--savepath', help='Path to save results')
	parser.add_argument('--max-year', help='Most recent year in observed data (to be completed)')
	parser.add_argument('--min-year', help='Earliest year in predicted data')
	parser.add_argument('--data-transform', help='Type of transform to apply to raw data',default='neglog10', choices=('raw','neglog10','log10'))
	parser.add_argument('--high-titer-thresh', help='Threshold for high titer values (transformed units)',default=2,type=float)
	parser.add_argument('--save-data',dest='save_data',help='Save csvs for observed, unobserved and predicted', action='store_true')
	parser.add_argument('--heatmap',dest='heatmap',help='Plot heatmap from an example of matrix completion', action='store_true')
	parser.add_argument('--scatter',dest='scatter',help='Scatter plot from an example of matrix completion', action='store_true')
	parser.add_argument('--recall-plot',dest='recall_plot',help='Plot recall curve', action='store_true')
	parser.set_defaults(save_data=False)
	parser.set_defaults(heatmap=False)
	parser.set_defaults(scatter=False)
	parser.set_defaults(recall_plot=False)
	args,_ = parser.parse_known_args()
	for key,value in vars(args).items():
		print('%s\t%s' % (key,str(value)))
	flat_data = load_flat_data(args.dataset)
	X1 = get_value_matrix(flat_data, duplicate_resolution_mode='max_year', year=args.max_year)
	X2 = get_value_matrix(flat_data, duplicate_resolution_mode='min_year', year=args.min_year, min_count=1)
	if args.data_transform == 'raw':
		def transform(x):
			return x
	elif args.data_transform == 'neglog10':
		def transform(x):
			return -np.log10(x)
	elif args.data_transform == 'log10':
		def transform(x):
			return np.log10(x)
	X1 = transform(X1)
	X2 = transform(X2)
	print(X1.shape)
	X1, X2, mask = get_common_subset(X1,X2)
	X_hat = pd.DataFrame(complete_matrix(X1,mask,offset=True), index=X1.index, columns=X1.columns)
	rmse = calc_unobserved_rmse(X2,X_hat.values,mask)
	r2 = calc_unobserved_r2(X2,X_hat.values,mask)
	rmse_obs = calc_observed_rmse(X2,X1.values,mask)
	r2_obs = calc_observed_r2(X2,X1.values,mask)
	rmse_obs_hat = calc_observed_rmse(X2,X_hat.values,mask)
	r2_obs_hat = calc_observed_r2(X2,X_hat.values,mask)
	print(X1.shape)
	print(rmse,r2)
	print(rmse_obs,r2_obs)
	print(rmse_obs_hat,r2_obs_hat)
	if args.save_data:
		_=X1.to_csv(path_or_buf='%s/data.observed.observation_end_%s.prediction_start_%s.csv' % (args.savepath, args.max_year, args.min_year))
		_=X2.to_csv(path_or_buf='%s/data.unobserved.observation_end_%s.prediction_start_%s.csv' % (args.savepath, args.max_year, args.min_year))
		_=X_hat.to_csv(path_or_buf='%s/data.predicted.observation_end_%s.prediction_start_%s.csv' % (args.savepath, args.max_year, args.min_year))
	if args.recall_plot:
		plot_recall(X2,X_hat,mask,args.high_titer_thresh,'%s/recall.observation_end_%s.prediction_start_%s.png' % (args.savepath, args.max_year, args.min_year), args.max_year, args.min_year)
	if args.scatter:
		plot_scatter(X2,X_hat,mask,'%s/scatter.observation_end_%s.prediction_start_%s.png' % (args.savepath, args.max_year, args.min_year), rmse, r2, args.data_transform, args.max_year, args.min_year)
	if args.heatmap:
		plot_heatmap(X2,X_hat,mask,'%s/heatmap.observation_end_%s.prediction_start_%s.png' % (args.savepath, args.max_year, args.min_year),rmse,r2)


