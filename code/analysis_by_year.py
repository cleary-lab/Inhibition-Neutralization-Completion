import numpy as np
from parse_datasets import load_flat_data, get_value_matrix, get_common_subset
from matrix_completion_analysis import get_mask, complete_matrix, calc_unobserved_r2, calc_unobserved_rmse, calc_observed_rmse, calc_observed_r2
from analyze_individual_trials import plot_heatmap, plot_rank
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
import os

def plot_recall(X,X_hat,mask,thresh,savepath,max_year,min_year,obs_frac=0,as_absolute=False):
	if as_absolute:
		x_label = 'Total number of entries from %s and later observed' % min_year
		y_label = 'Total number of high titers > %.2f identified' % (thresh)
	else:
		x_label = 'Total fraction of entries from %s and later observed' % min_year
		y_label = 'Total fraction of high titers > %.2f identified' % (thresh)
	available = np.invert(np.isnan(X.values))
	df = {x_label: [], y_label: []}
	x = X.values[np.where((1-mask)*available)].flatten()
	x_hat = X_hat.values[np.where((1-mask)*available)].flatten()
	xs = np.argsort(-x_hat)
	total_high = (x >= thresh).sum()
	if as_absolute:
		x_hat_frac = np.arange(1,len(x)+1)
		x_frac = np.cumsum((x[xs] >= thresh))
		xtick_locs = (obs_frac*len(x),np.where(x_frac > 0.8*total_high)[0][0],np.where(x_frac > 0.9*total_high)[0][0],np.where(x_frac > 0.95*total_high)[0][0],len(x))
		ytick_locs = (np.rint(0.8*total_high),np.rint(0.9*total_high),np.rint(0.95*total_high),total_high)
	else:
		x_hat_frac = np.arange(1,len(x)+1)/len(x)*(1-obs_frac) + obs_frac
		x_frac = np.cumsum((x[xs] >= thresh)/total_high)
		xtick_locs = (obs_frac,np.where(x_frac > 0.8)[0][0],np.where(x_frac > 0.9)[0][0],np.where(x_frac > 0.95)[0][0],1)
		ytick_locs = (0.8,0.9,0.95,1)
	df[x_label] = x_hat_frac
	df[y_label] = x_frac
	df = pd.DataFrame(df)
	ax=sns.lineplot(data=df,x=x_label, y=y_label, ci=None)
	_=ax.axhline(ytick_locs[0],ls='--',c='grey')
	_=ax.axhline(ytick_locs[1],ls='--',c='grey')
	_=ax.axhline(ytick_locs[2],ls='--',c='grey')
	_=ax.axvline(x_hat_frac[xtick_locs[1]],ls='--',c='grey')
	_=ax.axvline(x_hat_frac[xtick_locs[2]],ls='--',c='grey')
	_=ax.axvline(x_hat_frac[xtick_locs[3]],ls='--',c='grey')
	_=ax.set_yticks([x_frac.min(),ytick_locs[0],ytick_locs[1],ytick_locs[2],ytick_locs[3]])
	_=ax.set_xticks([xtick_locs[0], x_hat_frac[xtick_locs[1]], x_hat_frac[xtick_locs[2]], x_hat_frac[xtick_locs[3]],xtick_locs[4]])
	if as_absolute:
		_=ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
		_=ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
	else:
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

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', help='Path to data (csv flatfile)')
	parser.add_argument('--savepath', help='Path to save results')
	parser.add_argument('--max-year', help='Most recent year in observed data (to be completed)')
	parser.add_argument('--min-year', help='Earliest year in predicted data')
	parser.add_argument('--data-transform', help='Type of transform to apply to raw data',default='neglog10', choices=('raw','neglog10','log10'))
	parser.add_argument('--high-titer-thresh', help='Threshold for high titer values (transformed units)',default=1,type=float)
	parser.add_argument('--value-name', help='Column name for values (IC50 or titer)', default='IC50 (ug/mL)')
	parser.add_argument('--save-data',dest='save_data',help='Save csvs for observed, unobserved and predicted', action='store_true')
	parser.add_argument('--heatmap',dest='heatmap',help='Plot heatmap from an example of matrix completion', action='store_true')
	parser.add_argument('--scatter',dest='scatter',help='Scatter plot from an example of matrix completion', action='store_true')
	parser.add_argument('--recall-plot',dest='recall_plot',help='Plot recall curve', action='store_true')
	parser.add_argument('--rank-plot',dest='rank_plot',help='Plot rank curve', action='store_true')
	parser.set_defaults(save_data=False)
	parser.set_defaults(heatmap=False)
	parser.set_defaults(scatter=False)
	parser.set_defaults(recall_plot=False)
	parser.set_defaults(rank_plot=False)
	args,_ = parser.parse_known_args()
	for key,value in vars(args).items():
		print('%s\t%s' % (key,str(value)))
	flat_data = load_flat_data(args.dataset)
	X1 = get_value_matrix(flat_data, max_year=args.max_year)
	X2 = get_value_matrix(flat_data, min_year=args.min_year, min_count=1)
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
	savepath_full = '%s/observation_end_%s.prediction_start_%s' % (args.savepath, args.max_year, args.min_year)
	if not os.path.exists(savepath_full):
		os.makedirs(savepath_full)
	if args.save_data:
		_=X1.to_csv(path_or_buf='%s/data.observed.csv' % (savepath_full))
		_=X2.to_csv(path_or_buf='%s/data.unobserved.csv' % (savepath_full))
		_=X_hat.to_csv(path_or_buf='%s/data.predicted.csv' % (savepath_full))
	if args.recall_plot:
		plot_recall(X2,X_hat,mask,args.high_titer_thresh,'%s/recall.png' % (savepath_full), args.max_year, args.min_year)
		plot_recall(X2,X_hat,mask,args.high_titer_thresh,'%s/recall_absolute.png' % (savepath_full), args.max_year, args.min_year, as_absolute=True)
	if args.scatter:
		plot_scatter(X2,X_hat,mask,'%s/scatter.png' % (savepath_full), rmse, r2, args.data_transform, args.max_year, args.min_year)
	if args.heatmap:
		plot_heatmap(X2,X_hat,mask,'%s/heatmap.png' % (savepath_full),args.data_transform,args.value_name)
	if args.rank_plot:
		plot_rank(X_hat,'%s/rank.png' % (savepath_full))


