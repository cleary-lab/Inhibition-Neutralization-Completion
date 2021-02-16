import numpy as np
from parse_datasets import load_flat_data, get_value_matrix, get_common_subset
from matrix_completion import nuclear_norm_solve, svt_solve # https://github.com/tonyduan/matrix-completion
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

def get_mask(X,observed_fraction):
	# NOTE on alternative methods
	# balanced random: choosing the same number of samples per row or column
	#	no better than full random, often worse, algorithm often hangs (unclear why)
	# full row and column: choosing a random subset of rows and columns to observe
	#	decidely worse than full random
	not_available = np.isnan(X.values)
	available_idx = np.where(1-not_available.flatten())[0]
	n = len(available_idx)
	p = int(np.round(n*observed_fraction))
	mask = np.zeros(np.product(X.shape),dtype=np.int)
	mask[np.random.choice(available_idx,p,replace=False)] = 1
	mask = mask.reshape(X.shape)
	return mask

def complete_matrix(X,mask,offset=True,lda=1):
	if offset:
		min_val = X.values[np.invert(np.isnan(X.values))].min()
		X_hat = nuclear_norm_solve(X-min_val,mask,mu=lda) + min_val
	else:
		X_hat = nuclear_norm_solve(X,mask,mu=lda)
	return X_hat

# seems to be no better than nuclear_norm_solve (constraint in objective)
def cvx_nuc(A,mask):
	X = cp.Variable(shape=A.shape)
	objective = cp.Minimize(cp.norm(X, "nuc"))
	constraints = [X[np.where(mask)] == A.values[np.where(mask)]]
	problem = cp.Problem(objective, constraints)
	problem.solve(solver=cp.SCS)
	return X.value

def calc_unobserved_rmse(X,X_hat,mask):
	available = np.invert(np.isnan(X.values))
	idx = np.where((1-mask)*available)
	return np.average((X.values[idx]-X_hat[idx])**2)**.5

def calc_unobserved_r2(X,X_hat,mask):
	available = np.invert(np.isnan(X.values))
	return (1-distance.correlation(X.values[np.where((1-mask)*available)].flatten(),X_hat[np.where((1-mask)*available)].flatten()))**2

def calc_observed_rmse(X,X_hat,mask):
	available = np.invert(np.isnan(X.values))
	idx = np.where(mask*available)
	return np.average((X.values[idx]-X_hat[idx])**2)**.5

def calc_observed_r2(X,X_hat,mask):
	available = np.invert(np.isnan(X.values))
	return (1-distance.correlation(X.values[np.where(mask*available)].flatten(),X_hat[np.where(mask*available)].flatten()))**2

def observed_fraction_curve(X):
	Results = []
	Label = []
	P = []
	for p in np.linspace(0.05,0.95,19):
		for _ in range(25):
			mask = get_mask(X,p)
			X_hat = complete_matrix(X,mask,offset=True)
			rmse = calc_unobserved_rmse(X, X_hat, mask)
			r2 = calc_unobserved_r2(X, X_hat, mask)
			P.append(p)
			Label.append('r^2')
			Results.append(r2)
			P.append(p)
			Label.append('rmse')
			Results.append(rmse)
	return Results, Label, P

def plot_results_curves(Results,Label,P,filename):
	df = pd.DataFrame({'value': Results, 'observed fraction': P, 'statistic': Label})
	g = sns.FacetGrid(df, col="statistic", col_wrap=1, hue='statistic', sharey=False)
	g = (g.map_dataframe(sns.lineplot, "observed fraction", "value"))
	plt.savefig(filename)
	plt.close()

def plot_heatmap(X,obs_frac,filename,combined_datasets=False):
	available = np.invert(np.isnan(X.values))
	mask = get_mask(X,obs_frac)
	X_hat = pd.DataFrame(complete_matrix(X,mask,offset=True), index=X.index, columns=X.columns)
	rmse = calc_unobserved_rmse(X, X_hat.values, mask)
	r2 = calc_unobserved_r2(X, X_hat.values, mask)
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
	if combined_datasets:
		df = df.rename_axis(['Unobserved','VaccineStatus','Neutralization[Titers]'])
	else:
		df = df.rename_axis(['Unobserved','Neutralization[Titers]'])
	df_mask = np.vstack([mask,mask])
	fig, ax = plt.subplots(figsize=(8,12))
	_=plt.title('%.1f%% of available entries unobserved; RMSE=%.3f; r^2=%.1f%%' % (100 - np.average(mask)*100, rmse, r2*100))
	ax=sns.heatmap(df,mask=df_mask,ax=ax,cmap=sns.cm.rocket_r)
	_=ax.axhline(X.shape[0],color='blue')
	_=plt.tight_layout()
	bottom, top = ax.get_ylim()
	ax.set_ylim(bottom + 0.5, top - 0.5) # sorry, this may cut off the bottom row...some sort of matplotlib bug
	plt.savefig(filename)
	plt.close()

def plot_scatter(X,obs_frac,filename):
	available = np.invert(np.isnan(X.values))
	mask = get_mask(X,obs_frac)
	X_hat = pd.DataFrame(complete_matrix(X,mask,offset=True), index=X.index, columns=X.columns)
	rmse = calc_unobserved_rmse(X, X_hat.values, mask)
	r2 = calc_unobserved_r2(X, X_hat.values, mask)
	_=sns.regplot(x=X.values[np.where((1-mask)*available)], y = X_hat.values[np.where((1-mask)*available)], x_jitter=.1, scatter_kws={'alpha': 0.35})
	_=plt.xlabel('True titer (log10)')
	_=plt.ylabel('Predicted titer (log10)')
	_=plt.title('%.1f%% of entries unobserved; RMSE=%.3f; r^2=%.1f%%' % (100-100*np.average(mask), rmse, 100*r2))
	plt.savefig(filename)
	plt.close()

def plot_recall(X,obs_frac,filename,thresh,n=25):
	available = np.invert(np.isnan(X.values))
	df = {'Total fraction of entries observed (random sample + validation)': [], 'Total fraction of high titers > %.2f identified' % (thresh): []}
	for _ in range(n):
		mask = get_mask(X,obs_frac)
		X_hat = pd.DataFrame(complete_matrix(X,mask,offset=True), index=X.index, columns=X.columns)
		x = X.values[np.where((1-mask)*available)].flatten()
		x_hat = X_hat.values[np.where((1-mask)*available)].flatten()
		xs = np.argsort(-x_hat)
		x_hat_frac = np.arange(1,len(x)+1)/len(x)*(1-obs_frac) + obs_frac
		x_frac = np.cumsum((x[xs] >= thresh)/(X.values >= thresh).sum()) + (X.values[np.where(mask)] >= thresh).sum()/(X.values >= thresh).sum()
		df['Total fraction of entries observed (random sample + validation)'].append(x_hat_frac)
		df['Total fraction of high titers > %.2f identified' % (thresh)].append(x_frac)
	x_frac_avg = np.average(df['Total fraction of high titers > %.2f identified' % (thresh)],axis=0)
	df['Total fraction of entries observed (random sample + validation)'] = np.hstack(df['Total fraction of entries observed (random sample + validation)'])
	df['Total fraction of high titers > %.2f identified' % (thresh)] = np.hstack(df['Total fraction of high titers > %.2f identified' % (thresh)])
	df = pd.DataFrame(df)
	ax=sns.lineplot(data=df,x='Total fraction of entries observed (random sample + validation)', y='Total fraction of high titers > %.2f identified' % (thresh), ci='sd')
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
	_=plt.title('Recall of high titer entries from matrix completion (~%.1f%% unobserved) + validation' % (100-100*np.average(mask)), fontsize=8)
	_=plt.tight_layout()
	plt.savefig(filename)
	plt.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', help='Path to data (csv flatfile)')
	parser.add_argument('--savepath', help='Path to save results')
	parser.add_argument('--antibody-col-name', help='Column name for antibodies', default='Antibody')
	parser.add_argument('--data-transform', help='Type of transform to apply to raw data',default='neglog10', choices=('raw','neglog10','log10'))
	parser.add_argument('--high-titer-thresh', help='Threshold for high titer values (log10 units)',default=1.6,type=float)
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
	flat_data = load_flat_data(args.dataset)
	X = get_value_matrix(flat_data, rows=args.antibody_col_name)
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
	if args.rmse_r2_curves:
		R,L,P = observed_fraction_curve(X)
		plot_results_curves(R,L,P,'%s/rmse_r2.png' % savepath_full)
	if args.heatmap:
		plot_heatmap(X,args.example_obs_frac,'%s/heatmap.png' % savepath_full)
	if args.scatter:
		plot_scatter(X,args.example_obs_frac,'%s/scatter.png' % savepath_full)
	if args.recall_plot:
		plot_recall(X,args.example_obs_frac,'%s/recall.png' % savepath_full, args.high_titer_thresh)

