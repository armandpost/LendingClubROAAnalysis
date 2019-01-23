import pandas as pd
import numpy as np

# For Building Histograms
from scipy.stats import norm
import matplotlib.mlab as mlab
import scipy.stats as st
import scipy.stats
from scipy import stats

# For Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# For Confusion Matrix
from sklearn.metrics import confusion_matrix
import itertools


def null_values(df):
	mis_val = df.isnull().sum()
	mis_val_percent = 100 * df.isnull().sum() / len(df)
	mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
	mis_val_table_ren_columns = mis_val_table.rename(
	columns = {0 : 'Missing Values', 1 : '% of Total Values'})
	mis_val_table_ren_columns = mis_val_table_ren_columns[
		mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
	print ("Dataframe has " + str(df.shape[1]) + " columns.\n"      
		"There are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.")
	return mis_val_table_ren_columns
		
### Create a Function to Build Histograms

def histogram(variable, name, color, bins):
    
    totalVals = variable.count
    missings = variable.isnull().sum(axis = 0).sum()
    variable2 = variable.dropna()
   
    # best fit of data
    (mu, sigma) = norm.fit(variable2)

    # the histogram of the data
    n, bins, patches = plt.hist(variable2, bins, density = True, facecolor = color, alpha = 0.75)
    
    # add a 'best fit' line
    y = scipy.stats.norm.pdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth = 2)

    #plot
    plt.xlabel(name + ': Missings = ' + str(missings))
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ ' + name + ':}\ \mu =%.3f,\ \sigma =%.3f$' %(mu, sigma))
    plt.grid(True)

    plt.show()

### Create Function for Barplots
  
def countplotshorizontal(frame, field, title, ylabel, palette):
    frame[field] = pd.Categorical(frame[field])
    g = sns.countplot(y = field, data = frame, order = frame[field].value_counts().index, palette = palette)
    g.set_ylabel(ylabel)
    g.set_title(title)
    plt.show()
    
### Create Function for Boxplots

def boxplot(x, y, **kwargs):
    sns.boxplot(x = x, y = y)
    x = plt.xticks(rotation = 90)
    
### Create Function Using Anova to Quickly Estimate Impact of Categorical Variables on Target

def anova(frame, List, target):
    anv = pd.DataFrame()
    anv['feature'] = List
    pvals = []
    for c in List:
        samples = []
        for cls in df[c].unique():
            s = df[df[c] == cls][target].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

### Create Functions to Quickly Assess the Correlation of Variables to the Target

def encode(frame, feature, target):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['spmean'] = frame[[feature, target]].groupby(feature).mean()[target]
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    ordering = ordering['ordering'].to_dict()
    
    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, feature + '_E'] = o
    
def spearman_df(frame, features, target):
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['spearman'] = [frame[f].corr(frame[target], 'spearman') for f in features]
    df_spr = spr.copy(deep = True)
    return df_spr

def spearman_chart(frame, features, target):
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['spearman'] = [frame[f].corr(frame[target], 'spearman') for f in features]
    spr = spr.sort_values('spearman')
    plt.figure(figsize=(6, 0.25 * len(features)))
    sns.barplot(data = spr, y = 'feature', x = 'spearman', orient = 'h')
	
def spearman_df_abs(frame, features, target):
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['spearman'] = [frame[f].corr(frame[target], 'spearman').astype(float).fabs() for f in features]
    df_spr = spr.copy(deep = True)
    return df_spr

def spearman_chart_abs(frame, features, target):
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['spearman'] = [frame[f].corr(frame[target], 'spearman').astype(float).fabs() for f in features]
    spr = spr.sort_values('spearman')
    plt.figure(figsize=(6, 0.25 * len(features)))
    sns.barplot(data = spr, y = 'feature', x = 'spearman', orient = 'h')

def stacked_bar(data, series_labels, category_labels = None, 
                show_values = False, value_format = "{}",
				y_label=None, x_label = None, ch_title = None,
                grid = True, reverse = False):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will 
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        axes.append(plt.bar(ind, row_data, bottom=cum_size, 
                            label=series_labels[i]))
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels, rotation = 'vertical')

    if y_label:
        plt.ylabel(y_label)
    if x_label:
        plt.xlabel(x_label)
    if ch_title:
        plt.title(ch_title)

    plt.legend()

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w/2, bar.get_y() + h/2, 
                         value_format.format(h), ha="center", 
                         va="center")
						 
### Create Function to Show Histogram and QQ Plot

def QQ(frame, variable, title):
    sns.distplot(frame[variable] , fit = norm);
    
    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(frame[variable])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    # Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc = 'best')
    plt.ylabel('Frequency')
    plt.title(title + ' Distribution')

    # Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(frame[variable], plot = plt)
    plt.show()
	
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')