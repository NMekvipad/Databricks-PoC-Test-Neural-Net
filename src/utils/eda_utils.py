import numpy as np
import plotly.express as px
from scipy.stats import sigmaclip
from pandas import ExcelWriter


def save_xls(df_list, xls_path, sheet_names=None):
    with ExcelWriter(xls_path) as writer:

        if sheet_names is not None:
            for name, df in zip(sheet_names, df_list):
                df.to_excel(writer, sheet_name=name)
        else:
            for n, df in enumerate(df_list):
                df.to_excel(writer, sheet_name='sheet%s' % n)
        writer.save()


# clipper for power law distributed variables with extremeley large value
def fat_tail_clipper(x):
    """

    Args:
        x: input array

    Returns:

    """

    x_min = x.min()
    x_max = x.max()
    x_mean = x.mean()
    

    # lower tail
    if x_min == 0:
        lower_bound = 0
    else:
        min_order = np.log10(np.absolute(x_min))
        mean_order = np.log10(np.absolute(x_mean))
        
        if x_mean < 0 and min_order - mean_order > 2:
            order = np.floor((mean_order + min_order)/2 * -1) * -1
            
            if order == 0:
                order = 1
                
            lower_bound = 10 ** order  * -1
        elif x_mean < 0:
            order = np.floor(mean_order * -1) * -1
            
            if order == 0:
                order = 1
            
            lower_bound = 10 ** order  * -1
        else:
            order = np.clip(np.floor(min_order * -1), -5, -1) * -1
            
            if order == 0:
                order = 1
            
            lower_bound = 10 ** order  * -1
    
    # upper tail
    if x_max == 0:
        upper_bound = 0
    else:
        max_order = np.log10(np.absolute(x_max))
        mean_order = np.log10(np.absolute(x_mean))
        
        if x_mean > 0 and max_order - mean_order > 2:
            order = np.ceil((mean_order + max_order)/2 - 1)
            
            if order == 0:
                order = 1
                
            upper_bound = 10 ** order
            
        elif x_mean > 0:
            order = np.ceil(mean_order - 1)
            
            if order == 0:
                order = 1
                
            upper_bound = 10 ** order
            
        else:
            order = np.ceil(np.clip(mean_order, 1, 5) - 1)
            
            if order == 0:
                order = 1
                
            upper_bound = 10 ** order            
 

    return np.clip(x, lower_bound, upper_bound), lower_bound, upper_bound

# clipper for gaussian variables
def sigma_clipper(x): 
    c, low, upp = sigmaclip(x)
    clipped_value = np.clip(x, low, upp)
    
    return clipped_value, low, upp

def series_clipper(x):
    clipped, lower_bound, upper_bound = sigma_clipper(x)
    
    if lower_bound == 0 and upper_bound == 0:
        clipped, lower_bound, upper_bound = fat_tail_clipper(x)
        
    return clipped

def plot_distribution_over_label(df, columns=None, clipper=sigma_clipper, **kwargs):
    
    plot_dict = dict()
    
    if columns is None:
        use_cols = df.columns
    else:
        use_cols = columns
        
    plotting_args = kwargs    
    
    for col in use_cols:   
        
        tmp_args = plotting_args.copy()
        tmp_args['title'] = col         
        
        if df[col].max() < tmp_args['nbins']:
            tmp_args['nbins'] = df[col].max()            
        
        if clipper is None:
            fig = px.histogram(df, x=df[col], **tmp_args)
        else:
            fig = px.histogram(df, x=clipper(df[col]), **tmp_args)
            
        plot_dict[col] = fig
    
    return plot_dict



