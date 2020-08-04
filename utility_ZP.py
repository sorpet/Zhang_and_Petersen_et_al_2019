import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.interpolate import UnivariateSpline

def splines(row):
    '''Returns od, gfp, and gfp/od splines as well as predicted gfp_p_od values for each line/strain in strains dataframe'''
    
    time_range = row['time_series'].time.values[row['index_gfp_min']:row['index_gfp_max']+1]
    od_range   = row['time_series'].od_minus_bg.values[row['index_gfp_min']:row['index_gfp_max']+1]
    gfp_range  = row['time_series'].gfp_minus_bg.values[row['index_gfp_min']:row['index_gfp_max']+1]
    
    spl_od     = UnivariateSpline(time_range,od_range, s=0.05)
    spl_gfp    = UnivariateSpline(time_range,gfp_range, s=150000)
    
    gfp_p_od   = spl_gfp(time_range) / spl_od(time_range)
    spl_gfp_p_od = UnivariateSpline(time_range, gfp_p_od, s=100000)
    
    return (spl_od, spl_gfp, spl_gfp_p_od, gfp_p_od)


def find_times(row):
    '''Returns 4 timepoints for each line/strain in strains dataframe'''
    t_min           = row['time_series'].time[:row['index_gfp_max']].min()
    t_max           = row['time_series'].time[:row['index_gfp_max']].max()
    xtime           = np.linspace(t_min,t_max,1000)
    t1 = 0.075
    t2 = 0.15
    
    od_pred         = row['spl_od'](xtime)
    index_od_t1     = abs(od_pred - t1).argmin()
    t_od_t1         = xtime[index_od_t1].round(2)
    
    index_od_t2     = abs(od_pred - t2).argmin()
    t_od_t2         = xtime[index_od_t2].round(2)
    
    gfp_prime_pred  = row['spl_gfp'](xtime,1)
    index_r_gfp_max = gfp_prime_pred.argmax()
    t_r_gfp_max     = xtime[index_r_gfp_max].round(2)
    
    od_prime_pred   = row['spl_od'](xtime,1)
    index_r_od_max  = od_prime_pred.argmax()
    t_r_od_max      = xtime[index_r_od_max].round(2)
    r_od_max = od_prime_pred[index_r_od_max]
    
    return (t_od_t1,t_od_t2, t_r_gfp_max, t_r_od_max, r_od_max)

def plot_strain_by_index(df, lst):
    '''input list of indexes to df strains and plot od, gfp, & gfp/od as function of time '''
    no_of_strains = len(lst)
    
    fig = plt.figure(figsize=(15, 15*no_of_strains))
    fig.set_tight_layout(True)

    for i,no in enumerate(lst):
        index_gfp_min        = df.index_gfp_min[no]
        index_gfp_max        = df.index_gfp_max[no]
        
        time                 = df.time_series[no].time.values
        adj_time             = time[index_gfp_min:index_gfp_max+1]
        
        xtime                = np.linspace(min(time), max(time), 1000)
        adj_xtime            = np.linspace(min(time[index_gfp_min:]), max(time[:index_gfp_max+1]), 1000)
        
        t_od_t1              = df.t_od_t1[no]
        t_od_t2              = df.t_od_t2[no]
        #t_r_gfp_max          = df.t_r_gfp_max[no]
        t_r_od_max           = df.t_r_od_max[no]
        
        ax = fig.add_subplot(2*no_of_strains, 3, (3*i)+1)
        ax.plot(time        , df.time_series[no].od_minus_bg.values , 'ko', ms=5, alpha = 0.5)
        ax.plot(adj_xtime   , df.spl_od[no](adj_xtime)              , 'r')
        ax.plot(t_od_t1     , df.od_t_od_t1[no]                     , 'go', markersize = 16)
        ax.plot(t_od_t2     , df.od_t_od_t2[no]                     , 'yo', markersize = 16)
        #ax.plot(t_r_gfp_max , df.od_t_r_gfp_max[no]                 , 'ro', markersize = 16)
        ax.plot(t_r_od_max , df.od_t_r_od_max[no]                  , 'bo', markersize = 16)                      
        #ax.plot(xtime      , df.spl_od[no](xtime,1)                , 'b')
        ax.set_xlabel('time'                                             ,fontsize=25)
        ax.set_ylabel('od'                                               ,fontsize=25)
        ax.set_title(df['Line Name'][no] + '_' + df['replicate_count'][no],fontsize=16)
    
        ax = fig.add_subplot(2*no_of_strains, 3, (3*i)+2)
        ax.plot(time        , df.time_series[no].gfp_minus_bg.values , 'ko', ms=5, alpha = 0.5)
        ax.plot(adj_xtime   , df.spl_gfp[no](adj_xtime)              , 'r')
        ax.plot(t_od_t1     , df.gfp_t_od_t1[no]                     , 'go', markersize = 16)
        ax.plot(t_od_t2     , df.gfp_t_od_t2[no]                     , 'yo', markersize = 16)
        #ax.plot(t_r_gfp_max , df.gfp_t_r_gfp_max[no]                 , 'ro', markersize = 16)
        ax.plot(t_r_od_max , df.gfp_t_r_od_max[no]                  , 'bo', markersize = 16)
        #ax.plot(xtime      , df.spl_gfp[no](xtime,1)                , 'b')
        ax.set_xlabel('time'                                              ,fontsize=25)
        ax.set_ylabel('gfp'                                               ,fontsize=25)
        #ax.set_title(df['Line Name'][no] + '_' + df['replicate_count'][no],fontsize=16)
        
        ax = fig.add_subplot(2*no_of_strains, 3, (3*i)+3)
        ax.plot(adj_time        , df.gfp_p_od_ts[no]                  , 'ko', ms=5, alpha = 0.5)
        ax.plot(adj_xtime       , df.spl_gfp_p_od[no](adj_xtime)      , 'r')
        ax.plot(t_od_t1         , df.gfp_p_od_t_od_t1[no]             , 'go', markersize = 16)
        ax.plot(t_od_t2         , df.gfp_p_od_t_od_t2[no]             , 'yo', markersize = 16)
        #ax.plot(t_r_gfp_max     , df.gfp_p_od_t_r_gfp_max[no]         , 'ro', markersize = 16)
        ax.plot(t_r_od_max     , df.gfp_p_od_t_r_od_max[no]          , 'bo', markersize = 16)
        #ax.plot(xtime          , df.spl_gfp_p_od[no](xtime,1)        , 'b')
        ax.set_xlabel('time'                                               ,fontsize=25)
        ax.set_ylabel('gfp_p_od'                                           ,fontsize=25)
        #ax.set_title(df['Line Name'][no] + '_' + df['replicate_count'][no],fontsize=16)
    fig.savefig('./figures/Figure S4.pdf')
