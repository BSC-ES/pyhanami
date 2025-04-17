import numpy as np
from scipy.stats import bootstrap, norm


def area_weights(data):
    """ Compute weights area-averaging based on grid cell area """
    return np.cos(np.deg2rad(data.lat))


def exp_RK_index(data_sim, data_obs, var_name):
    """
    Calculate squared root of Reichler-Kim index variation for  
    an ensemble E, it incorporates an exponential to make it  
    more robust to small values of the standard deviation.

    Arguments
    ---------
    data_sim (xarray.Dataset): Climate simulation ensemble.
    data_obs (xarray.Dataset): Climate observations ensemble.
    var_name (str): Climate variable.

    Returns
    -------
    RK_index (numpy.ndarray): RK index for each ensemble member.
    """

    # Compute error variance for each grid cell
    obs_mean = data_obs.mean(dim='time')
    obs_std = data_obs.std(dim='time')
    err_var = (data_sim-obs_mean)/obs_std

    # Add weights and average
    err2_var = err_var*err_var
    err_var_exp = np.exp(-err2_var)
    weights = area_weights(data_sim)
    err_var_weighted = err_var_exp.weighted(weights)

    exp_RK_index = np.sqrt(err_var_weighted.mean(dim = ['lon', 'lat'])[var_name].values)

    return exp_RK_index


def ilamb_crms(data, var_name):
    """ Calculate centralized root mean square (RMS) along time. 

    Arguments
    ---------
    data (xarray.Dataset): Climate simulation ensemble.
    var_name (str): Climate variable.

    Returns
    -------
    v (xarray.Dataset): Centralized RMS for each grid cell and ensemble member.
    """

    v = data.load().copy()
    v_ref = v[var_name]
    v_mean = v_ref.mean(dim='time')
    v_diff = (v_ref-v_mean)**2

    dt = data.sizes['time']-1
    crms = np.sqrt(v_diff.sum(dim='time')/dt)

    v[var_name] = crms
    return v


def ilamb_crmse(data_sim, data_obs, var_name):
    """ 
    Calculate centralized root mean squared error (RMSE) along time. 
    
    Arguments
    ---------
    data_sim (xarray.Dataset): Climate simulation ensemble.
    data_obs (xarray.Dataset): Climate observations ensemble.
    var_name (str): Climate variable.

    Returns
    -------
    vs (xarray.Dataset): Centralized RMSE for each grid cell and ensemble member.
    """

    vs = data_sim.load().copy()
    v_mod = vs[var_name]
    v_mmean = v_mod.mean(dim='time')
    v_mdiff = (v_mod-v_mmean)

    vo = data_obs.load().copy()
    v_ref = vo[var_name]
    v_rmean = v_ref.mean(dim='time')
    v_rdiff = (v_ref-v_rmean)

    diff = ((v_mdiff - v_rdiff)**2)
    dt = data_sim.sizes['time']-1
    crmse = np.sqrt(diff.sum(dim='time')/dt)

    vs[var_name] = crmse
    return vs


def ilamb_weighted_bias(data_sim, data_obs, var_name):
    """
    Calculate weighted bias score for an ensemble E.

    Arguments
    ---------
    data_sim (xarray.Dataset): Climate simulation ensemble.
    data_obs (xarray.Dataset): Climate observations ensemble.
    var_name (str): Climate variable.

    Returns
    -------
    S_bias (numpy.ndarray): Bias for each ensemble member.
    """

    # Compute bias for each grid cell
    v_mmean = data_sim
    v_rmean = data_obs.mean(dim='time')
    bias = np.abs(v_mmean - v_rmean)
    
    cmrs = ilamb_crms(data_obs, var_name)
    e_bias = bias/cmrs
    s_bias = np.exp(-e_bias)
    

    # Weighted mean
    weights = area_weights(data_sim)
    s_weighted = s_bias.weighted(weights)
    S_bias = s_weighted.mean(dim = ['lon', 'lat'])[var_name].values

    return S_bias


def ilamb_weighted_RMSE(data_sim, data_obs, var_name):
    """
    Calculate weighted root mean square error (RMSE) score for an ensemble E.

    Arguments
    ---------
    data_sim (xarray.Dataset): Climate simulation ensemble.
    data_obs (xarray.Dataset): Climate observations ensemble.
    var_name (str): Climate variable.

    Returns
    -------
    S_rmse (numpy.ndarray): RMSE for each ensemble member.
    """

    # Compute RMSE for each grid cell
    crmse = ilamb_crmse(data_sim, data_obs, var_name)
    crms = ilamb_crms(data_obs, var_name)

    e_rmse = crmse/crms
    s_rmse = np.exp(-e_rmse)


    # Weighted mean
    weights = area_weights(data_sim)
    s_weighted = s_rmse.weighted(weights)
    S_rmse = s_weighted.mean(dim = ['lon', 'lat'])[var_name].values

    return S_rmse


def cp_effect_size(sample_1, sample_2):
    """
    Compute Cohen's d effect size between two samples using
    the pooled standard deviation.

    Arguments
    ---------
    sample_1 (numpy.ndarray): Sample.
    sample_2 (numpy.ndarray): Sample.

    Returns
    -------
    d (float): Cohen's d effect size
    """

    mean_1 = np.mean(sample_1) 
    mean_2 = np.mean(sample_2) 

    std_1 = np.std(sample_1)
    std_2 = np.std(sample_2)
    std_pooled = np.sqrt((std_1**2+std_2**2)/2)

    d = (mean_1 - mean_2)/std_pooled
    return d


def bootstrap_test(score_ref, score_test, bstat=cp_effect_size):
    """
    Perform statistical test based on the bootstrap method
    to check whether two distributions of scores come from
    the same underlying distribution.

    Arguments
    ---------
    score_ref (numpy.ndarray): Reference sample.
    score_test (numpy.ndarray): Test sample.
    bstat (callable): Statistic to be used by the bootstrap method.

    Returns
    -------
    d (float): Mean of the bootstrap distribution.
    sigma (float): Standard deviation of the bootstrap distribution.
    p_value (float): Outcome of the bootstrap test (True for rejection).
    """

    res = bootstrap((score_ref, score_test), bstat, confidence_level=0.95)
    d = np.mean(res.bootstrap_distribution)
    sigma = np.std(res.bootstrap_distribution)

    if  d < 0:
        p_value = 2*norm.cdf(d, loc=0, scale=sigma)
    elif d > 0:
        p_value = 2*(1-norm.cdf(d, loc=0, scale=sigma))
    else:
        p_value = 1.0

    return d, sigma, p_value