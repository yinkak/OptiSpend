import pymc as pm
import pytensor.tensor as pt
import pandas as pd
import numpy as np

def geometric_adstock(x, alpha, l_max=4):
    """
    Applies Geometric Adstock (Decay) to a spend channel.
    alpha: The retention rate (how much memory is kept each week).
    l_max: Maximum number of weeks the ad is remembered.
    """
    # Create the decay weights (e.g., 1, alpha, alpha^2, alpha^3)
    w = pt.as_tensor_variable([pt.power(alpha, i) for i in range(l_max)])
    
    # Create shifted versions of the spend data for the past 'l_max' weeks
    x_lags = pt.stack([
        pt.concatenate([pt.zeros(i), x[:x.shape[0]-i]]) 
        for i in range(l_max)
    ])
    
    # Multiply spend by the decay weights
    return pt.dot(w, x_lags)

def logistic_saturation(x, lam):
    """
    Applies a Logistic Saturation curve (Diminishing Returns).
    lam: The shape parameter (how quickly the channel saturates).
    """
    return 1 - pt.exp(-lam * x)

def build_mmm(X_df, y_array):
    """
    Builds the PyMC Marketing Mix Model.
    X_df: DataFrame of scaled marketing spend.
    y_array: Array of scaled sales values.
    """
    channels = X_df.columns.tolist()
    
    with pm.Model() as mmm:
        # 1. Baseline Sales (Sales with zero marketing)
        baseline = pm.Exponential('baseline', lam=1.0)
        
        # List to hold the modeled impact of each channel
        channel_contributions = []
        
        # 2. Loop through each channel to apply Adstock and Saturation
        for channel in channels:
            channel_data = X_df[channel].values
            
            # Priors (Our educated guesses)
            beta = pm.Exponential(f'beta_{channel}', lam=1.0)      # How strong is the channel?
            alpha = pm.Beta(f'alpha_{channel}', alpha=2, beta=2)   # How long is the memory?
            lam = pm.Gamma(f'lam_{channel}', alpha=3, beta=1)      # How fast does it saturate?
            
            # Apply transformations
            adstocked = geometric_adstock(channel_data, alpha)
            saturated = logistic_saturation(adstocked, lam)
            
            # Calculate final contribution for this channel
            contribution = pm.Deterministic(
                f'contribution_{channel}', 
                beta * saturated
            )
            channel_contributions.append(contribution)
            
        # 3. Add them all together to get Total Expected Sales
        mu = baseline + sum(channel_contributions)
        
        # 4. Likelihood (Compare expected sales to actual data)
        sigma = pm.Exponential('sigma', lam=1.0) # The "noise" in the data
        sales_obs = pm.Normal(
            'sales_obs', 
            mu=mu, 
            sigma=sigma, 
            observed=y_array
        )
        
    return mmm


