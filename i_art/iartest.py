import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from sklearn.base import clone
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.exceptions import ConvergenceWarning
from sklearn import linear_model
import lightgbm as lgb
import xgboost as xgb
from sklearn.impute import SimpleImputer
import time
from sklearn.exceptions import DataConversionWarning
import warnings

def holm_bonferroni(p_values, alpha = 0.05):
    """
    Perform the Holm-Bonferroni correction on the p-values
    """

    # Perform the Holm-Bonferroni correction
    reject, corrected_p_values, _, _ = multipletests(p_values, alpha=alpha, method='holm')

    # Check if any null hypothesis can be rejected
    any_rejected = any(reject)

    return any_rejected

def getY(G, Z, X,Y, covariate_adjustment = False):
    """
    Calculate the imputed Y values using G and df_Z
    if covariate_adjustment is True, return the adjusted Y values based on predicted Y values and X
    else return the predicted Y values
    """
    if covariate_adjustment:
        G_adjusted = clone(G)
    df_Z = pd.DataFrame(np.concatenate((Z, X, Y), axis=1))
    # lenY is the number of how many columns are Y
    lenY = Y.shape[1]
    # indexY is the index of the first column of Y
    indexY = Z.shape[1] + X.shape[1]
    # fit the imputation model G
    df_imputed = G.fit_transform(df_Z)

    Y_head = df_imputed[:, indexY:indexY+lenY]
    if covariate_adjustment:
        warnings.filterwarnings(action='ignore', category=DataConversionWarning)
        G_adjusted.fit(X, Y_head)
        Y_head2 = G_adjusted.predict(X)
        return Y_head - Y_head2
    else:
        return Y_head

def T(z,y):
    """
    Calculate the Wilcoxon rank sum test statistics
    """

    #the Wilcoxon rank sum test
    n = len(z)
    t = 0
    my_list = []
    for i in range(n):
        my_list.append((z[i],y[i]))
    sorted_list = sorted(my_list, key=lambda x: x[1])
    for i in range(n):
        t += sorted_list[i][0] * (i + 1)

    return t

def split(y, z, M):
    """
    Split the data into missing and non-missing parts
    """
    
    missing_indices = M[:].astype(bool)
    non_missing_indices = ~missing_indices

    y_missing = y[missing_indices].reshape(-1,)
    y_non_missing = y[non_missing_indices].reshape(-1,)

    z_missing = z[missing_indices].reshape(-1,)
    z_non_missing = z[non_missing_indices].reshape(-1,)

    return y_missing, y_non_missing, z_missing, z_non_missing

def getT(y, z, lenY, M):
    """
    Separately calculate T for missing and non-missing parts of each outcome using Wilcoxon rank sum test
    Return the sum of T values for all outcomes
    """

    t = []
    for i in range(lenY):
        # Split the data into missing and non-missing parts using the split function
        y_missing, y_non_missing, z_missing, z_non_missing = split(y[:,i], z, M[:,i])
        
        # Calculate T for missing and non-missing parts
        t_missing = T(z_missing, y_missing.reshape(-1,))
        t_non_missing = T(z_non_missing, y_non_missing.reshape(-1,))

        # Sum the T values for both parts
        t_combined = t_missing + t_non_missing
        t.append(t_combined)

    return np.array(t)

def getZsimTemplates(Z_sorted, S):
    """
    Create a Z_sim template for each unique value in S
    """

    # Create a Z_sim template for each unique value in S
    Z_sim_templates = []
    unique_strata = np.unique(S)
    for stratum in unique_strata:
        strata_indices = np.where(S == stratum)[0]
        strata_Z = Z_sorted[strata_indices]
        p = np.mean(strata_Z)
        strata_size = len(strata_indices)
        Z_sim_template = [0.0] * int(strata_size * (1 - p)) + [1.0] * int(strata_size * p)
        Z_sim_templates.append(Z_sim_template)
    return Z_sim_templates

def getZsim(Z_sim_templates):
    """ 
    Shuffle each Z_sim template and concatenate them into a single permutated Z_sim array 
    """

    Z_sim = []
    for Z_sim_template in Z_sim_templates:
        strata_Z_sim = np.array(Z_sim_template.copy())
        np.random.shuffle(strata_Z_sim)
        Z_sim.append(strata_Z_sim)
    Z_sim = np.concatenate(Z_sim).reshape(-1, 1)
    return Z_sim

def preprocess(Z, X, Y, S):
    """ 
    Preprocess the input variables, including reshaping, concatenating, sorting, and extracting
    """

    # Reshape Z, X, Y, S, M to (-1, 1) if they're not already in that shape
    X = X.reshape(-1, X.shape[1])
    Z = Z.reshape(-1, 1)
    if S == None:
        S = np.ones(Z.shape)
        M = np.isnan(Y).reshape(-1, Y.shape[1])
        return Z, X, Y, S, M
    S = S.reshape(-1, 1)

    # Concatenate Z, X, Y, S, and M into a single DataFrame
    df = pd.DataFrame(np.concatenate((Z, X, Y, S), axis=1))

    # Sort the DataFrame based on S (assuming S is the column before M)
    df = df.sort_values(by=df.columns[-1])

    # Extract Z, X, Y, S, and M back into separate arrays
    Z = df.iloc[:, :Z.shape[1]].values.reshape(-1, 1)
    X = df.iloc[:, Z.shape[1]:Z.shape[1] + X.shape[1]].values.reshape(-1, X.shape[1])
    Y = df.iloc[:, Z.shape[1] + X.shape[1]:Z.shape[1] + X.shape[1] + Y.shape[1]].values.reshape(-1, Y.shape[1])
    S = df.iloc[:, Z.shape[1] + X.shape[1] + Y.shape[1]:Z.shape[1] + X.shape[1] + Y.shape[1] + S.shape[1]].values.reshape(-1, 1)

    M = np.isnan(Y).reshape(-1, Y.shape[1])
    return Z, X, Y, S, M


def check_param(Z, X, Y, S, G, L, verbose, covariate_adjustment,alpha,alternative,random_state):
    """
    Check the validity of the input parameters
    """

    # List of variables to check
    variables = [Z, X, Y]

    # Check if each variable is a numpy array
    for var in variables:
        if not isinstance(var, np.ndarray):
            raise ValueError("Z, X, Y, S must be numpy.ndarray")

    # Check Z: must be one of 1, 0, 1.0, 0.0
    if not np.all(np.isin(Z, [0, 1])):
        raise ValueError("Z must contain only 0, 1")

    # Check X: must be a 2D array
    if len(X.shape) != 2:
        raise ValueError("X must be a 2D array")
    
    # Check Y: must be a 2D array
    if len(Y.shape) != 2:
        raise ValueError("Y must be a 2D array")

    # Check S: must all be integer
    if S != None and not np.all(np.equal(S, S.astype(int))):
        raise ValueError("S must contain only integer values")

    # Check L: must be an integer greater than 0
    if not isinstance(L, int) or L <= 0:
        raise ValueError("L must be an integer greater than 0")

    # Check verbose: must be True or False
    if verbose not in [True, False, 1, 0]:
        raise ValueError("verbose must be True or False")

    # Check alpha: must be > 0 and <= 1
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be greater than 0 and less than or equal to 1")
    
    # Check G: Cannot be None
    if G is None:
        raise ValueError("G cannot be None")
    
    # Check covariate_adjustment: must be True or False
    if covariate_adjustment not in [True, False, 1, 0]:
        raise ValueError("covariate_adjustment must be True or False")

    # Check alternative: must be one of "one-sided" or "two-sided"
    if alternative not in ["one-sided", "two-sided"]:
        raise ValueError("alternative must be one of 'one-sided' or 'two-sided'")
    
    # Check random_state: must be an integer greater than 0 or None
    if random_state != None and (not isinstance(random_state, int) or random_state <= 0):
        raise ValueError("random_state must be an integer greater than 0 or None")
    
def choosemodel(G):
    """ 
    Choose the imputation model based on the input parameter G.
    If G is a string, choose the imputation model based on the string.
    If G is a function, return the function.
    """

    #if G is string
    if isinstance(G, str):
        G = G.lower()
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        if G == 'xgboost':
            G = IterativeImputer(estimator = xgb.XGBRegressor(), max_iter = 1)
        if G == 'bayesianridge':
            G = IterativeImputer(estimator = linear_model.BayesianRidge(), max_iter = 1,verbose=0)
        if G == 'median':
            G = SimpleImputer(missing_values=np.nan, strategy='median')
        if G == 'mean':
            G = SimpleImputer(missing_values=np.nan, strategy='mean')
        if G == 'lightgbm':
            G = IterativeImputer(estimator = lgb.LGBMRegressor(verbosity = -1), max_iter = 1)
        if G == 'mice':
            G = IterativeImputer(estimator = linear_model.BayesianRidge())
        if G == 'mice+lightgbm':
            G = IterativeImputer(estimator = lgb.LGBMRegressor(verbosity = -1))
        if G == 'mice+xgboost':
            G = IterativeImputer(estimator = xgb.XGBRegressor())
    return G

def transformX(X, threshold=0.1, verbose=True):
    """
    Imputes columns in the array X with a missing rate below the given threshold using median imputation.
    Parameters:
        X (numpy.ndarray): The data array with potential missing values (NaN).
        threshold (float): Missing rate threshold for imputation. Defaults to 0.1 (10%).
        verbose (bool): Whether to print information about the transformation.
        
    Returns:
        numpy.ndarray: Transformed data array.
    """
    
    # Step 1: Calculate missing rate for each column
    missing_rate = np.isnan(X).mean(axis=0)
    
    # Step 2: Identify columns with missing rate < threshold and > 0
    columns_to_impute = np.where((missing_rate < threshold) & (missing_rate > 0))[0]
    
    # Step 3: Impute missing values in selected columns with median
    imputer = SimpleImputer(strategy='median')
    imputed_columns = []
    for col in columns_to_impute:
        X[:, col] = imputer.fit_transform(X[:, col].reshape(-1, 1)).ravel()
        imputed_columns.append(col)
    
    # Calculate missing rate after imputation
    missing_rate_after = np.isnan(X).mean(axis=0)
    
    # Columns that are not imputed
    not_imputed_columns = [col for col in range(X.shape[1]) if col not in imputed_columns]
    
    if verbose:
        print(f"Missing Rate Before Imputation for X: {missing_rate * 100}")
        
        if len(columns_to_impute)>0:
            print(f"Missing Rate After Imputation for X: {missing_rate_after * 100}")
            print(f"Columns Imputed for X: {imputed_columns}")
        print(f"Columns Not Imputed for X: {not_imputed_columns}")

    return X

def iartest(*,Z, X, Y, G='bayesianridge', S=None,L = 10000,threshholdForX = 0.1,verbose = False, covariate_adjustment = False, alpha = 0.05, alternative = "one-sided",random_state=None):
    """
    I-ART: Imputation-Assisted Randomization Tests
 
    Usage Example:
    --------------
    >> Z = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    >> X = np.array([[5.1, 3.5], [4.9, np.nan], [4.7, 3.2], [4.5, np.nan], [7.2, 2.3], [8.6, 3.1], [6.0, 3.6], [8.4, 3.9]])
    >> Y = np.array([[4.4, 0.5], [4.3, 0.7], [4.1, np.nan], [5.0, 0.4], [1.7, 0.1], [np.nan, 0.2], [1.4, np.nan], [1.7, 0.4]])
    >> result = iartest(Z=Z,X=X,Y=Y,L=1000,verbose=1)

    Args:
    Z: 2D array of observed treatment indicators
    X: 2D array of observed covariates
    Y: 2D array of K outcomes with missing values
    S: 2D array of the strata indicators
    threshholdForX: threshhold for missing outcome to be imputed in advance
    G: a string for the eight available choice or a function that takes (Z, M, Y_k) as input and returns the imputed complete values 
    L: number of Monte Carlo simulations (default is 10000)
    verbose: a boolean indicating whether to print training start and end (default is False)
    covarite_adjustment: a boolean indicating whether to do covariate adjustment (default is False)
    alpha: significance level (default is 0.05)
    alternative: a string indicating the alternative hypothesis (default is "one-sided")
    random_state: an integer indicating the random seed (default is None)

    Returns:
    p_values: a 1D array of p-values for lenY outcomes
    reject: a boolean indicating whether the null hypothesis is rejected for each outcome
    """
    start_time = time.time()

    # Check the validity of the input parameters
    check_param(Z, X, Y, S, G, L, verbose,covariate_adjustment,alpha,alternative,random_state)

    # Set random seed
    np.random.seed(random_state)

    # preprocess the variable
    Z, X, Y, S, M = preprocess(Z, X, Y, S)
    X = transformX(X,threshholdForX,verbose)

    # choose the imputation model
    G_model = choosemodel(G)

    # impuate the missing values to get the observed test statistics in part 1
    Y_pred = getY(clone(G_model), Z, X, Y, covariate_adjustment)
    t_obs = getT(Y_pred, Z, Y.shape[1], M)
    
    if verbose:
        if isinstance(G, str):
            # the method used is :
            print("The method used is " + G)
        else:
            print("The method used is a user-defined function")
        if covariate_adjustment:
            print("Covariate adjustment is used")
        else:
            print("Covariate adjustment is not used")
        print("prediction Wilcoxon rank-sum test statistics:"+str(t_obs))
        #print wheather covariate adjustment is used
        print("=========================================================")

    # re-impute the missing values and calculate the observed test statistics in part 2
    t_sim = [ [] for _ in range(L)]
    Z_sim_templates = getZsimTemplates(Z, S)

    for l in range(L):
        
        # simulate treatment indicators
        Z_sim = getZsim(Z_sim_templates)

        # impute the missing values and get the predicted Y values        
        Y_pred = getY(clone(G_model), Z_sim, X, Y, covariate_adjustment)
        
        # get the test statistics 
        t_sim[l] = getT(Y_pred, Z_sim, Y.shape[1], M)

        if verbose:
            print(f"re-prediction iteration {l+1}/{L} completed. Test statistics[{l}]: {t_sim[l]}")

    if verbose:
        print("=========================================================")
        print("Re-impute mean t-value:"+str(np.mean(t_sim)))

    # convert t_sim to numpy array
    t_sim = np.array(t_sim)

    # perform Holm-Bonferroni correction
    p_values = []
    for i in range(Y.shape[1]):
        p_values.append(np.mean(t_sim[:,i] >= t_obs[i], axis=0))

    # perform Holm-Bonferroni correction
    reject = holm_bonferroni(p_values,alpha = alpha)

    if verbose:
        print("\nthe time used for the prediction and re-prediction framework:"+str(time.time() - start_time) + " seconds\n")
    
    return reject, p_values

