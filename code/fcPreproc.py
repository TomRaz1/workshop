# MIT License

# Copyright (c) 2021 Arthur Spencer

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
Preprocessing script for dFC analysis. Runs sliding-window correlations (SWC), 
calculating functional connectivity by estimating covariance from the precision 
matrix, regularised with the L1-norm (using the inverse_covariance package).
'''

import os
import numpy as np
import multiprocessing as proc
from scipy import stats
from scipy import linalg
from sklearn.covariance import GraphicalLassoCV
from sklearn.covariance import GraphicalLasso

def get_lambda(windowed_func):
    return 0.01


def get_fc(func, lambda_):
    model = GraphicalLasso(alpha=lambda_, max_iter=50, tol=1e-2)
    model.fit(func.T)

    cov = np.array(model.covariance_)
    D = np.sqrt(np.diag(np.diag(cov)))
    DInv = np.linalg.inv(D)
    fc = np.matmul(DInv, np.matmul(cov, DInv))
    np.fill_diagonal(fc, 0)

    return np.arctanh(fc)



def get_dfc(in_):
    '''
    Run SWC for a single subject, with limited number of windows for fast testing/debug.
    '''
    func = in_[0]
    window_size = in_[1]
    window_shape = in_[2]
    n_nodes = in_[3]
    step = in_[4]

    n_nodes = len(func[:, 0])

    if window_shape == 'rectangle':
        window = np.ones(window_size)
    elif window_shape == 'hamming':
        window = np.hamming(window_size)
    elif window_shape == 'hanning':
        window = np.hanning(window_size)
    else:
        raise Exception('%s window shape not recognised. Choose rectangle, hamming or hanning.' % window_shape)

    inds = range(0, len(func[0]) - window_size, step)
    nwindows = len(inds)
    
    print(f"[INFO] Number of windows for subject: {nwindows}")  

    max_windows = nwindows  # no window limitation
    print(f"[DEBUG] Total windows: {nwindows}, using max {max_windows} for speed") 

    dfc = np.zeros([max_windows, n_nodes, n_nodes])
    windowed_func = np.zeros([max_windows, n_nodes, window_size])

    for i in range(max_windows):
        this_sec = func[:, inds[i]:inds[i] + window_size]
        windowed_func[i] = this_sec * window

    lambda_ = 0.1
    print(f"[DEBUG] Using fixed lambda: {lambda_}")

    for i in range(max_windows):
        try:
            dfc[i, :, :] = get_fc(windowed_func[i], lambda_)
        except FloatingPointError as e:
            print(f"[WARNING] Window {i}: GraphicalLasso failed ({e}). Filling zeros.")
            dfc[i, :, :] = np.zeros((n_nodes, n_nodes))
        except Exception as e:
            print(f"[ERROR] Window {i}: Unexpected error in get_fc ({e}). Filling zeros.")
            dfc[i, :, :] = np.zeros((n_nodes, n_nodes))

    print("[DEBUG] Finished computing DFC")
    return dfc




'''def load_data(func_path, zscore=True, hcp=False):
    
    Load raw timeseries data.
    
    files = os.listdir(func_path)

    if hcp:
        files = sorted([file for file in files if file.endswith('.txt')])
        subjs = np.array([stats.zscore(np.loadtxt('%s/%s' % (func_path, file)).T, axis=1) for file in files])
    else:
        #files = sorted([file for file in files if file.endswith('.csv')])
        files = sorted([file for file in files if file.endswith('.npy')]) #addaptation to 246_data
        if zscore:
            subjs = np.array([stats.zscore(np.loadtxt('%s/%s' % (func_path, file), delimiter=','), axis=1) for file in files])
        else:
            subjs = np.array([np.loadtxt('%s/%s' % (func_path, file), delimiter=',') for file in files])

    return subjs
    '''

def load_data(func_path, zscore=True, hcp=False):
    '''
    Load raw timeseries data from .csv or .npy files
    '''
    print("Loading from path:", func_path)

    files = sorted([file for file in os.listdir(func_path) if file.endswith('.csv') or file.endswith('.npy')])
    subjs = []

    for file in files:
        full_path = os.path.join(func_path, file)
        if file.endswith('.csv'):
            data = np.loadtxt(full_path, delimiter=',')
        elif file.endswith('.npy'):
            data = np.load(full_path)
        else:
            continue

        if zscore:
            data = stats.zscore(data, axis=1)

        subjs.append(data)

    return np.array(subjs)

