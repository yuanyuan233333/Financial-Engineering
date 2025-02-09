import sys
from typing import Dict, List
import numpy as np


def build_target_quantiles(target_alpha: List):
    """
    Build target quantiles from the list of alpha, including the median
    """
    target_quantiles = [0.5]
    for alpha in target_alpha:
        target_quantiles.append(alpha / 2)
        target_quantiles.append(1 - alpha / 2)
    target_quantiles.sort()
    return target_quantiles


def build_cp_pis(preds_cali: np.array, y_cali: np.array, preds_test: np.array,
                 settings: Dict, method: str='higher'):
    """
    Compute PIs at the different alpha levels using conformal prediction
    """
    preds_cali = np.squeeze(preds_cali, axis=-1)
    if preds_test.shape[0]>1:
        sys.exit('ERROR: exec_cup supports single test samples')
    # Compute conformity score (absolute residual)
    conf_score = np.abs(preds_cali - y_cali)
    n=conf_score.shape[0]
    # Stack the quantiles to the point pred for each alpha
    preds_test_q=[preds_test]
    for alpha in settings['target_alpha']:
        q = np.ceil((n + 1) * (1 - alpha)) / n
        Q_1_alpha= np.expand_dims(np.quantile(a=conf_score, q=q, axis=0, method=method),
                                  axis=(0,-1))
        # Append lower/upper PIs for the current alpha
        preds_test_q.append(preds_test - Q_1_alpha)
        preds_test_q.append(preds_test + Q_1_alpha)
    preds_test_q = np.concatenate(preds_test_q, axis=2)
    # Fix quantile crossing by sorting and return prediction flattened in temporal dimension (sample over pred horizon)
    return np.sort(preds_test_q.reshape(-1, preds_test_q.shape[-1]), axis=-1)

def compute_cp(recalib_preds, settings: Dict):
    """
    Reshape recalibration predictions and execute conformal prediciton for each test sample
    """
    settings['target_quantiles'] = build_target_quantiles(settings['target_alpha'])
    ens_p = recalib_preds.loc[:,0.5].to_numpy()
    ens_p_d = ens_p.reshape(-1, settings['pred_horiz'], 1)
    target_d = recalib_preds.filter([settings['task_name']], axis=1).to_numpy().reshape(-1, settings['pred_horiz'])
    num_test_samples = ens_p_d.shape[0] - settings['num_cali_samples']
    test_PIs=[]
    for t_s in range(num_test_samples):
        preds_cali = ens_p_d[t_s:settings['num_cali_samples'] + t_s]
        preds_test = ens_p_d[settings['num_cali_samples'] + t_s:settings['num_cali_samples'] + t_s+1]
        y_cali = target_d[t_s:settings['num_cali_samples'] + t_s]
        test_PIs.append(build_cp_pis(preds_cali=preds_cali,
                                     y_cali=y_cali,
                                     preds_test=preds_test,
                                     settings=settings))
    test_PIs=np.concatenate(test_PIs, axis=0)
    # Build updated dataframe
    aggr_df=recalib_preds.filter([settings['task_name']], axis=1)
    aggr_df=aggr_df.iloc[settings['pred_horiz'] * settings['num_cali_samples']:]
    for j in range(len(settings['target_quantiles'])):
        aggr_df[settings['target_quantiles'][j]]=test_PIs[:,j]
    return aggr_df