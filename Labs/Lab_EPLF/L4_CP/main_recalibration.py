"""
Main script to run the recalibration experiments
"""
# Author: Alessandro Brusaferri
# License: Apache-2.0 license

import os
import pandas as pd
import numpy as np
os.environ["TF_USE_LEGACY_KERAS"]="1"
from tools.PrTSF_Recalib_tools import PrTsfRecalibEngine, load_data_model_configs
from tools.prediction_quantiles_tools import plot_quantiles
from tools.conformal_prediction import compute_cp

#--------------------------------------------------------------------------------------------------------------------
def compute_pinball_scores(y_true, pred_quantiles, quantiles_levels):
    """
    Utility function to compute the pinball score on the test results
    return: pinball scores computed for each quantile level and each step in the pred horizon
    """
    score = []
    for i, q in enumerate(quantiles_levels):
        error = np.subtract(y_true, pred_quantiles[:, :, i])
        loss_q = np.maximum(q * error, (q - 1) * error)
        score.append(np.expand_dims(loss_q,-1))
    score = np.mean(np.concatenate(score, axis=-1), axis=0)
    return score

#--------------------------------------------------------------------------------------------------------------------
# Set PEPF task to execute
PF_task_name = 'NetLoad'
# Set Model setup to execute
exper_setup = 'point-DNN'

#---------------------------------------------------------------------------------------------------------------------
# Set run configs
run_id = 'recalib_opt_grid_1_1'
# Load hyperparams from file (select: load_tuned or optuna_tuner)
hyper_mode = 'load_tuned'
# Plot train history flag
plot_train_history=False
plot_weights=False

#---------------------------------------------------------------------------------------------------------------------
# Load experiments configuration from json file
configs=load_data_model_configs(task_name=PF_task_name, exper_setup=exper_setup, run_id=run_id)

# Load dataset
dir_path = os.getcwd()
ds = pd.read_csv(os.path.join(dir_path, 'data', 'datasets', configs['data_config'].dataset_name))
ds.set_index(ds.columns[0], inplace=True)

#---------------------------------------------------------------------------------------------------------------------
# Instantiate recalibratione engine
PrTSF_eng = PrTsfRecalibEngine(dataset=ds,
                               data_configs=configs['data_config'],
                               model_configs=configs['model_config'])

# Get model hyperparameters (previously saved or by tuning)
model_hyperparams = PrTSF_eng.get_model_hyperparams(method=hyper_mode, optuna_m=configs['model_config']['optuna_m'])

# Exec recalib loop over the test_set samples, using the tuned hyperparams
test_predictions = PrTSF_eng.run_recalibration(model_hyperparams=model_hyperparams,
                                               plot_history=plot_train_history,
                                               plot_weights=plot_weights)

#--------------------------------------------------------------------------------------------------------------------
# Conformal prediction settings
exec_CP = True
# set the size of the calibration set sufficiently large to cover the target alpha (tails)
#cp_settings={'target_alpha':[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]}
#num_cali_samples = 122
cp_settings={'target_alpha':[0.10]}
num_cali_samples = 31

if exec_CP:
    if exper_setup[:5]=='point':
        # build the settings to build PF from point using CP
        cp_settings['pred_horiz']=configs['data_config'].pred_horiz
        cp_settings['task_name']=configs['data_config'].task_name
        cp_settings['num_cali_samples']=num_cali_samples
        # exec conformal prediction
        test_predictions = compute_cp(test_predictions,cp_settings)
    else:
        print('conformal prediction implemented on point predictions')

#--------------------------------------------------------------------------------------------------------------------
# Plot test predictions
plot_quantiles(test_predictions, target=PF_task_name)

#--------------------------------------------------------------------------------------------------------------------
# Compute pinball score
quantiles_levels = PrTSF_eng.model_configs['target_quantiles']
pred_steps = configs['model_config']['pred_horiz']

pinball_scores = compute_pinball_scores(y_true=test_predictions[PF_task_name].to_numpy().reshape(-1,pred_steps),
                                        pred_quantiles=test_predictions.loc[:,test_predictions.columns != PF_task_name].
                                        to_numpy().reshape(-1, pred_steps, len(quantiles_levels)),
                                        quantiles_levels=quantiles_levels)

#--------------------------------------------------------------------------------------------------------------------
print('Done!')
