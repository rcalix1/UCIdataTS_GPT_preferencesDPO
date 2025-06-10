## Author: Ricardo A. Calix, Ph.D.
## Last update Oct 1, 2024
## Released as is with no warranty
## MIT License

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import sklearn
import random
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
## coefficient of determination 
from sklearn.metrics import r2_score

from einops import rearrange
from math import sqrt, log
torch.manual_seed(256)

import json, uuid, os

from sklearn.metrics import mean_squared_error, mean_absolute_error
import copy




############################################################

class inferenceGPT:

    def __init__(self):
        self.MyName         = 'inferenceGPT'
        self.eval_criterion = nn.MSELoss()
        self.the_offset     = None
        self.device         = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.excel_matrix   = np.zeros( (250, 30) )
        self.train_or_test  = None
        self.how_many       = 9
        ###################################################
        self.DPOannotate    = True               ## step 3
        self.DPOtrain       = True               ## step 2
        ## GPT dropout GLOBAL 0.8 OR 0.1 ???     ## step 1
        ## Noise level input in annotation       ## step 0
        ###################################################
        
        self.all_real_si_400to500 = []
        self.all_pred_si_400to500 = []

        self.all_real_si_300to500 = []
        self.all_pred_si_300to500 = []

        self.all_real_si_200to500 = []
        self.all_pred_si_200to500 = []

        self.all_real_si_100to500 = []
        self.all_pred_si_100to500 = []

        self.all_real_si_000to500 = []
        self.all_pred_si_000to500 = []


    def RSE(self, pred, true):
        return np.sqrt( np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2) )

    def MAE(self, pred, true):
        return np.mean(np.abs(pred - true))

    def MSE(self, pred, true):
        return np.mean((pred - true) ** 2)

    def RMSE(self, pred, true):
        return np.sqrt( self.MSE(pred, true) )

    def MAPE(self, pred, true):
        return np.mean(np.abs((pred - true) / true))

    def MSPE(self, pred, true):
        return np.mean(np.square((pred - true) / true))

    def metric(self, pred, true):
        mae  = self.MAE( pred, true)
        mse  = self.MSE( pred, true)
        rmse = self.RMSE(pred, true)
        mape = self.MAPE(pred, true)
        mspe = self.MSPE(pred, true)
        rse  = self.RSE( pred, true)
        return mae, mse, rmse, mape, mspe, rse  

    
    def metrics_function_all_details(self, l_pred, l_real, l_pred_all_24_features,l_real_all_24_features  ):
        print( l_pred.shape )
        print( l_real.shape )
        mse_eval_bins      = self.eval_criterion( 
                      torch.FloatTensor( l_real ),    
                      torch.FloatTensor( l_pred ) 
        )
      

        metric_mse_loss_SI_only                = mse_eval_bins.item()
        
        metric_mae_mse_rmse_mape_mspe_rse_corr = self.metric(    l_pred, l_real ) 
        
        print("mae, mse, rmse, mape, mspe, rse, corr")
        print(    metric_mae_mse_rmse_mape_mspe_rse_corr    )
        
        metric_rsquare_SI_only                 = r2_score(  l_real, l_pred )
        print( "Testing R**2 - SI only: ", metric_rsquare_SI_only  )
        
        metric_rsquare_all_features            = r2_score( 
                 np.reshape( l_real_all_24_features, (-1) ), 
                 np.reshape( l_pred_all_24_features, (-1) ) 
        ) 
        print( "Testing R**2 - All features (yes inputs): ", metric_rsquare_all_features )
        
        exclude_input_all_24_features            = r2_score( 
                 np.reshape( l_real_all_24_features[-self.how_many:, :], (-1) ), 
                 np.reshape( l_pred_all_24_features[-self.how_many:, :], (-1) ) 
        ) 
        print( "Testing R**2 - (all) - (no inputs): ", exclude_input_all_24_features  )
        
        
        
        metric_rsquare_SI_f2_features            = r2_score( 
                 np.reshape( l_real_all_24_features[:, 2], (-1) ), 
                 np.reshape( l_pred_all_24_features[:, 2], (-1) ) 
        ) 
        print( "Testing R**2 - (f2) - SI full (yes inputs): ", metric_rsquare_SI_f2_features )
        
        
        exclude_metric_rsquare_SI_f2_features            = r2_score( 
                 np.reshape( l_real_all_24_features[-self.how_many:, 2], (-1) ), 
                 np.reshape( l_pred_all_24_features[-self.how_many:, 2], (-1) ) 
        ) 
        print( "Testing R**2 - (f2) - SI full (no inputs): ", exclude_metric_rsquare_SI_f2_features )
        
        
        
        results_string = "mse_SI_only," + str(round( metric_mse_loss_SI_only, 4)) 
        results_string = results_string + "," + "rsquare_SI_only" + "," + str(round( metric_rsquare_SI_only, 4))
        results_string = results_string + "," + "rsquare_all_features" + "," + str(round( metric_rsquare_all_features, 4))
        several_metrics = str( metric_mae_mse_rmse_mape_mspe_rse_corr ).replace("(", "").replace(")","")
        results_string = results_string + "," + "mae_mse_rmse_mape_mspe_rse"  + "," + several_metrics
        print("Test MSE Loss - SI only: ",        mse_eval_bins.item()         )     ## :.4f }')
      
        print( "Testing R**2 - SI only: ", r2_score(  np.reshape( l_real, (-1) ), np.reshape( l_pred, (-1) )      )  )
        
        return results_string 
        

    def get_j( self, the_offset ):
        ## 0, 15, 30, 45, 60, 75, 90, 105
        if the_offset == 0:
            j = 0
        if the_offset == 15:
            j = 4
        if the_offset == 30:
            j = 8
        if the_offset == 45:
            j = 12
        if the_offset == 60:
            j = 16
        if the_offset == 75:
            j = 20
        if the_offset == 90:
            j = 24
        return j
        

    def add_data_to_excel_matrix(self, l_real, l_pred, yellow_l_SI_data_pred, si_2_all_real_24 ):
        for i in range(l_real.shape[0]):
            j = self.get_j( self.the_offset )
            self.excel_matrix[ self.the_offset+i, j  ] =  l_real[i].round(decimals=2)        ## np.round(l_real[i], 2)  ## deltas
            self.excel_matrix[ self.the_offset+i, j+1] =  l_pred[i].round(decimals=2)        ## np.round(l_pred[i], 2)  ## deltas
            self.excel_matrix[ self.the_offset+i, j+2] =  si_2_all_real_24[i].round(decimals=2) ## full SI
            self.excel_matrix[ self.the_offset+i, j+3] =  yellow_l_SI_data_pred[i].round(decimals=2) ## Full SI
        

    def plots_inference_one( self,   l_f2_real, l_f2_pred, pred_si  ):

        plt.axvline(x = 9,  color = 'b') 
        plt.axvline(x = 13, color = 'b') 
        x                     = [ i for i in range(   len(l_f2_real)   ) ] 
        ##l_f2_pred        = np.roll(l_f2_pred,        0)
        l_f2_real        = np.roll(l_f2_real, +1)
        a = l_f2_real[:10]
        
        b = pred_si
        squeezed_b = np.squeeze(b, axis=0)
        squeezed_b = np.squeeze(squeezed_b, axis=1)
        
        res_cat = np.concatenate((a, squeezed_b ))
    
        
        plt.plot(   x,      l_f2_real, label = "real SI",       color='red'  )   
        plt.plot(   x,      l_f2_pred, label = "f2 pred of 35",       color='black'  ) 
        plt.plot(   x,      res_cat  , label = "from SI head",  color='green'  ) 
        
       
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.legend() 
        plt.show()

  


    def GPT_get_batch_test( self, x_time_series  ):
        x  = torch.stack(   [   x_time_series[ 0 : -1    ]    ]    ) 
        y  = torch.stack(   [   x_time_series[ 1 :       ]    ]    )
        x, y = x.to(self.device), y.to(self.device)
        return x, y

                                   
    def prep_data_for_GPT_gen(self, train_data, test_CIVS, x_means, x_standard_devs ):
        if self.train_or_test:
            frames           = [ train_data[ -10: ], test_CIVS ]    ## 10 + 30
            test_CIVS_concat = pd.concat( frames )
        else:
            test_CIVS_concat = train_data[ : 50 ]
        
        test_CIVS_tr        = torch.tensor(test_CIVS_concat.values).float()
        test_CIVS_tr_scaled = ( test_CIVS_tr - x_means ) / x_standard_devs
        return test_CIVS_tr_scaled 
        

    def save_Excel_to_CSV(self):
        excel_matrix_pd = pd.DataFrame( self.excel_matrix )
        excel_matrix_pd.to_csv("for_excel_15_slide_window.csv")
        line = 'id,delta_real,delta_pred,DrZ_real,DrZ_pred,delta_real,delta_pred,DrZ_real,DrZ_pred,delta_real,delta_pred,DrZ_real,DrZ_pred,'
        line = line + 'delta_real,delta_pred,DrZ_real,DrZ_pred,delta_real,delta_pred,DrZ_real,DrZ_pred,delta_real,delta_pred,DrZ_real,DrZ_pred,'
        line = line + 'delta_real,delta_pred,DrZ_real,DrZ_pred,None,None'
        with open("for_excel_15_slide_window.csv", 'r+') as file: 
            file_data = file.read() 
            file.seek(0, 0) 
            file.write(line + '\n' + file_data)
        file.close()
        

    def get_prev_cast_plus_delta(self, l_f0_pred, xb_real_gpt, l_real_all_24_features):
        yellow_l_SI_data_pred = []
        for i in range( len(l_f0_pred) ):
            if (i-1) < 0:
                prev_cast = xb_real_gpt[0, 2] 
            else:
                prev_cast = l_real_all_24_features[i-1, 2]
            ## the_curr_val =  prev_cast + l_f0_pred[i]
            the_curr_val =  xb_real_gpt[i-1, 2]  + l_f0_pred[i]
            yellow_l_SI_data_pred.append( the_curr_val ) 
        return yellow_l_SI_data_pred
    
    def plot_r2_curves(self):
        steps = list(range(1, 10))

        r2_100 = [-0.121, 0.097, 0.116, 0.070, -0.005, -0.051, -0.114, -0.121, -0.058]
        r2_200 = [0.132, 0.245, 0.350, 0.341, 0.223, -0.028, -0.316, -0.505, -0.762]
        r2_300 = [0.304, -0.110, -0.173, -0.458, -0.662, -1.125, -1.568, -1.685, -1.631]
        r2_400 = [0.521, 0.202, 0.189, 0.187, 0.107, -0.064, -0.212, -0.217, -0.178]

        plt.plot(steps, r2_100, marker='o', label='100 Steps')
        plt.plot(steps, r2_200, marker='o', label='200 Steps')
        plt.plot(steps, r2_300, marker='o', label='300 Steps')
        plt.plot(steps, r2_400, marker='o', label='400 Steps')

        plt.title("R² Score vs. Prediction Horizon")
        plt.xlabel("Prediction Step (t+1 to t+9)")
        plt.ylabel("R² Score")
        plt.legend()
        plt.grid(True)
        plt.axhline(0, color='gray', linestyle='--')
        plt.show()
        

    def un_scale_pred_real_data(self, the_data, x_means, x_standard_devs ):
        si_mean_all_24_features         =         x_means[0, :].numpy()
        si_standard_dev_all_24_features = x_standard_devs[0, :].numpy()
        data_all_24_features  = the_data.detach().cpu().numpy().squeeze(0)
        data_all_24_features  = data_all_24_features   * si_standard_dev_all_24_features   + si_mean_all_24_features
        return data_all_24_features 
    
    def un_scale_pred_real_data_SI_head(self, the_data, x_means, x_standard_devs ):
        si_mean_all_24_features         =         x_means[0, 2].numpy()
        si_standard_dev_all_24_features = x_standard_devs[0, 2].numpy()
        data_all_24_features  = the_data  ##.detach().cpu().numpy().squeeze(0)
        data_all_24_features  = data_all_24_features   * si_standard_dev_all_24_features   + si_mean_all_24_features
        return data_all_24_features 
    
    def print_first_few_R2_individual(self,  real_si_concat, pred_si_concat, THE_FIRST_FEW ):

        first_n = THE_FIRST_FEW

        real = np.array( real_si_concat  )
        pred = np.array( pred_si_concat  )

        real_first4 = []
        pred_first4 = []

        start =  THE_FIRST_FEW - 1
        for i in range( start, len(real), 9 ):
            real_first4.append(real[i])
            pred_first4.append(pred[i])

        # Convert to numpy arrays
        real_first4 = np.array(real_first4)
        pred_first4 = np.array(pred_first4)

        # Compute R²
        r2_first4 = r2_score(real_first4, pred_first4)
        print(THE_FIRST_FEW, ".......................R² on just step n - R²:", r2_first4)
    
        self.time_series_metrics(real_first4, pred_first4)
        

    
    def print_first_few_R2(self, real_si_concat, pred_si_concat, THE_FIRST_FEW ):

        first_n = THE_FIRST_FEW

        real = np.array( real_si_concat  )
        pred = np.array( pred_si_concat  )

        real_first4 = []
        pred_first4 = []

        for i in range(0, len(real), 9):
            real_first4.extend(real[i:i+first_n])
            pred_first4.extend(pred[i:i+first_n])

        # Convert to numpy arrays
        real_first4 = np.array(real_first4)
        pred_first4 = np.array(pred_first4)

        # Compute R²
        r2_first4 = r2_score(real_first4, pred_first4)
        print(THE_FIRST_FEW, ".....................R² on first n steps - R²:", r2_first4)
    
    
        self.time_series_metrics(real_first4, pred_first4)
    
        self.print_first_few_R2_individual( real_si_concat, pred_si_concat, THE_FIRST_FEW )
    
        print('===================================================================')
    
    
    def initialize_preds_lists(self):
        
        self.all_real_si_400to500 = []
        self.all_pred_si_400to500 = []

        self.all_real_si_300to500 = []
        self.all_pred_si_300to500 = []

        self.all_real_si_200to500 = []
        self.all_pred_si_200to500 = []

        self.all_real_si_100to500 = []
        self.all_pred_si_100to500 = []

        self.all_real_si_000to500 = []
        self.all_pred_si_000to500 = []

    

    def time_series_metrics(self, y_true, y_pred):
        """
        y_true: numpy array of shape (N,)
        y_pred: numpy array of shape (N,)
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        # Ensure proper alignment
        assert y_true.shape == y_pred.shape, "Mismatched shapes"

        # Naive forecast (lag-1)
        y_naive = np.roll(y_true, 1)
        y_naive[0] = y_true[0]

        # Errors
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        # Naive R²
        ss_res = np.sum((y_true - y_pred)**2)
        ss_naive = np.sum((y_true - y_naive)**2)
        naive_r2 = 1 - ss_res / ss_naive

        # MASE (mean absolute scaled error)
        mae_naive = np.mean(np.abs(y_true[1:] - y_true[:-1])) + 1e-8  # avoid div-by-zero
        mase = np.mean(np.abs(y_true - y_pred)) / mae_naive

        metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "Naive_R2": naive_r2,
            "MASE": mase
        }

        print(f"Naive_R2:....................................R²: {naive_r2:.4f}")
        print(f"RMSE:{rmse:.4f},MAE:{mae:.4f},MASE:{mase:.4f}")
    
        ## for k, v in metrics.items():
        ##     print(f"{k}: {v:.4f}")
    

    
    
    
    def POST_Process_GPT_inference(self, pred_20_seq, xb_test, yb_test, x_means, x_standard_devs ):
        
        
        l_pred_all_24_features  = self.un_scale_pred_real_data( pred_20_seq, x_means, x_standard_devs )
        l_f2_pred               = l_pred_all_24_features[ :, 2 ]

        l_real_all_24_features  = self.un_scale_pred_real_data( yb_test, x_means, x_standard_devs )
        l_f2_real               = l_real_all_24_features[ :, 2 ]

        xb_real_gpt             = self.un_scale_pred_real_data( xb_test, x_means, x_standard_devs )
 
        results_string = self.metrics_function_all_details(  
                 l_f2_pred, 
                 l_f2_real,  
                 l_pred_all_24_features, 
                 l_real_all_24_features  
        )
        
        self.plots_inference_one( l_f2_real, l_f2_pred )
        
        return results_string
      
    
    def function_test_rc( self, train_data, test_CIVS,  model, x_means, x_standard_devs, train_or_test, how_many ):
        self.train_or_test = train_or_test
        self.how_many      = how_many
        
        
        x_test = self.prep_data_for_GPT_gen( train_data, test_CIVS, x_means, x_standard_devs )
        
        xb_test, yb_test = self.GPT_get_batch_test( x_test )   
       
        ## 10 + 30 - 1 = 39
        ## input_test_x  = xb_test[ :,  : 5 ]         ## give first 4 or 5 in sequence for GPT to generate the rest
        
        input_test_x     = xb_test[ :,  : 10 ] 
        
        pred_20_seq      = model.generate(  input_test_x,  how_many, reasoning_steps=10)
        
        results_string = self.POST_Process_GPT_inference( pred_20_seq, xb_test, yb_test, x_means, x_standard_devs )
        return results_string 
    
    


    def compare_and_label_prediction(self, yb_test, pred_a, pred_b, x_means, x_standard_devs , save_dir="preferences"):
        
        
        l_pred_a  = self.un_scale_pred_real_data( pred_a, x_means, x_standard_devs )
        l_pred_b  = self.un_scale_pred_real_data( pred_b, x_means, x_standard_devs )
        l_real_yb = self.un_scale_pred_real_data( yb_test, x_means, x_standard_devs )
        
        
        l_pred_a               = l_pred_a[  :, 2 ]
        l_pred_b               = l_pred_b[  :, 2 ]
        l_real_yb              = l_real_yb[ :, 2]
        
        os.makedirs(save_dir, exist_ok=True)
        ########################################
        
        plt.axvline(x = 9,  color = 'b') 
  
        x                     = [ i for i in range(   len(l_real_yb)   ) ] 
        ## x = list(range(len(input_seq) + len(pred_a)))
        
        ## context_line = np.array(input_seq + [None]*len(pred_a))

        ## plt.plot(x, context_line, label="context", color='gray', linestyle='--')
        ## plt.plot(x, input_seq + pred_a, label="A", color='green')
        ## plt.plot(x, input_seq + pred_b, label="B", color='blue')
        ########################################
        
        l_real_yb        = np.roll(l_real_yb, +1)

            
        plt.plot(   x,      l_real_yb, label = "real",       color='red'  )   
        plt.plot(   x,      l_pred_a,  label = "A",       color='black'  ) 
        plt.plot(   x,      l_pred_b,  label = "B",       color='gold'  ) 

        ########################################
        ## plt.axvline(x=len(input_seq)-1, color='black', linestyle='--')
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.legend()
        plt.show()

        choice = input("Which prediction is better? (A/B/skip): ").strip().lower()
        if choice in ['a', 'b']:
            data = {
                    "input": yb_test.detach().cpu().numpy().tolist(),
                    "preferred": (pred_a if choice == 'a' else pred_b).detach().cpu().numpy().tolist(),
                    "rejected": (pred_b if choice == 'a' else pred_a).detach().cpu().numpy().tolist()
            }

            with open(os.path.join(save_dir, f"{uuid.uuid4().hex}.json"), "w") as f:
                json.dump(data, f, indent=2)
            print("Saved preference.")

    
    
    def function_test_rc_42(self, train_data, test_CIVS, model, x_means, x_standard_devs,  how_many):
        
        self.how_many       = how_many
      
        frames              = [ train_data[ -10: ], test_CIVS ]    ## last 10 of train, and next 10
        test_CIVS_concat    = pd.concat( frames ) 
        test_CIVS_tr        = torch.tensor(test_CIVS_concat.values).float()
       
        test_CIVS_tr_scaled = ( test_CIVS_tr - x_means ) / x_standard_devs
      
       
        xb_test, yb_test = self.GPT_get_batch_test( test_CIVS_tr_scaled ) 
        
     
        input_test_x     = xb_test[ :,  : 10, : ] 
        
       
        pred_20_seq, generated_si    = model.generate( input_test_x, how_many, reasoning_steps=10 )
        
        
        ####################################################
        ## DPO preferences annotation
        
        if self.DPOannotate:
        
            model.train()
            
            goblin_rate = 0.1       ## 0.05
            ## input_test_x += 0.05 * torch.randn_like(input_test_x)  # or even 0.1

        
            with torch.no_grad():
                
                torch.manual_seed(0)
                input_test_x += goblin_rate * torch.randn_like(input_test_x)
                pred_a, _     = model.generate(input_test_x, how_many, reasoning_steps=10 )

                torch.manual_seed(42)
                input_test_x += goblin_rate * torch.randn_like(input_test_x)
                pred_b, _     = model.generate(input_test_x, how_many, reasoning_steps=10 )
            
            model.eval() 
            self.compare_and_label_prediction(yb_test, pred_a, pred_b, x_means, x_standard_devs )

        ####################################################
        
        pred_si = generated_si.squeeze(-1).detach().cpu().numpy()
        ## pred_si = generated_si.squeeze(-1).cpu().numpy()  # shape [B, 10]
        
        pred_si = self.un_scale_pred_real_data_SI_head( pred_si, x_means, x_standard_devs )
        
        
        l_pred_all_24_features  = self.un_scale_pred_real_data( pred_20_seq, x_means, x_standard_devs )
        l_real_all_24_features  = self.un_scale_pred_real_data( yb_test, x_means, x_standard_devs )
        
        
        l_f2_pred               = l_pred_all_24_features[ :, 2 ]
        l_f2_real               = l_real_all_24_features[ :, 2 ]
       
        
        
        
        self.plots_inference_one( l_f2_real, l_f2_pred, pred_si )
        
        
        
        exclude_input_all_24_features            = r2_score( 
                 np.reshape( l_real_all_24_features[-self.how_many:, :], (-1) ), 
                 np.reshape( l_pred_all_24_features[-self.how_many:, :], (-1) ) 
        ) 
        print( "Testing R**2 - (all) - (no inputs): ", exclude_input_all_24_features  )
        
          
        exclude_metric_rsquare_SI_f2_features            = r2_score( 
                 np.reshape( l_real_all_24_features[-self.how_many:, 2], (-1) ), 
                 np.reshape( l_pred_all_24_features[-self.how_many:, 2], (-1) ) 
        ) 
        print( "Testing R**2 - (f2) - SI full (no inputs): ", exclude_metric_rsquare_SI_f2_features )
        
        SI_head_only_metric_rsquare_features            = r2_score( 
                 np.reshape( l_real_all_24_features[-self.how_many:, 2], (-1) ), 
                 np.reshape( pred_si, (-1) ) 
        ) 
        print( "Testing R**2 - SI head only (no inputs): ", SI_head_only_metric_rsquare_features  )
         
        for i in range( l_real_all_24_features.shape[1]   ):
            per_i_metric_rsquare_features            = r2_score( 
                 np.reshape( l_real_all_24_features[-self.how_many:, i], (-1) ), 
                 np.reshape( l_pred_all_24_features[-self.how_many:, i], (-1) ) 
            ) 
            print( i, "...index R**2 (no inputs): ", per_i_metric_rsquare_features)
            
        
        return  np.reshape( l_real_all_24_features[-self.how_many:, 2], (-1) ), np.reshape( pred_si, (-1) ) 
        
        
    def print_ALL_R2s_THE_END(self):
        real_si_concat_400to500 = np.concatenate(self.all_real_si_400to500)
        pred_si_concat_400to500 = np.concatenate(self.all_pred_si_400to500)

        real_si_concat_300to500 = np.concatenate(self.all_real_si_300to500)
        pred_si_concat_300to500 = np.concatenate(self.all_pred_si_300to500)

        real_si_concat_200to500 = np.concatenate(self.all_real_si_200to500)
        pred_si_concat_200to500 = np.concatenate(self.all_pred_si_200to500)

        real_si_concat_100to500 = np.concatenate(self.all_real_si_100to500)
        pred_si_concat_100to500 = np.concatenate(self.all_pred_si_100to500)

        real_si_concat_000to500 = np.concatenate(self.all_real_si_000to500)
        pred_si_concat_000to500 = np.concatenate(self.all_pred_si_000to500)
        
        print('====================================')
        print('====================================')
        print('====================================400to500')
        print('====================================')
        self.print_first_few_R2( real_si_concat_400to500, pred_si_concat_400to500, 1 )
        self.print_first_few_R2( real_si_concat_400to500, pred_si_concat_400to500, 2 )
        self.print_first_few_R2( real_si_concat_400to500, pred_si_concat_400to500, 3 )
        self.print_first_few_R2( real_si_concat_400to500, pred_si_concat_400to500, 4 )
        self.print_first_few_R2( real_si_concat_400to500, pred_si_concat_400to500, 5 )
        self.print_first_few_R2( real_si_concat_400to500, pred_si_concat_400to500, 6 )
        self.print_first_few_R2( real_si_concat_400to500, pred_si_concat_400to500, 7 )
        self.print_first_few_R2( real_si_concat_400to500, pred_si_concat_400to500, 8 )
        self.print_first_few_R2( real_si_concat_400to500, pred_si_concat_400to500, 9 )
        
        print('====================================')
        print('====================================')
        print('====================================300to500')
        print('====================================')
        self.print_first_few_R2( real_si_concat_300to500, pred_si_concat_300to500, 1 )
        self.print_first_few_R2( real_si_concat_300to500, pred_si_concat_300to500, 2 )
        self.print_first_few_R2( real_si_concat_300to500, pred_si_concat_300to500, 3 )
        self.print_first_few_R2( real_si_concat_300to500, pred_si_concat_300to500, 4 )
        self.print_first_few_R2( real_si_concat_300to500, pred_si_concat_300to500, 5 )
        self.print_first_few_R2( real_si_concat_300to500, pred_si_concat_300to500, 6 )
        self.print_first_few_R2( real_si_concat_300to500, pred_si_concat_300to500, 7 )
        self.print_first_few_R2( real_si_concat_300to500, pred_si_concat_300to500, 8 )
        self.print_first_few_R2( real_si_concat_300to500, pred_si_concat_300to500, 9 )
        
        
        
        print('====================================')
        print('====================================')
        print('====================================200to500')
        print('====================================')
        self.print_first_few_R2( real_si_concat_200to500, pred_si_concat_200to500, 1 )
        self.print_first_few_R2( real_si_concat_200to500, pred_si_concat_200to500, 2 )
        self.print_first_few_R2( real_si_concat_200to500, pred_si_concat_200to500, 3 )
        self.print_first_few_R2( real_si_concat_200to500, pred_si_concat_200to500, 4 )
        self.print_first_few_R2( real_si_concat_200to500, pred_si_concat_200to500, 5 )
        self.print_first_few_R2( real_si_concat_200to500, pred_si_concat_200to500, 6 )
        self.print_first_few_R2( real_si_concat_200to500, pred_si_concat_200to500, 7 )
        self.print_first_few_R2( real_si_concat_200to500, pred_si_concat_200to500, 8 )
        self.print_first_few_R2( real_si_concat_200to500, pred_si_concat_200to500, 9 )
        
        print('====================================')
        print('====================================')
        print('====================================100to500')
        print('====================================')
        self.print_first_few_R2( real_si_concat_100to500, pred_si_concat_100to500, 1 )
        self.print_first_few_R2( real_si_concat_100to500, pred_si_concat_100to500, 2 )
        self.print_first_few_R2( real_si_concat_100to500, pred_si_concat_100to500, 3 )
        self.print_first_few_R2( real_si_concat_100to500, pred_si_concat_100to500, 4 )
        self.print_first_few_R2( real_si_concat_100to500, pred_si_concat_100to500, 5 )
        self.print_first_few_R2( real_si_concat_100to500, pred_si_concat_100to500, 6 )
        self.print_first_few_R2( real_si_concat_100to500, pred_si_concat_100to500, 7 )
        self.print_first_few_R2( real_si_concat_100to500, pred_si_concat_100to500, 8 )
        self.print_first_few_R2( real_si_concat_100to500, pred_si_concat_100to500, 9 )
        
        print('====================================')
        print('====================================')
        print('====================================000to500')
        print('====================================')
        self.print_first_few_R2( real_si_concat_000to500, pred_si_concat_000to500, 1 )
        self.print_first_few_R2( real_si_concat_000to500, pred_si_concat_000to500, 2 )
        self.print_first_few_R2( real_si_concat_000to500, pred_si_concat_000to500, 3 )
        self.print_first_few_R2( real_si_concat_000to500, pred_si_concat_000to500, 4 )
        self.print_first_few_R2( real_si_concat_000to500, pred_si_concat_000to500, 5 )
        self.print_first_few_R2( real_si_concat_000to500, pred_si_concat_000to500, 6 )
        self.print_first_few_R2( real_si_concat_000to500, pred_si_concat_000to500, 7 )
        self.print_first_few_R2( real_si_concat_000to500, pred_si_concat_000to500, 8 )
        self.print_first_few_R2( real_si_concat_000to500, pred_si_concat_000to500, 9 )
        
    
    def load_preferences2(self, save_dir="preferences"):
        prefs = []
        for fname in os.listdir(save_dir):
            if fname.endswith(".json"):
                with open(os.path.join(save_dir, fname), "r") as f:
                    prefs.append(json.load(f))
        return prefs

    
    def load_preferences(self, save_dir="preferences"):
        prefs = []
        bad_dir = os.path.join(save_dir, "bad_files")
        os.makedirs(bad_dir, exist_ok=True)

        for fname in os.listdir(save_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(save_dir, fname)
                try:
                    with open(fpath, "r") as f:
                        prefs.append(json.load(f))
                except Exception as e:
                    print(f"❌ Skipping bad file: {fname} — {e}")
                    os.rename(fpath, os.path.join(bad_dir, fname))
        print(f"✅ Loaded {len(prefs)} valid preference files.")
        return prefs

    
    
    def dpo_finetune_with_kl(self, model, base_model, preference_data, optimizer=None, device="cuda", epochs=3, beta=0.05):
    
        model.train()
        base_model.eval()
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(epochs):
            total_loss = 0
            for item in preference_data:
                input_tensor     = torch.tensor(item["input"],     dtype=torch.float32).unsqueeze(0).to(device)
                preferred_tensor = torch.tensor(item["preferred"], dtype=torch.float32).unsqueeze(0).to(device)
                rejected_tensor  = torch.tensor(item["rejected"],  dtype=torch.float32).unsqueeze(0).to(device)
            
                input_tensor_19     = input_tensor.squeeze(0)
                preferred_tensor    = preferred_tensor.squeeze(0)
                rejected_tensor     = rejected_tensor.squeeze(0) 
            
                input_tensor = input_tensor_19[:, :10, :]
            
                ## print("prefs shapes")
                ## print(    input_tensor.shape)
                ## print(preferred_tensor.shape)
                ## print( rejected_tensor.shape)

                # Log probs from new model
            
                pred_full, _     = model.generate(input_tensor, max_new_tokens=9, reasoning_steps=10)
                pred_new         = pred_full[:, -9:, :]  # [1, 9, 35]
            
                logp_pref_new = -F.mse_loss(pred_new, preferred_tensor[:, -9:, :], reduction='sum')
                logp_rej_new  = -F.mse_loss(pred_new,  rejected_tensor[:, -9:, :], reduction='sum')
            

                # Log probs from base model (for KL)
                with torch.no_grad():
                    pred_old_full, _     = base_model.generate(input_tensor, max_new_tokens=9, reasoning_steps=10)
                    pred_old             = pred_old_full[:, -9:, :]  # [1, 9, 35]
                
                    logp_pref_old = -F.mse_loss(pred_old , preferred_tensor[:, -9:, :], reduction='sum')
                    logp_rej_old  = -F.mse_loss(pred_old ,  rejected_tensor[:, -9:, :],  reduction='sum')

                # DPO loss
                ## print( torch.stack([logp_pref_new, logp_rej_new]).shape )
          
                logits   = torch.stack([logp_pref_new, logp_rej_new]).unsqueeze(0)
                labels   = torch.tensor([0], device=device)
                dpo_loss = F.cross_entropy(logits, labels)

                # KL regularization
                kl_pref = logp_pref_new - logp_pref_old
                kl_rej  = logp_rej_new  - logp_rej_old
                kl_term = (kl_pref + kl_rej) / 2

                # Final combined loss
                beta = 0.001    ## 0.05    ## 0.01 or 0.001
                total_dpo_loss = dpo_loss + beta * kl_term

                optimizer.zero_grad()
                total_dpo_loss.backward()
                optimizer.step()
                total_loss += total_dpo_loss.item()

            print(f"Epoch {epoch+1}: Total DPO Loss (with KL) = {total_loss:.4f}")

        return model


    def DPO_RLHF(self, si_GPT):
        
        prefs       = self.load_preferences()

        base_model = copy.deepcopy(si_GPT)  # Clone the model before training
        base_model.eval()                   # Important: don't let it train
        
        model_DPO = self.dpo_finetune_with_kl(model=si_GPT,
                                              base_model=base_model,  # clone if needed
                                              preference_data=prefs,
                                              device="cuda",  # or "cpu"
                                              epochs=5,
                                              beta=0.05
        )
        return model_DPO
        
    
    def printName(self):
        print( self.MyName  )










