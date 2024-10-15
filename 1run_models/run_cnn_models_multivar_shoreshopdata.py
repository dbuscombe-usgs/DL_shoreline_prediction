
from utils.splits import normal_data, split_sequences
from utils.loss_errors import index_mielke, mielke_loss

import os
import random
import numpy as np
import pandas as pd
import math
import scipy.stats

from datetime import  timedelta
from sklearn.metrics import mean_squared_error
import datetime as dt

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D

import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error

abs_path= os.path.abspath(os.getcwd())
os.chdir(abs_path)


##Import Observed Shoreline positions
Y_in= pd.read_csv('../data/shoreshop2_shorelines_obs.csv',index_col=False) 
Y_time = Y_in['Datetime']
Y_in.reset_index(drop=True, inplace=True)


yobs = Y_in.drop(columns=['Datetime']).mean(axis=1)
y_tlinux = (pd.to_datetime(Y_time)- dt.datetime(1970,1,1)).dt.total_seconds()



H_in= pd.read_csv('../data/shoreshop2_Hs.csv')#$,index_col=False) 
H_in['Datetime'] = pd.to_datetime(H_in['Datetime'])  
# shoreline['Datetime'] = pd.to_datetime(shoreline['Datetime'])  

## clip waves to just shoreline obs time period
mask = (H_in['Datetime'] > '1999-02-16') & (H_in['Datetime'] <= '2018-12-30')
H_in = H_in.loc[mask]

hobs = H_in.drop(columns=['Datetime']).mean(axis=1)
h_tlinux = (pd.to_datetime(H_in['Datetime'])- dt.datetime(1970,1,1)).dt.total_seconds()


yobs_i = np.interp(h_tlinux,y_tlinux, yobs)





T_in= pd.read_csv('../data/shoreshop2_Tp.csv')#$,index_col=False) 
T_in['Datetime'] = pd.to_datetime(T_in['Datetime'])  
# shoreline['Datetime'] = pd.to_datetime(shoreline['Datetime'])  

## clip waves to just shoreline obs time period
mask = (T_in['Datetime'] > '1999-02-16') & (T_in['Datetime'] <= '2018-12-30')
T_in = T_in.loc[mask]

tobs = T_in.drop(columns=['Datetime']).mean(axis=1)





D_in= pd.read_csv('../data/shoreshop2_Dir.csv')#$,index_col=False) 
D_in['Datetime'] = pd.to_datetime(D_in['Datetime'])  
# shoreline['Datetime'] = pd.to_datetime(shoreline['Datetime'])  

## clip waves to just shoreline obs time period
mask = (D_in['Datetime'] > '1999-02-16') & (D_in['Datetime'] <= '2018-12-30')
D_in = D_in.loc[mask]

dobs = D_in.drop(columns=['Datetime']).mean(axis=1)






mat_in = pd.DataFrame.from_dict({'Datetime': H_in['Datetime'].values, 'Hs': hobs, 'Dir': dobs, 'Tp': tobs, 'yout': yobs_i })
mat_in = mat_in.interpolate()


mat_in['Datetime'] = pd.to_datetime(mat_in['Datetime'])
mat_in = mat_in.set_index(['Datetime'])
mat_out= pd.DataFrame(mat_in.yout.values, index=mat_in.index)
mat_out = mat_out.interpolate()



###################TRAIN, DEV AND TEST SPLITS################################
#Manually split by date 
#Remember: Tairua the forecast is from July 2014 (2014-07-01), 

date_forecast= '2014-01-01'
#Train until  2 years before the test set
train_date=pd.to_datetime(date_forecast) - timedelta(days=365*2)
train_date= str(train_date.strftime("%Y-%m-%d"))

###########################DATA NORMALIZATION################################
_, mat_in = normal_data(mat_in,train_date)
scaler, mat_out_norm = normal_data(mat_out,train_date)


###################TRAIN, DEV AND TEST SPLITS################################
#Manually split by date 
train = mat_in[mat_in.index[0]:train_date].values.astype('float32')




#Development set (2 years before the test set)
devinit_date=pd.to_datetime(train_date) + timedelta(days=1)
devinit_date= str(devinit_date.strftime("%Y-%m-%d"))
dev_date=pd.to_datetime(date_forecast) - timedelta(days=1)
dev_date= str(dev_date.strftime("%Y-%m-%d"))
dev= mat_in[devinit_date:dev_date].values.astype('float32')
#Test set, depends on study site
test = mat_in[date_forecast:mat_in.index[-1]].values.astype('float32')


#############################################################################
#From pandas to array, HERE WE SEPARATE THE INPUTS FROM THE Y_OUTPUT
# split a multivariate sequence into samples 
n_steps_in, n_steps_out =40,1
train_x, train_y = split_sequences(train, n_steps_in, n_steps_out)
dev_x, dev_y = split_sequences(dev, n_steps_in, n_steps_out)
test_x, test_y = split_sequences(test, n_steps_in, n_steps_out)
# # the dataset knows the number of features, e.g. 2
n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[
    2], train_y.shape[1]



####################Define Loss functions: NEURAL NETWORK##################
def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


#CNN model
loss= mielke_loss
min_delta= 0.001
def cnn_custom(train_x, train_y, dev_x, dev_y, cfg, second_ccn_bank, linear_activation):
    print("--------------------------------")
    print("Model:", cfg)
    set_seed(33)
    # define model    # create configs
    n_filters, n_kernels, n_drop, n_epochs,n_batch = cfg    
    model = Sequential()
    model.add(Conv1D(filters=n_filters, kernel_size=n_kernels,
                     activation='relu', input_shape=(n_steps_in, n_features)))
    

    if second_ccn_bank is True:
        model.add(Conv1D(filters=n_filters, kernel_size=n_kernels,
                     activation='relu'))
    else:
        print("using smaller cnn model")

    model.add(MaxPooling1D(pool_size=2)) 
    model.add(Flatten()) 
    if linear_activation is True:
        model.add(Dense(100, activation='linear')) 
    else:
        model.add(Dense(100, activation='relu')) 
    model.add(Dropout(n_drop))       
    model.add(Dense(1))    
    
    model.compile(optimizer='adam', loss=loss)
    # fit model
    es = EarlyStopping(patience=10, verbose=0, min_delta=min_delta,
                       monitor='val_loss', mode='auto',
                       restore_best_weights=True)
    history= model.fit(train_x, train_y, validation_data=(dev_x, dev_y),
                         batch_size=n_batch, epochs=n_epochs, verbose=2,
                         callbacks=[es])  
    return model, history




#############################################################################
# Load Grid Search Hyperparameters
scores = list()   
cfg_list=pd.read_csv('./hyp/10best_hyp_Wavesonly_Mielke_CNN.csv')
cfg_list= cfg_list[["f","k","D","e","b"]]
cfg_list= cfg_list.values.tolist()
for i in range(len(cfg_list)):     
    for element in range(len(cfg_list[i])):
        if element != 2:
            cfg_list[i][element] = int(cfg_list[i][element])   

## set to 200 epochs - why not? early stopping is used anyway
for k in range(len(cfg_list)):
    cfg_list[k][3]=200


second_ccn_bank = False 
linear_activation = True
#%Run model configurations in loop###########################################
#Predefine empty dataframe
plot_date = pd.to_datetime(date_forecast) + timedelta(days=n_steps_in-1)
plot_date= str(plot_date.strftime("%Y-%m-%d"))
yresults= pd.DataFrame(index=mat_in[ plot_date :mat_in.index[-1]].index,
                       columns=['ann1','ann2','ann3','ann4','ann5',
                                'ann6','ann7','ann8','ann9','ann10'])
#Rescale target shoreline time series
testY = scaler.inverse_transform(test_y)
for (index, colname) in enumerate(yresults):
    print('Model number:' + str(index))
    #Train model with hyp config from config list
    model,_ = cnn_custom(train_x, train_y, dev_x, dev_y, cfg_list[index], second_ccn_bank, linear_activation) 
    testdl = model.predict(test_x)     
    yresults.iloc[:,index]= scaler.inverse_transform(testdl)
    print('Metrics')
    print('RMSE:' )
    print(str(math.sqrt(mean_squared_error(yresults.iloc[:,index].values,
                                           testY))))
    print('Pearson:' )
    print(str(scipy.stats.pearsonr(yresults.iloc[:,index].values,
                                   testY[:,0])[0]))
    print('Mielke:' )    
    print(str(index_mielke(yresults.iloc[:,index].values,testY[:,0])))

yresults["obs"]=testY[:,0]


model.summary()
###130,737




yresults.to_csv('./output/CNN_ensemble_multivar_smallcnn_linearact_shoreshop2.csv')


## Metrics 
ymetrics= pd.DataFrame(index=np.arange(11),
                       columns=['rmse_arr','pear_arr','mielke_arr'])

ymetrics['rmse_arr']=np.array([math.sqrt(mean_squared_error(yresults[colname].values,testY)) for (index, colname) in enumerate(yresults)])
ymetrics['pear_arr']=np.array([scipy.stats.pearsonr(yresults[colname].values,testY[:,0])[0] for (index, colname) in enumerate(yresults)])
ymetrics['mielke_arr']=np.array( [index_mielke(yresults[colname].values,testY[:,0]) for (index, colname) in enumerate(yresults)])

ymetrics.to_csv('./output/CNN_ensemble_multivar_smallcnn_linearact_metrics_shoreshop2.csv')





#################################

second_ccn_bank = False 
linear_activation = False
#%Run model configurations in loop###########################################
#Predefine empty dataframe
plot_date = pd.to_datetime(date_forecast) + timedelta(days=n_steps_in-1)
plot_date= str(plot_date.strftime("%Y-%m-%d"))
yresults= pd.DataFrame(index=mat_in[ plot_date :mat_in.index[-1]].index,
                       columns=['ann1','ann2','ann3','ann4','ann5',
                                'ann6','ann7','ann8','ann9','ann10'])
#Rescale target shoreline time series
testY = scaler.inverse_transform(test_y)
for (index, colname) in enumerate(yresults):
    print('Model number:' + str(index))
    #Train model with hyp config from config list
    model,_ = cnn_custom(train_x, train_y, dev_x, dev_y, cfg_list[index], second_ccn_bank, linear_activation) 
    testdl = model.predict(test_x)     
    yresults.iloc[:,index]= scaler.inverse_transform(testdl)
    print('Metrics')
    print('RMSE:' )
    print(str(math.sqrt(mean_squared_error(yresults.iloc[:,index].values,
                                           testY))))
    print('Pearson:' )
    print(str(scipy.stats.pearsonr(yresults.iloc[:,index].values,
                                   testY[:,0])[0]))
    print('Mielke:' )    
    print(str(index_mielke(yresults.iloc[:,index].values,testY[:,0])))


##EXPORT ENSEMBLE
yresults["obs"]=testY[:,0]
yresults.to_csv('./output/CNN_ensemble_multivar_smallcnn_shoreshop2.csv')


## Metrics 
ymetrics= pd.DataFrame(index=np.arange(11),
                       columns=['rmse_arr','pear_arr','mielke_arr'])

ymetrics['rmse_arr']=np.array([math.sqrt(mean_squared_error(yresults[colname].values,testY)) for (index, colname) in enumerate(yresults)])
ymetrics['pear_arr']=np.array([scipy.stats.pearsonr(yresults[colname].values,testY[:,0])[0] for (index, colname) in enumerate(yresults)])
ymetrics['mielke_arr']=np.array( [index_mielke(yresults[colname].values,testY[:,0]) for (index, colname) in enumerate(yresults)])

ymetrics.to_csv('./output/CNN_ensemble_multivar_smallcnn_metrics_shoreshop2.csv')


model.summary()
###


#################################

second_ccn_bank = True 
linear_activation = False
#%Run model configurations in loop###########################################
#Predefine empty dataframe
plot_date = pd.to_datetime(date_forecast) + timedelta(days=n_steps_in-1)
plot_date= str(plot_date.strftime("%Y-%m-%d"))
yresults= pd.DataFrame(index=mat_in[ plot_date :mat_in.index[-1]].index,
                       columns=['ann1','ann2','ann3','ann4','ann5',
                                'ann6','ann7','ann8','ann9','ann10'])
#Rescale target shoreline time series
testY = scaler.inverse_transform(test_y)
for (index, colname) in enumerate(yresults):
    print('Model number:' + str(index))
    #Train model with hyp config from config list
    model,_ = cnn_custom(train_x, train_y, dev_x, dev_y, cfg_list[index], second_ccn_bank, linear_activation) 
    testdl = model.predict(test_x)     
    yresults.iloc[:,index]= scaler.inverse_transform(testdl)
    print('Metrics')
    print('RMSE:' )
    print(str(math.sqrt(mean_squared_error(yresults.iloc[:,index].values,
                                           testY))))
    print('Pearson:' )
    print(str(scipy.stats.pearsonr(yresults.iloc[:,index].values,
                                   testY[:,0])[0]))
    print('Mielke:' )    
    print(str(index_mielke(yresults.iloc[:,index].values,testY[:,0])))


##EXPORT ENSEMBLE
#Cut spads and shorefor to match the DL test time series output
yresults["obs"]=testY[:,0]
yresults.to_csv('./output/CNN_ensemble_multivar_shoreshop2.csv')

## Metrics 
ymetrics= pd.DataFrame(index=np.arange(11),
                       columns=['rmse_arr','pear_arr','mielke_arr'])

ymetrics['rmse_arr']=np.array([math.sqrt(mean_squared_error(yresults[colname].values,testY)) for (index, colname) in enumerate(yresults)])
ymetrics['pear_arr']=np.array([scipy.stats.pearsonr(yresults[colname].values,testY[:,0])[0] for (index, colname) in enumerate(yresults)])
ymetrics['mielke_arr']=np.array( [index_mielke(yresults[colname].values,testY[:,0]) for (index, colname) in enumerate(yresults)])

ymetrics.to_csv('./output/CNN_ensemble_multivar_shoreshop2.csv')

##3645




#######################################################################

## set all filters to 2
for k in range(len(cfg_list)):
    cfg_list[k][0]=2


second_ccn_bank = False 
linear_activation = True
#%Run model configurations in loop###########################################
#Predefine empty dataframe
plot_date = pd.to_datetime(date_forecast) + timedelta(days=n_steps_in-1)
plot_date= str(plot_date.strftime("%Y-%m-%d"))
yresults= pd.DataFrame(index=mat_in[ plot_date :mat_in.index[-1]].index,
                       columns=['ann1','ann2','ann3','ann4','ann5',
                                'ann6','ann7','ann8','ann9','ann10'])


#Rescale target shoreline time series
testY = scaler.inverse_transform(test_y)
for (index, colname) in enumerate(yresults):
    print('Model number:' + str(index))
    #Train model with hyp config from config list
    model,_ = cnn_custom(train_x, train_y, dev_x, dev_y, cfg_list[index], second_ccn_bank, linear_activation) 
    testdl = model.predict(test_x)     
    yresults.iloc[:,index]= scaler.inverse_transform(testdl)
    print('Metrics')
    print('RMSE:' )
    print(str(math.sqrt(mean_squared_error(yresults.iloc[:,index].values,
                                           testY))))
    print('Pearson:' )
    print(str(scipy.stats.pearsonr(yresults.iloc[:,index].values,
                                   testY[:,0])[0]))
    print('Mielke:' )    
    print(str(index_mielke(yresults.iloc[:,index].values,testY[:,0])))


##EXPORT ENSEMBLE
#Cut spads and shorefor to match the DL test time series output
# yresults["spads"]=mspads[plot_date :mspads.index[-1]]
# yresults["shorefor"]=mshorefor[plot_date :mshorefor.index[-1]]
yresults["obs"]=testY[:,0]
yresults.to_csv('./output/CNN_ensemble_multivar_tinycnn_linearact_shoreshop2.csv')

## Metrics 
ymetrics= pd.DataFrame(index=np.arange(11),
                       columns=['rmse_arr','pear_arr','mielke_arr'])

ymetrics['rmse_arr']=np.array([math.sqrt(mean_squared_error(yresults[colname].values,testY)) for (index, colname) in enumerate(yresults)])
ymetrics['pear_arr']=np.array([scipy.stats.pearsonr(yresults[colname].values,testY[:,0])[0] for (index, colname) in enumerate(yresults)])
ymetrics['mielke_arr']=np.array( [index_mielke(yresults[colname].values,testY[:,0]) for (index, colname) in enumerate(yresults)])

ymetrics.to_csv('./output/CNN_ensemble_multivar_tinycnn_linearact_metrics_shoreshop2.csv')


