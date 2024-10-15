


from utils.splits import normal_data, split_sequences
from utils.loss_errors import index_mielke, mielke_loss

import os
import random
import numpy as np
import pandas as pd
# import math
import scipy.stats

from datetime import  timedelta
from sklearn.metrics import root_mean_squared_error

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D

# from keras.utils import plot_model
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.layer_utils import count_params
import matplotlib.dates as mdates
import matplotlib
import datetime as dt


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
obs_shoreline = mat_in[['Datetime', 'yout']]


mat_in['Datetime'] = pd.to_datetime(mat_in['Datetime'])
mat_in = mat_in.set_index(['Datetime'])
mat_out= pd.DataFrame(mat_in.yout.values, index=mat_in.index)
mat_out = mat_out.interpolate()



## just waves
Hs = mat_in['Hs'].values

Hbar = np.mean(Hs)
## waver energy anomaly
f=(Hs**2-Hbar**2)/Hbar**2 

mat_in = mat_in.assign(f=f)
mat_in = mat_in[['f','Dir','Tp','yout']]


#dy instead of y
# mat_in['yout'] = np.gradient(mat_in.yout.values)




###################TRAIN, DEV AND TEST SPLITS################################
### first create an 'evaluate' seq which is for final model plotting and for input scaling 
date_evaluate= '1999-02-17'
eval_date=pd.to_datetime(date_evaluate)

_, mat_eval = normal_data(mat_in,eval_date)
scaler_eval, mateval_out_norm = normal_data(mat_out,eval_date)

eval = mat_eval.values.astype('float32')

n_steps_in, n_steps_out =40,1
eval_x, eval_y = split_sequences(eval, n_steps_in, n_steps_out)

evalY = scaler_eval.inverse_transform(eval_y)



#Remember: Tairua the forecast is from July 2014 (2014-07-01), previous data 
date_forecast= '2014-01-01'
#Train until  2 years before the test set
train_date=pd.to_datetime(date_forecast) - timedelta(days=365*2)
train_date= str(train_date.strftime("%Y-%m-%d"))

###########################DATA NORMALIZATION################################
_, mat_in = normal_data(mat_in,train_date)
scaler, mat_out_norm = normal_data(mat_out,train_date)

# scaler = scaler_eval

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
def cnn_custom(train_x, train_y, dev_x, dev_y, cfg, second_ccn_bank, linear_activation, num_neurons):
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
        model.add(Dense(num_neurons, activation='linear')) 
    else:
        model.add(Dense(num_neurons, activation='relu')) 
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




num_neurons = 1

#### a range of hyperparams, only varying kernel size
##n_filters, n_kernels, n_drop, n_epochs,n_batch 
cfg_list = [
 [2, 2, 0.0, 200, 8],
 [2, 4, 0.0, 200, 8],
 [2, 6, 0.0, 200, 8],
 [2, 8, 0.0, 200, 8],
 [2, 10, 0.0, 200, 8],
 [2, 12, 0.0, 200, 8]
]

second_ccn_bank = False 
linear_activation = True


#%Run model configurations in loop###########################################
#Predefine empty dataframe
plot_date = pd.to_datetime(date_forecast) + timedelta(days=n_steps_in-1)
plot_date= str(plot_date.strftime("%Y-%m-%d"))
yresults= pd.DataFrame(index=mat_in[ plot_date :mat_in.index[-1]].index,
                    columns=['ann1','ann2','ann3','ann4','ann5',
                                'ann6'])

#Rescale target shoreline time series
testY = scaler.inverse_transform(test_y)


MODELS=[]
HISTORY=[]
PARAMS=[]

for (index, colname) in enumerate(yresults):
    print('Model number:' + str(index))
    #Train model with hyp config from config list
    model,history = cnn_custom(train_x, train_y, dev_x, dev_y, cfg_list[index], second_ccn_bank, linear_activation, num_neurons) 
    # print(model.summary())

    MODELS.append(model)
    HISTORY.append(history)

    trainable_count = count_params(model.trainable_weights)
    batch = cfg_list[index][-1]
    kernel = cfg_list[index][1]
    PARAMS.append(trainable_count)

    # plot_model(model,to_file=f'microcnn{index}.png')

    testdl = model.predict(test_x)     
    yresults.iloc[:,index]= scaler.inverse_transform(testdl)
    print('Metrics')
    print('RMSE:' )
    print(str(root_mean_squared_error(yresults.iloc[:,index].values,
                                        testY)))
    print('Pearson:' )
    print(str(scipy.stats.pearsonr(yresults.iloc[:,index].values,
                                testY[:,0])[0]))
    print('Mielke:' )    
    print(str(index_mielke(yresults.iloc[:,index].values,testY[:,0])))

    ##rename columns something useful
    yresults = yresults.rename(columns={yresults.columns[index]:'p'+str(trainable_count)+'b'+str(batch)+'k'+str(kernel)})

    print(yresults.head())

    ## use the model as afeature extractor to examine correlation between features and outputs

    extract = Model(model.inputs, model.layers[-3].output)
    features = extract.predict(train_x)


    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(mat_in['f'], obs_shoreline['yout'], '.')
    xc = np.min(np.corrcoef(mat_in['f'], obs_shoreline['yout']))
    plt.text(10,170,'R='+str(xc)[:6])
    plt.xlabel('Wave height anomaly (m)')
    plt.ylabel('Shoreline position (m)')

    plt.subplot(122)
    plt.plot(features[:,0], scaler.inverse_transform(train_y), '.')
    xc = np.min(np.corrcoef(features[:,0], scaler.inverse_transform(train_y).squeeze()))
    plt.text(10,170,'R='+str(xc)[:6])
    plt.xlabel('CNN feature vector 1 (-)')
    plt.ylabel('Shoreline position (m)')
    # plt.show()
    plt.savefig(f'./output/CNN_multivar_shoreshop_microcnn_nn{num_neurons}_p'+str(trainable_count)+'b'+str(batch)+'k'+str(kernel)+'_linearact', dpi=200, bbox_inches='tight')
    plt.close()




##EXPORT ENSEMBLE
yresults["obs"]=testY[:,0]
yresults.to_csv(f'./output/CNN_multivar_shoreshop_microcnn_nn{num_neurons}_p{trainable_count}_linearact.csv')

## Metrics 
ymetrics= pd.DataFrame(index=np.arange(7),
                    columns=['rmse_arr','pear_arr','mielke_arr'])

ymetrics['rmse_arr']=np.array([root_mean_squared_error(yresults[colname].values,testY) for (index, colname) in enumerate(yresults)])
ymetrics['pear_arr']=np.array([scipy.stats.pearsonr(yresults[colname].values,testY[:,0])[0] for (index, colname) in enumerate(yresults)])
ymetrics['mielke_arr']=np.array( [index_mielke(yresults[colname].values,testY[:,0]) for (index, colname) in enumerate(yresults)])

ymetrics.to_csv(f'./output/CNN_multivar_shoreshop_microcnn__nn{num_neurons}_p{trainable_count}_linearact_metrics.csv')


best_model = MODELS[np.argmin(ymetrics.rmse_arr.values[:-1])]

traindl = best_model.predict(train_x)     
traindl = scaler.inverse_transform(traindl)
trainY = scaler.inverse_transform(train_y)

testdl = best_model.predict(test_x)     
testdl = scaler.inverse_transform(testdl)

### output plot

plt.figure(figsize=(16,8))

plt.subplot(211)
# plt.plot(yresults['obs'], yresults['p51b8k6'], 'ko', label='test')
plt.plot(trainY, traindl, 'b.', label='train set')
plt.plot(testY, testdl, 'ko', label='test set')
plt.legend()

xl = plt.xlim()
plt.plot(xl,xl,'r--')
plt.title(f'H, Dir, and Tp, 51-parameters')
plt.text(210, 170, 'RMSE (test) = 7.96-m')
# plt.text(65, 48, '51 trainable params.')

plt.xlabel('Observed shoreline position (m)')
plt.ylabel('Estimated shoreline position (m)')


model_out_all = best_model.predict(eval_x)

# y_est = scaler_eval.inverse_transform(model_out_all).squeeze()
y_est = scaler.inverse_transform(model_out_all).squeeze()

bias = np.mean(evalY.squeeze() - y_est)

plt.subplot(212)
dates = matplotlib.dates.date2num(obs_shoreline['Datetime'][:len(eval_x)].values)

plt.plot(dates, evalY.squeeze() - bias, lw=2, label='Observations')
plt.plot(dates, y_est,'r--', label='Estimates')
plt.legend()
# plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%Y')
plt.gca().xaxis.set_major_formatter(myFmt)


plt.xlabel('Time')
plt.ylabel('Shoreline position (m)')
# plt.show()
plt.savefig(f'./output/CNN_multivar_shoreshop_microCNN_ts.png', dpi=200, bbox_inches='tight')
plt.close()



# plt.plot(yresults['obs'],'k', lw=2, label='Observations')
# plt.plot(yresults['p51b8k6'], 'r--', label='Estimates')
# plt.title(f'Hs-only, 51-parameters')



# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024JH000172
#     BIGmodel = Sequential()

#     BIGmodel.add(Conv1D(filters=128, kernel_size=4,
#                      activation='relu', input_shape=(n_steps_in, n_features)))

#     # BIGmodel.add(Dense(128, activation='relu')) 
#     BIGmodel.add(Dense(256, activation='relu')) 
#     BIGmodel.add(Dropout(.5))       
#     BIGmodel.add(Dense(1, activation='linear'))    




#     model.add(Conv1D(filters=n_filters, kernel_size=n_kernels,
#                      activation='relu', input_shape=(n_steps_in, n_features)))
    

#     if second_ccn_bank is True:
#         model.add(Conv1D(filters=n_filters, kernel_size=n_kernels,
#                      activation='relu'))
#     else:
#         print("using smaller cnn model")

#     model.add(MaxPooling1D(pool_size=2)) 
#     model.add(Flatten()) 
#     if linear_activation is True:
#         model.add(Dense(num_neurons, activation='linear')) 
#     else:
#         model.add(Dense(num_neurons, activation='relu')) 
#     model.add(Dropout(n_drop))       
#     model.add(Dense(1))    







    # plt.figure(figsize=(12,4))
    # plt.subplot(131)
    # plt.plot(mat_in['f'], obs_shoreline['yout'], '.')
    # xc = np.min(np.corrcoef(mat_in['f'], obs_shoreline['yout']))
    # plt.text(.5,75,'R='+str(xc)[:6])
    # plt.xlabel('Wave height anomaly (m)')
    # plt.ylabel('Shoreline position (m)')

    # plt.subplot(132)
    # plt.plot(features[:,0], scaler.inverse_transform(train_y), '.')
    # xc = np.min(np.corrcoef(features[:,0], scaler.inverse_transform(train_y).squeeze()))
    # plt.text(.5,75,'R='+str(xc)[:6])
    # plt.xlabel('CNN feature vector 1 (-)')
    # plt.ylabel('Shoreline position (m)')

    # plt.subplot(133)
    # plt.plot(features[:,1], scaler.inverse_transform(train_y), '.')
    # xc = np.min(np.corrcoef(features[:,1], scaler.inverse_transform(train_y).squeeze()))
    # plt.text(.5,75,'R='+str(xc)[:6])
    # plt.xlabel('CNN feature vector 2 (-)')
    # plt.ylabel('Shoreline position (m)')
