
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

##========================================================
def rescale_array(dat, mn, mx):
    """
    rescales an input dat between mn and mx
    """
    m = min(dat.flatten())
    M = max(dat.flatten())
    return (mx - mn) * (dat - m) / (M - m) + mn


##Import Observed Shoreline positions
all_in= pd.read_csv('../data/OB_waves_shorelines.csv',index_col=False) 

all_in['time'] = pd.to_datetime(all_in['time'])  


Y_time = all_in['time']
all_in.reset_index(drop=True, inplace=True)

hobs = all_in.drop(columns=['time', 'Y_detrend',  'Y' ])#.mean(axis=1)

yobs = all_in.drop(columns=['time', 'Hs',  'Y' ])#.mean(axis=1)

y_tlinux = (pd.to_datetime(Y_time)- dt.datetime(1970,1,1)).dt.total_seconds()


mat_in = pd.DataFrame.from_dict({'Datetime': all_in['time'].values.squeeze(), 'Hs': hobs.values.squeeze(), 'yout': yobs.values.squeeze() })
mat_in = mat_in.interpolate()

mat_in['Datetime'] = pd.to_datetime(mat_in['Datetime'])

mat_in = mat_in.set_index(['Datetime'])
mat_in = mat_in.dropna(how='any') 



obs_shoreline = pd.DataFrame(mat_in.yout.values, index=mat_in.index)

realtime = mat_in.index.values



## just waves
Hs = mat_in['Hs'].values

Hbar = np.mean(Hs)
## waver energy anomaly
f=(Hs**2-Hbar**2)/Hbar**2 

mat_in = mat_in.assign(f=f)
mat_in = mat_in[['f','yout']]


mat_out= pd.DataFrame(mat_in.yout.values, index=mat_in.index)
mat_out = mat_out.interpolate()

from sklearn import preprocessing

scaler_x = preprocessing.StandardScaler() 

scaler_x = scaler_x.fit(f.reshape(-1,1))


###################TRAIN, DEV AND TEST SPLITS################################
### first create an 'evaluate' seq which is for final model plotting and for input scaling 
date_evaluate= '2000-03-04'
eval_date=pd.to_datetime(date_evaluate)

_, mat_eval = normal_data(mat_in,eval_date)
scaler_eval, mateval_out_norm = normal_data(mat_out,eval_date)

eval = mat_eval.values.astype('float32')


# n_steps_in, n_steps_out =40,1

n_steps_out =1

for n_steps_in in [20,30,40,50,60,70,80]:

    eval_x, eval_y = split_sequences(eval, n_steps_in, n_steps_out)
    evalY = scaler_eval.inverse_transform(eval_y)

    ### Ocean Beach forecast period from 2015 onwards
    date_forecast= '2015-01-01'
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

    #####
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



    ##EXPORT ENSEMBLE
    yresults["obs"]=testY[:,0]
    yresults.to_csv(f'./output/CNN_f_microcnn_OceanBeach_nn{num_neurons}_p{trainable_count}_linearact_DT{n_steps_in}.csv')

    ## Metrics 
    ymetrics= pd.DataFrame(index=np.arange(7),
                        columns=['rmse_arr','pear_arr','mielke_arr'])

    ymetrics['rmse_arr']=np.array([root_mean_squared_error(yresults[colname].values,testY) for (index, colname) in enumerate(yresults)])
    ymetrics['pear_arr']=np.array([scipy.stats.pearsonr(yresults[colname].values,testY[:,0])[0] for (index, colname) in enumerate(yresults)])
    ymetrics['mielke_arr']=np.array( [index_mielke(yresults[colname].values,testY[:,0]) for (index, colname) in enumerate(yresults)])

    ymetrics.to_csv(f'./output/CNN_f_microcnn_OceanBeach_nn{num_neurons}_p{trainable_count}_DT{n_steps_in}_linearact_metrics.csv')


    #################

    num_params = PARAMS[np.argmin(ymetrics['rmse_arr'][:-1])]
    best_model = MODELS[np.argmin(ymetrics['rmse_arr'][:-1])]

    best_rmse = ymetrics['rmse_arr'][np.argmin(ymetrics['rmse_arr'][:-1])]
    best_rmse = str(best_rmse)[:5]

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
    plt.title(f'Hs-only, {num_params}-parameters')
    plt.text(20, -40, 'RMSE (test) = {best_rmse}-m')
    # plt.text(65, 48, '51 trainable params.')

    plt.xlabel('Observed shoreline position (m)')
    plt.ylabel('Estimated shoreline position (m)')


    model_out_all = best_model.predict(eval_x)

    # y_est = scaler_eval.inverse_transform(model_out_all).squeeze()
    y_est = scaler.inverse_transform(model_out_all).squeeze()

    bias = np.mean(evalY.squeeze() - y_est)

    plt.subplot(212)
    dates = matplotlib.dates.date2num(obs_shoreline.index.values[:len(eval_x)])

    plt.plot(dates, evalY.squeeze(), lw=2, label='Observations')
    plt.plot(dates, y_est + bias,'r--', label='Estimates')
    plt.legend()
    # plt.gcf().autofmt_xdate()
    myFmt = mdates.DateFormatter('%Y')
    plt.gca().xaxis.set_major_formatter(myFmt)


    plt.xlabel('Time')
    plt.ylabel('Shoreline position (m)')
    # plt.show()
    plt.savefig(f'./output/CNN_f_OceanBeach_microCNN_ts_DT{n_steps_in}_params{num_params}.png', dpi=200, bbox_inches='tight')
    plt.close()




    ## use the model as afeature extractor to examine correlation between features and outputs

    plt.close()
    plt.figure(figsize=(16,8))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.subplot(311)
    dates = matplotlib.dates.date2num(obs_shoreline.index.values[:len(eval_x)])

    plt.plot(dates, evalY.squeeze(), lw=2, label='Observations')
    plt.plot(dates, y_est + bias,'r--', label='Estimates')
    plt.legend()
    # plt.gcf().autofmt_xdate()
    myFmt = mdates.DateFormatter('%Y')
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.text(dates[int(len(dates)/2)], -50, f'RMSE (test) = {best_rmse}-m')
    plt.xlabel('Time')
    plt.ylabel('Shoreline position (m)')

    ax = plt.subplot(312)
    ax2 = ax.twinx()
  
    weights = Model(best_model.inputs, best_model.layers[0].output)
    cnn_weights = weights.predict(eval_x)

    ax.plot(dates, cnn_weights[:,:,0].mean(axis=1),'r', label='CNN weights')
    ax2.plot(dates, cnn_weights[:,:,1].mean(axis=1),'b',label='bias')
    # plt.legend()
    plt.xlabel('Time')
    myFmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_formatter(myFmt)
    ax.set_xlabel('Time')
    ax.set_ylabel('Convolution layer weights (-)', color='r')
    ax2.set_ylabel('Convolution layer biases (-)', color='b')

    ax3 = plt.subplot(313)
    ax4 = ax3.twinx()

    # shoreML = np.convolve(cnn_weights[:,:,0].mean(axis=1), features[:,0],'full')[:len(dates)]
    ax3.plot(dates,  evalY.squeeze(), lw=2, label='Observations')
    ax4.plot(dates, cnn_weights[:,:,0].mean(axis=1),'r--', label='CNN weights')
    myFmt = mdates.DateFormatter('%Y')
    ax3.xaxis.set_major_formatter(myFmt)
    plt.xlabel('Time')

    ax3.set_xlabel('Time')
    ax3.set_ylabel('Observed shoreline position (m)', color='b')
    ax4.set_ylabel('Convolution layer weights (-)', color='r')

    # plt.show()
    plt.savefig(f'./output/CNN_f_microcnn_OceanBeach_nn{num_neurons}_p'+str(trainable_count)+'b'+str(batch)+'k'+str(kernel)+'_linearact_DT'+str(n_steps_in)+'_model_explore1.png', dpi=200, bbox_inches='tight')
    plt.close('all')



    ######
    extract = Model(best_model.inputs, best_model.layers[-3].output)
    features = extract.predict(train_x)

    ###====================================
    plt.figure(figsize=(12,12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.subplot(2,2,1)
    plt.plot(mat_in['f'], obs_shoreline.values.squeeze(), '.')
    xc = np.min(np.corrcoef(mat_in['f'], obs_shoreline.values.squeeze()))
    plt.text(7.5,-30,'R='+str(xc)[:6])
    plt.xlabel('Wave height anomaly (m)')
    plt.ylabel('Shoreline position (m)')

    plt.subplot(2,2,2)
    plt.plot(features[:,0], scaler.inverse_transform(train_y), '.')
    xc = np.min(np.corrcoef(features[:,0], scaler.inverse_transform(train_y).squeeze()))
    plt.text(-5,-30,'R='+str(xc)[:6])
    plt.xlabel('Dense layer feature vector (-)')
    plt.ylabel('Shoreline position (m)')

    plt.subplot(2,2,3)
    plt.plot(cnn_weights[:,:,0].mean(axis=1), evalY.squeeze(), '.')
    xc = np.min(np.corrcoef(cnn_weights[:,:,0].mean(axis=1), evalY.squeeze()))
    plt.text(3,30,'R='+str(xc)[:6])
    plt.xlabel('Convolution layer weights (-)')
    plt.ylabel('Shoreline position (m)')

    plt.subplot(2,2,4)
    # plt.plot(cnn_weights[:,:,1].mean(axis=1), eval_x.mean(axis=1), '.')
    plt.plot(cnn_weights[:,:,1].mean(axis=1), rescale_array(eval_x.mean(axis=1),f.min(), f.max()).squeeze(), '.')
    # plt.plot(cnn_weights[:,:,1].mean(axis=1), (y_est + bias) - evalY.squeeze(), '.')
    xc = np.min(np.corrcoef(cnn_weights[:,:,1].mean(axis=1), rescale_array(eval_x.mean(axis=1),f.min(), f.max()).squeeze()))
    plt.text(.1,2,'R='+str(xc)[:6])
    plt.xlabel('Convolution layer bias (-)')
    plt.ylabel('Wave height anomaly (m)')

    # plt.show()
    plt.savefig(f'./output/CNN_f_microcnn_OceanBeach_nn{num_neurons}_p'+str(trainable_count)+'b'+str(batch)+'k'+str(kernel)+'_linearact_DT'+str(n_steps_in)+'_modelexplore2.png', dpi=200, bbox_inches='tight')
    plt.close()

