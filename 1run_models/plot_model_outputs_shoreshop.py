

import pandas as pd 
import matplotlib.pyplot as plt 


# orig = pd.read_csv('output/CNN_ensemble_orig.csv')
# smallcnn = pd.read_csv('output/CNN_ensemble_smallcnn_shoreshop2.csv')
Hsonly = pd.read_csv('output/CNN_ensemble_Hs_only_smallcnn_shoreshop2.csv')
Hsonly_linear = pd.read_csv('output/CNN_ensemble_Hs_only_smallcnn_linearact_shoreshop2.csv')
Hsonly_linear_tiny = pd.read_csv('output/CNN_ensemble_Hs_only_tinycnn_linearact_shoreshop2.csv')



# orig_metrics = pd.read_csv('output/CNN_ensemble_orig_metrics.csv')
# smallcnn_metrics = pd.read_csv('output/CNN_ensemble_smallcnn_metrics_shoreshop2.csv')
Hsonly_metrics = pd.read_csv('output/CNN_ensemble_Hs_only_smallcnn_metrics_shoreshop2.csv')
Hsonly_linear_metrics = pd.read_csv('output/CNN_ensemble_Hs_only_smallcnn_linearact_metrics_shoreshop2.csv')
Hsonly_linear_tiny_metrics = pd.read_csv('output/CNN_ensemble_Hs_only_tinycnn_linearact_metrics_shoreshop2.csv')



plt.figure(figsize=(10,6))
# plt.subplot(221)
# plt.plot(orig['obs'], orig['ann3'], 'r.')
# xl = plt.xlim()
# plt.plot(xl,xl,'r--')
# plt.title('Gomez-de la Pena et al. 2023 (best multivariate model)')
# plt.text(65, 50, 'RMSE (m) = 4.54')
# plt.text(65, 45, '144,633 trainable params.')


plt.subplot(131)
plt.plot(Hsonly['obs'], Hsonly['ann1'], 'r.')
xl = plt.xlim()
plt.plot(xl,xl,'r--')
plt.title('smaller CNN, and Hs-only \n(univariate)')
plt.text(180, 164, 'RMSE (m) = 8.34')
plt.text(180, 158, '130,161\n trainable params.')


plt.subplot(132)
plt.plot(Hsonly_linear['obs'], Hsonly_linear['ann7'], 'r.')
xl = plt.xlim()
plt.plot(xl,xl,'r--')
plt.title('smaller CNN, Hs-only,\n linear activation')
plt.text(180, 164, 'RMSE (m) = 8.75')
plt.text(180, 158, '130,161\n trainable params.')

plt.subplot(133)
plt.plot(Hsonly_linear_tiny['obs'], Hsonly_linear_tiny['ann5'], 'r.')
xl = plt.xlim()
plt.plot(xl,xl,'r--')
plt.title('tiny CNN, Hs-only,\n linear activation')
plt.text(180, 164, 'RMSE (m) = 8.27')
plt.text(180, 158, '3,811\n trainable params.')


# plt.show()
plt.savefig('CNN_shoreline_models_Hsonly_shoreshop.png',dpi=200, bbox_inches='tight')
plt.close()














Hsonly = pd.read_csv('output/CNN_ensemble_multivar_smallcnn_shoreshop2.csv')
Hsonly_linear = pd.read_csv('output/CNN_ensemble_multivar_smallcnn_linearact_shoreshop2.csv')
Hsonly_linear_tiny = pd.read_csv('output/CNN_ensemble_multivar_tinycnn_linearact_shoreshop2.csv')



Hsonly_metrics = pd.read_csv('output/CNN_ensemble_multivar_smallcnn_metrics_shoreshop2.csv')
Hsonly_linear_metrics = pd.read_csv('output/CNN_ensemble_multivar_smallcnn_linearact_metrics_shoreshop2.csv')
Hsonly_linear_tiny_metrics = pd.read_csv('output/CNN_ensemble_multivar_tinycnn_linearact_metrics_shoreshop2.csv')



plt.figure(figsize=(10,6))
# plt.subplot(221)
# plt.plot(orig['obs'], orig['ann3'], 'r.')
# xl = plt.xlim()
# plt.plot(xl,xl,'r--')
# plt.title('Gomez-de la Pena et al. 2023 (best multivariate model)')
# plt.text(65, 50, 'RMSE (m) = 4.54')
# plt.text(65, 45, '144,633 trainable params.')


plt.subplot(131)
plt.plot(Hsonly['obs'], Hsonly['ann7'], 'r.')
xl = plt.xlim()
plt.plot(xl,xl,'r--')
plt.title('smaller CNN \n(multivariate)')
plt.text(180, 164, 'RMSE (m) = 7.83')
plt.text(180, 158, '130,737\n trainable params.')


plt.subplot(132)
plt.plot(Hsonly_linear['obs'], Hsonly_linear['ann7'], 'r.')
xl = plt.xlim()
plt.plot(xl,xl,'r--')
plt.title('smaller CNN, multivariate,\n linear activation')
plt.text(180, 164, 'RMSE (m) = 8.25')
plt.text(180, 158, '130,737\n trainable params.')

plt.subplot(133)
plt.plot(Hsonly_linear_tiny['obs'], Hsonly_linear_tiny['ann9'], 'r.')
xl = plt.xlim()
plt.plot(xl,xl,'r--')
plt.title('tiny CNN, multivariate,\n linear activation')
plt.text(180, 164, 'RMSE (m) = 7.99')
plt.text(180, 158, '3,645\n trainable params.')


# plt.show()
plt.savefig('CNN_shoreline_models_multivariate_shoreshop.png',dpi=200, bbox_inches='tight')
plt.close()