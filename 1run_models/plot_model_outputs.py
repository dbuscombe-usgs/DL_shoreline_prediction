

import pandas as pd 
import matplotlib.pyplot as plt 


orig = pd.read_csv('output/CNN_ensemble_orig.csv')
smallcnn = pd.read_csv('output/CNN_ensemble_smallcnn.csv')
Hsonly = pd.read_csv('output/CNN_ensemble_Hs_only_smallcnn.csv')
Hsonly_linear = pd.read_csv('output/CNN_ensemble_Hs_only_smallcnn_linearact.csv')
Hsonly_linear_tiny = pd.read_csv('output/CNN_ensemble_Hs_only_tinycnn_linearact.csv')



orig_metrics = pd.read_csv('output/CNN_ensemble_orig_metrics.csv')
smallcnn_metrics = pd.read_csv('output/CNN_ensemble_smallcnn_metrics.csv')
Hsonly_metrics = pd.read_csv('output/CNN_ensemble_Hs_only_smallcnn_metrics.csv')
Hsonly_linear_metrics = pd.read_csv('output/CNN_ensemble_Hs_only_smallcnn_linearact_metrics.csv')
Hsonly_linear_tiny_metrics = pd.read_csv('output/CNN_ensemble_Hs_only_tinycnn_linearact_metrics.csv')



plt.figure(figsize=(10,10))
plt.subplot(221)
plt.plot(orig['obs'], orig['ann3'], 'r.')
xl = plt.xlim()
plt.plot(xl,xl,'r--')
plt.title('Gomez-de la Pena et al. 2023 (best multivariate model)')
plt.text(65, 50, 'RMSE (m) = 4.54')
plt.text(65, 45, '144,633 trainable params.')


plt.subplot(222)
plt.plot(Hsonly['obs'], Hsonly['ann2'], 'r.')
xl = plt.xlim()
plt.plot(xl,xl,'r--')
plt.title('smaller CNN, and Hs-only (univariate)')
plt.text(65, 50, 'RMSE (m) = 4.94')
plt.text(65, 45, '130,161 trainable params.')


plt.subplot(223)
plt.plot(Hsonly_linear['obs'], Hsonly_linear['ann1'], 'r.')
xl = plt.xlim()
plt.plot(xl,xl,'r--')
plt.title('smaller CNN, Hs-only, linear activation')
plt.text(65, 50, 'RMSE (m) = 4.93')
plt.text(65, 45, '130,161 trainable params.')

plt.subplot(224)
plt.plot(Hsonly_linear_tiny['obs'], Hsonly_linear_tiny['ann3'], 'r.')
xl = plt.xlim()
plt.plot(xl,xl,'r--')
plt.title('tiny CNN, Hs-only, linear activation')
plt.text(65, 50, 'RMSE (m) = 4.98')
plt.text(65, 45, '3,811 trainable params.')


# plt.show()
plt.savefig('CNN_shoreline_models.png',dpi=200, bbox_inches='tight')
plt.close()