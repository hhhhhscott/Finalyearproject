import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import copy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

trial = 30
M = 20
N = 1
L = 50
SNR = 90.0
Tau = 1
Set = 2

minTrial = 0
maxTrial = 15

Proposed_obj = []
DC_full_obj = []
NoDS_obj = []
NoRIS_obj = []

Proposed_result = []
DC_full_result = []
NoDS_result = []
NoRIS_result = []

# 1
# filename = './store/trial_{}_M_{}_N_{}_L_{}_\
# SNR_{}_Tau_{}_set_{}.npz'.format(trial, M, N, L, SNR, Tau, Set)
# result = np.load(filename, allow_pickle=1)
###

# 2
tempList = []
for trialnum in range(minTrial, maxTrial + 1):
    filename_1 = './store/SGD_all/SGD_all_temp_trialmax_{}_M_{}_N_{}_L_{}_' \
                 'SNR_{}_Tau_{}_set_{}_trialnum_{}.npz'.format(trial, M, N, L, SNR, Tau, Set, trialnum)
    res = np.load(filename_1, allow_pickle=1)
    trial_set = res['arr_1'][()]

    Proposed_obj.append(res['arr_2'][-1])
    DC_full_obj.append(res['arr_3'])
    NoRIS_obj.append(res['arr_4'])
    NoDS_obj.append(res['arr_5'])

    tempList.append(trial_set)
np.savez('temp.npz', tempList, tempList)
result = np.load('temp.npz', allow_pickle=1)
###


result_set = result['arr_1']

Noiseless_loss_train = []
Noiseless_acc_test = []
Noiseless_loss_test = []

Proposed_loss_train = []
Proposed_acc_test = []
Proposed_loss_test = []


DC_full_loss_train = []
DC_full_acc_test = []
DC_full_loss_test = []

NoDS_loss_train = []
NoDS_acc_test = []
NoDS_loss_test = []

NoRIS_loss_train = []
NoRIS_acc_test = []
NoRIS_loss_test = []


for trial_result in result_set:
    try:
        Noiseless_loss_train.append(trial_result['loss_train'])
        Noiseless_acc_test.append(trial_result['accuracy_test'][:-1])
        Noiseless_loss_test.append(trial_result['loss_test'])
        Noiseless_loss_train = np.mean(Noiseless_loss_train, axis=0)
        Noiseless_acc_test = np.mean(Noiseless_acc_test, axis=0)
        Noiseless_loss_test = np.mean(Noiseless_loss_test, axis=0)
    except:
        pass

    try:
        Proposed_loss_train.append(trial_result['loss_train1'])
        Proposed_acc_test.append(trial_result['accuracy_test1'][:-1])
        Proposed_loss_test.append(trial_result['loss_test1'])
        Proposed_loss_train = np.mean(Proposed_loss_train, axis=0)
        Proposed_acc_test = np.mean(Proposed_acc_test, axis=0)
        Proposed_loss_test = np.mean(Proposed_loss_test, axis=0)
    except:
        pass

    try:
        DC_full_loss_train.append(trial_result['loss_train_DC'])
        DC_full_acc_test.append(trial_result['accuracy_test_DC'][:-1])
        DC_full_loss_test.append(trial_result['loss_test_DC'])
        DC_full_loss_train = np.mean(DC_full_loss_train, axis=0)
        DC_full_acc_test = np.mean(DC_full_acc_test, axis=0)
        DC_full_loss_test = np.mean(DC_full_loss_test, axis=0)
    except:
        pass

    try:
        NoDS_loss_train.append(trial_result['loss_train2'])
        NoDS_acc_test.append(trial_result['accuracy_test2'][:-1])
        NoDS_loss_test.append(trial_result['loss_test2'])
        NoDS_loss_train = np.mean(NoDS_loss_train, axis=0)
        NoDS_acc_test = np.mean(NoDS_acc_test, axis=0)
        NoDS_loss_test = np.mean(NoDS_loss_test, axis=0)
    except:
        pass

    try:
        NoRIS_loss_train.append(trial_result['loss_train3'])
        NoRIS_acc_test.append(trial_result['accuracy_test3'][:-1])
        NoRIS_loss_test.append(trial_result['loss_test3'])
        NoRIS_loss_train = np.mean(NoRIS_loss_train, axis=0)
        NoRIS_acc_test = np.mean(NoRIS_acc_test, axis=0)
        NoRIS_loss_test = np.mean(NoRIS_loss_test, axis=0)
    except:
        pass

csv_dic = {
    'Noiseless_loss_train': Noiseless_loss_train, 'Noiseless_acc_test': Noiseless_acc_test,
    'Noiseless_loss_test': Noiseless_loss_test,
    'Proposed_loss_train': Proposed_loss_train, 'Proposed_acc_test': Proposed_acc_test,
    'Proposed_loss_test': Proposed_acc_test,
    'DC_full_loss_train': DC_full_loss_train, 'DC_full_acc_test': DC_full_acc_test,
    'DC_full_loss_test': DC_full_loss_test,
    'NoDS_loss_train': NoDS_loss_train, 'NoDS_acc_test': NoDS_acc_test, 'NoDS_loss_test': NoDS_loss_test,
    'NoRIS_loss_train': NoRIS_loss_train, 'NoRIS_acc_test': NoRIS_acc_test, 'NoRIS_loss_test': NoRIS_loss_test
}
df = pd.DataFrame(csv_dic)
df.to_csv('result_temp.csv', index=False)

# 画基础实验的
data = pd.read_csv('result_temp.csv')

y1 = data['Noiseless_acc_test'].T.values
x1 = list(range(0, len(y1)))

y2 = data['Proposed_acc_test'].T.values
x2 = list(range(0, len(y2)))

y3 = data['DC_full_acc_test'].T.values
x3 = list(range(0, len(y3)))

y4 = data['NoRIS_acc_test'].T.values
x4 = list(range(0, len(y4)))

y5 = data['NoDS_acc_test'].T.values
x5 = list(range(0, len(y5)))

l1, = plt.plot(x1, y1, marker='s', markevery=40)
l2, = plt.plot(x2, y2, marker='o', markevery=40)
l3, = plt.plot(x3, y3, marker='^', markevery=40)
l4, = plt.plot(x4, y4, marker='*', markevery=40)
l5, = plt.plot(x5, y5, marker='x', markevery=40)

plt.grid(True, linestyle='-', alpha=0.5)
plt.title('SGD, straggler issue')
plt.xlabel('Global Round')
plt.ylabel('Test Accuracy')
plt.legend(handles=[l1, l2, l3, l4, l5], labels=['Noiseless', 'Ref[13]', 'DC_Full', 'DC_NoRIS', 'DC_NoDS'], loc='best')
plt.show()
