import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


trial = 30
M = 20
N = 1
L_set = [10,20,30,40,50,60]
SNR = 90.0
Tau = 1
Set = 2

minTrial=0
maxTrial=30


Noiseless_acc_test_L = []
Proposed_acc_test_L = []
DC_full_acc_test_L = []
NoDS_acc_test_L = []
NoRIS_acc_test_L = []


for L in L_set:
    Noiseless_acc_test_trial = []
    Proposed_acc_test_trial = []
    DC_full_acc_test_trial = []
    NoDS_acc_test_trial = []
    NoRIS_acc_test_trial = []
    for trialnum in range(minTrial, maxTrial + 1):
        filename_1 = './store/SGD_all_L/SGD_all_L_temp_trialmax_{}_M_{}_N_{}_L_{}_' \
                     'SNR_{}_Tau_{}_set_{}_trialnum_{}.npz'.format(trial, M, N, L, SNR, Tau, Set, trialnum)
        try:
            res = np.load(filename_1, allow_pickle=1)
            print('L:{},trialnum:{}'.format(L,trialnum))
            trial_result = res['arr_1'][()]
            Noiseless_acc_test_trial.append(trial_result['accuracy_test'][-1])
            Proposed_acc_test_trial.append(trial_result['accuracy_test1'][-1])
            DC_full_acc_test_trial.append(trial_result['accuracy_test_DC'][-1])
            NoDS_acc_test_trial.append(trial_result['accuracy_test2'][-1])
            NoRIS_acc_test_trial.append(trial_result['accuracy_test3'][-1])
            pass
        except:
            break
            pass
    Noiseless_acc_test_L.append(np.mean(Noiseless_acc_test_trial))
    Proposed_acc_test_L.append(np.mean(Proposed_acc_test_trial))
    DC_full_acc_test_L.append(np.mean(DC_full_acc_test_trial))
    NoDS_acc_test_L.append(np.mean(NoDS_acc_test_trial))
    NoRIS_acc_test_L.append(np.mean(NoRIS_acc_test_trial))


csv_dic = {
    'L num':L_set,
    'Noiseless_acc_test': Noiseless_acc_test_L,
    'Proposed_acc_test': Proposed_acc_test_L,
    'DC_full_acc_test':DC_full_acc_test_L,
    'NoDS_acc_test': NoDS_acc_test_L,
    'NoRIS_acc_test': NoRIS_acc_test_L,
}
df = pd.DataFrame(csv_dic)
df.to_csv('result_temp.csv', index=False)


data = pd.read_csv('result_temp.csv')

y1 = data['Noiseless_acc_test'].T.values
x1 = data['L num'].T.values


y2 = data['Proposed_acc_test'].T.values
x2 = data['L num'].T.values


y3 = data['DC_full_acc_test'].T.values
x3 = data['L num'].T.values


y4 = data['NoRIS_acc_test'].T.values
x4 = data['L num'].T.values


y5 = data['NoDS_acc_test'].T.values
x5 = data['L num'].T.values


l1, = plt.plot(x1, y1, marker='s', markevery=1)
l2, = plt.plot(x2, y2, marker='o', markevery=1)
l3, = plt.plot(x3, y3, marker='^', markevery=1)
l4, = plt.plot(x4, y4, marker='*', markevery=1)
l5, = plt.plot(x5, y5, marker='x', markevery=1)

plt.grid(True, linestyle='-', alpha=0.5)
plt.title('BGD, straggler issue')
plt.xlabel('Varying L')
plt.ylabel('Test Accuracy')
plt.legend(handles=[l1, l2, l3, l4, l5], labels=['Noiseless', 'Ref[13]', 'DC_Full', 'DC_NoRIS', 'DC_NoDS'], loc='best')
plt.show()
