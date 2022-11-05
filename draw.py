import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d


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
plt.title('BGD,no straggler issue')
plt.xlabel('Global Round')
plt.ylabel('Test Accuracy')
plt.legend(handles=[l1, l2, l3, l4, l5], labels=['Noiseless', 'Ref[13]', 'DC_Full', 'DC_NoRIS', 'DC_NoDS'], loc='best')
plt.show()
