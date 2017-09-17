# To plot pretty figures
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

plt.ylabel(r'$\%\ Classified$')
plt.xlabel('Epoch')
plt.title('Training Sample, % Correct vs Epoch')
plt.plot(performance)