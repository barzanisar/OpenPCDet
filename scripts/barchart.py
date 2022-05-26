import matplotlib.pyplot as plt
import numpy as np

objects = ('Toronto', 'Vancouver', 'Montreal', 'San Francisco', 'Phoenix')
y_pos = np.arange(len(objects))
precip = [182, 192, 163, 72, 33 ]

plt.figure(figsize=(10,5))
plt.bar(y_pos, precip, align='center', alpha=0.5)
plt.bar(y_pos, precip, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.title('Days of Precipitation (per year)')
#plt.savefig('Precipitation_Days.png')
plt.show()