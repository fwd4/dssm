import matplotlib.pyplot as plt
import numpy as np



num_workers = range(1,8)
time = [769, 400, 370, 300, 285, 260, 242]

plt.grid()

barwidth = 0.3
x = np.arange(7) - barwidth / 2 + 1
barlist = plt.bar(x, time, barwidth)

plt.xticks(num_workers)
plt.title('Training Time Per Epoch')
plt.ylabel("Time (s)")
plt.xlabel("Number of Workers")
plt.xlim(0.5, len(num_workers) + 0.5)

plt.show()
