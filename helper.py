import matplotlib.pyplot as plt
from IPython import display
import numpy as np

x_range = np.arange(0, 151, 20).tolist()
y_range = np.arange(0, 201, 20).tolist()

plt.ion()
plt.figure(figsize=(10,6))
i = 0
def plot(score):
    global i
    if i % 5 == 0:
      display.clear_output(wait=True)
      display.display(plt.gcf())
      plt.title(f'On Iteration {i}')
      plt.xlabel('Number of Iterations')
      plt.ylabel('Highest Survived Episode')
      plt.yticks(y_range)
      plt.xticks(x_range)
      plt.ylim([0,200])
      plt.xlim([0,150])
      plt.scatter(i,score, c='black')
      plt.show(block=False)
      plt.pause(0.1)
    i += 1