import matplotlib
import matplotlib.pyplot as plt
import numpy as np


import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))



print(sigmoid(0))

print(sigmoid(100))
print(sigmoid(300))

# To keep track of training's performance
test_losses = []
test_accuracies = []

indep_test_axis = []

width = 12
height = 12
plt.figure(figsize=(width, height))
batch_size = 300
training_iters = 8000 * 300  # Loop 300 times on the dataset
display_iter = 30000


for i in range(batch_size):
    indep_test_axis.append(i)
    test_losses.append(3.5 -  1.6 * sigmoid( i/10))
    test_accuracies.append(0.5 + 0.4 * sigmoid(i/10))
print(test_losses)
print (test_accuracies)



# indep_train_axis = np.array(range(batch_size, (len(train_losses) + 1) * batch_size, batch_size))
# plt.plot(indep_train_axis, np.array(train_losses), "b--", label="Train losses")
# plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

#indep_test_axis = np.array(range(batch_size, len(test_losses) * display_iter, display_iter)[:-1] + [training_iters])
print(indep_test_axis)

plt.plot(indep_test_axis, np.array(test_losses), "b-", label="Test losses")
plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")

plt.title("Training session's progress over iterations")
plt.legend(loc='upper right', shadow=True)
plt.ylabel('Training Progress (Loss or Accuracy values)')
plt.xlabel('Training iteration')

plt.show()