from mnist import read, show
import numpy as np

class KNN:
    def __init__(self, data, labels, k):
        self.data = data
        self.labels = labels
        self.k = k
    def predict(self, test_data, k):
        diffs = np.substract(self.data, test_data)
        return diffs



training_data = list(read(dataset='training', path='./mnist'))
testing_data = list(read(dataset='testing', path='./mnist'))
train_data_pixels = { training_data }

print(training_data[0][1])

# print(len(training_data))
label, pixels = training_data[0]
print(label)
print(pixels.shape)
show(training_data[0][1])

knn = KNN(training_data, [0,1,2,3,4,5,6,7,8,9], 700)
print(knn.predict(testing_data, 1))