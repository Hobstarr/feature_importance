import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
sns.set_style('whitegrid')
# import dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# visualise the data
plt.imshow(x_train[0], cmap = 'gray_r')
plt.show()

# reshape training data and normalise
x_train_flat = np.reshape(x_train / 255, 
           (x_train.shape[0], x_train.shape[1] *x_train.shape[1]))

rf_class = RandomForestClassifier(n_estimators = 100)
rf_class.fit(x_train_flat, y_train)

im_importances = np.reshape(rf_class.feature_importances_, (28,28))
plt.imshow(im_importances)
plt.show()
