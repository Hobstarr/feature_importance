import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
sns.set_style('whitegrid')
# import dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape training data and normalise
x_train_flat = np.reshape(x_train / 255, 
           (x_train.shape[0], x_train.shape[1] *x_train.shape[1]))

x_test_flat = np.reshape(x_test / 255, 
           (x_test.shape[0], x_test.shape[1] *x_test.shape[1]))

rf_class = RandomForestClassifier(n_estimators = 100)
rf_class.fit(x_train_flat, y_train)

im_importances = np.reshape(rf_class.feature_importances_, (28,28))
plt.imshow(im_importances, cmap = 'viridis')
plt.title('feature importance for MNIST', fontsize = 18)
plt.axis('off')
plt.show()

# get the value of the n most important features
# and then create and score the model with n features
importances = rf_class.feature_importances_
score_list = []
top_n_list = []
for top_n in range(10,500,10):
    nth_important_feat = np.sort(importances)[-top_n]

    mask = importances >= nth_important_feat
    x_train_important = np.array([list(x[mask]) for x in x_train_flat])
    x_test_important = np.array([list(x[mask]) for x in x_test_flat])

    rf_class_small = RandomForestClassifier(n_estimators = 100)
    rf_class_small.fit(x_train_important, y_train)

    score_list.append(rf_class_small.score(x_test_important, y_test))
    top_n_list.append(top_n)

# plot results of reduced model and full model
full_score = rf_class.score(x_test_flat, y_test)
plt.plot(top_n_list, score_list, label = 'feature reduced model')
plt.axhline(full_score, label = 'full model', 
            color = 'green', linestyle = 'dashed')
plt.legend()
plt.title('feature reduced model vs full model')
plt.ylabel('score')
plt.xlabel('features used')
plt.show()
