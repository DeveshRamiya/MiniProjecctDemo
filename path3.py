#MiniProjectPath3
import numpy as np
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
#import models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
import copy


rng = np.random.RandomState(1)
digits = datasets.load_digits()
images = digits.images
labels = digits.target

#Get our training data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.6, shuffle=False)
#basically takes in a array of integers, images, and labels (dont worry about labels and images that is an inbuilt thing)
#it basically returns an object (which is an image) [1,2,3,4,5,6]
def dataset_searcher(number_list,images,labels):
  #insert code that when given a list of integers, will find the labels and images
  #and put them all in numpy arrary (at the same time, as training and testing data)
  # images_nparray = np.array([]);
  # labels_nparray = np.array([]);
  imgs= []
  lbls = []
  for i in number_list:
    imgs.append(images[i])
    lbls.append(labels[i])
    # images_nparray = np.append(images_nparray,images[i])
    # images_nparray = np.append(labels_nparray,labels[i])
  images_nparray =  np.array(imgs)
  labels_nparray = np.array(lbls)
  return images_nparray, labels_nparray

def print_numbers(images,labels):
  #insert code that when given images and labels (of numpy arrays)
  #the code will plot the images and their labels in the title. 
  for i in range(len(images)):
    plt.matshow(images[i])
    plt.title(labels[i])
  plt.show()


class_numbers = [2,0,8,7,5]
#Part 1
class_number_images , class_number_labels = dataset_searcher(class_numbers, images, labels)
#Part 2
# print_numbers(class_number_images , class_number_labels)


model_1 = GaussianNB()

#however, before we fit the model we need to change the 8x8 image data into 1 dimension
# so instead of having the Xtrain data beign of shape 718 (718 images) by 8 by 8
# the new shape would be 718 by 64
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)

#Now we can fit the model
model_1.fit(X_train_reshaped, y_train)
#Part 3 Calculate model1_results using model_1.predict()
model1_results = model_1.predict(X_test.reshape(X_test.shape[0], -1))#What should go in here? Hint, look at documentation and some reshaping may need to be done)


def OverallAccuracy(results, actual_values):
  #Calculate the overall accuracy of the model (out of the predicted labels, how many were correct?)
  Accuracy = metrics.accuracy_score(results,actual_values)
  return Accuracy


# Part 4
Model1_Overall_Accuracy = OverallAccuracy(model1_results, y_test)
print("The overall results of the Gaussian model is " + str(Model1_Overall_Accuracy))


#Part 5
allnumbers = [0,1,2,3,4,5,6,7,8,9]
allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers, images, labels)
allNumsreshaped = allnumbers_images.reshape(allnumbers_images.shape[0],-1)
model1_learnt_result = model_1.predict(allnumbers_images.reshape(allnumbers_images.shape[0],-1))
print(model1_learnt_result)
learnt_images, learnt_labels = dataset_searcher(model1_learnt_result, images, labels)
print_numbers(learnt_images, learnt_labels)


#Part 6
#Repeat for K Nearest Neighbors
model_2 = KNeighborsClassifier(n_neighbors=10)
model_2.fit(X_train_reshaped, y_train)
model2_results = model_2.predict(X_test.reshape(X_test.shape[0], -1))
Model2_Overall_Accuracy = OverallAccuracy(model2_results, y_test)
print("The overall results of the KNeighbors Classifier model is " + str(Model2_Overall_Accuracy))
allnumbers = [0,1,2,3,4,5,6,7,8,9]
allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers, images, labels)
allNumsreshaped = allnumbers_images.reshape(allnumbers_images.shape[0],-1)
model2_learnt_result = model_2.predict(allnumbers_images.reshape(allnumbers_images.shape[0],-1))
print(model2_learnt_result)
learnt_images, learnt_labels = dataset_searcher(model2_learnt_result, images, labels)
# print_numbers(learnt_images, learnt_labels)



model_3 = MLPClassifier(random_state=0)
model_3.fit(X_train_reshaped, y_train)
model3_results = model_3.predict(X_test.reshape(X_test.shape[0], -1))#What should go in here? Hint, look at documentation and some reshaping may need to be done)
Model3_Overall_Accuracy = OverallAccuracy(model3_results, y_test)
print("The overall results of the MLP Classifier model is " + str(Model3_Overall_Accuracy))
allnumbers = [0,1,2,3,4,5,6,7,8,9]
allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers, images, labels)
allNumsreshaped = allnumbers_images.reshape(allnumbers_images.shape[0],-1)
model3_learnt_result = model_3.predict(allnumbers_images.reshape(allnumbers_images.shape[0],-1))
print(model3_learnt_result)
learnt_images, learnt_labels = dataset_searcher(model3_learnt_result, images, labels)
# print_numbers(learnt_images, learnt_labels)

#Part 8
#Poisoning
# Code for generating poison data. There is nothing to change here.
noise_scale = 10.0
poison = rng.normal(scale=noise_scale, size=X_train.shape)

X_train_poison = X_train + poison


# #Part 9-11
# #Determine the 3 models performance but with the poisoned training data X_train_poison and y_train instead of X_train and y_train
#model1
X_train_poisoned_reshaped = X_train_poison.reshape(X_train_poison.shape[0], -1)
model_1.fit(X_train_poisoned_reshaped, y_train)
model1_poisoned_results = model_1.predict(X_test.reshape(X_test.shape[0], -1))
Model1_Poisoned_Overall_Accuracy = OverallAccuracy(model1_poisoned_results, y_test)
print("The overall results of the poisoned Gaussian model is " + str(Model1_Poisoned_Overall_Accuracy))

# model2
model_2.fit(X_train_poisoned_reshaped, y_train)
model2_poisoned_results = model_2.predict(X_test.reshape(X_test.shape[0], -1))
Model2_Poisoned_Overall_Accuracy = OverallAccuracy(model2_poisoned_results, y_test)
print("The overall results of the poisoned KNeighbors Classifier model is " + str(Model2_Poisoned_Overall_Accuracy))

#model3
model_3.fit(X_train_poisoned_reshaped, y_train)
model3_poisoned_results = model_3.predict(X_test.reshape(X_test.shape[0], -1))
Model3_Poisoned_Overall_Accuracy = OverallAccuracy(model3_poisoned_results, y_test)
print("The overall results of the poisoned MLP Classifier model is " + str(Model3_Poisoned_Overall_Accuracy))


# #Part 12-13
# # Denoise the poisoned training data, X_train_poison. 
# # hint --> Suggest using KernelPCA method from sklearn library, for denoising the data. 
# # When fitting the KernelPCA method, the input image of size 8x8 should be reshaped into 1 dimension
# # So instead of using the X_train_poison data of shape 718 (718 images) by 8 by 8, the new shape would be 718 by 64

# # X_train_denoised = # fill in the code here

kernel_pca = KernelPCA(
    n_components=400, kernel="rbf", gamma=1e-3, fit_inverse_transform=True, alpha=5e-3
)
kernel_pca.fit(X_train_poisoned_reshaped)
X_train_denoised = kernel_pca.inverse_transform(
    kernel_pca.transform(X_train_poisoned_reshaped)
)



# #Part 14-15
# #Determine the 3 models performance but with the denoised training data, X_train_denoised and y_train instead of X_train_poison and y_train
# #Explain how the model performances changed after the denoising process.
model_1.fit(X_train_denoised, y_train)
model1_poisoned_results = model_1.predict(X_test.reshape(X_test.shape[0], -1))
Model1_Poisoned_Overall_Accuracy = OverallAccuracy(model1_poisoned_results, y_test)
print("The overall results of the denoised Gaussian model is " + str(Model1_Poisoned_Overall_Accuracy))

# model2
model_2.fit(X_train_denoised, y_train)
model2_poisoned_results = model_2.predict(X_test.reshape(X_test.shape[0], -1))
Model2_Poisoned_Overall_Accuracy = OverallAccuracy(model2_poisoned_results, y_test)
print("The overall results of the denoised KNeighbors Classifier model is " + str(Model2_Poisoned_Overall_Accuracy))

#model3
model_3.fit(X_train_denoised, y_train)
model3_poisoned_results = model_3.predict(X_test.reshape(X_test.shape[0], -1))
Model3_Poisoned_Overall_Accuracy = OverallAccuracy(model3_poisoned_results, y_test)
print("The overall results of the denoised MLP Classifier model is " + str(Model3_Poisoned_Overall_Accuracy))

