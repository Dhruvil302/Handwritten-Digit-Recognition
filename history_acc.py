import pickle
import cv2
import matplotlib.pyplot as plt
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

with open('./history.pkl', "rb") as file:
    history = pickle.load(file)
#model=load_model('CNN_nodel.h5')
loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

(X_train,y_train), (X_test,y_test) = mnist.load_data()

test_images = X_test.reshape((10000, 28 , 28,1)).astype('float32') / 255
test_targets = to_categorical(y_test)

model=load_model('CNN_nodel.h5')
score = model.evaluate(test_images, test_targets, verbose = 0)
print('here')
y_pred=model.predict(test_images)
y_pred_arg=np.argmax(y_pred,axis=1)
#y_pred_arg = to_categorical(y_pred_arg)
print(test_targets)
print(y_pred_arg)
print(y_test)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_arg)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()



print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

print(model.summary())
'''def predict(image):
    x=[]
    for im in image:
        input = cv2.resize(image,(28,28)).reshape((28 , 28,1)).astype('float32') / 255.0
        x.append(model.predict_classes(np.array([input])))
    return x
pred=predict(test_images)
print(pred)
print(test_targets)'''