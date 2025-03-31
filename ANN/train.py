# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 21:51:14 2025

@author: koray
"""
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model #modelin geri yüklenmesi

from tensorflow.keras import backend as K


import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)  # Tahminleri 0 veya 1'e yuvarla
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)  # True Positive
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)  # False Positive
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)  # False Negative

    precision = tp / (tp + fp + K.epsilon())  # Kesinlik (Precision)
    recall = tp / (tp + fn + K.epsilon())  # Duyarlılık (Recall)

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())  # F1-score
    return K.mean(f1)  # F1'in ortalaması alınır

#%%
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_row_shape=x_train.shape[1]
x_col_shape=x_train.shape[2]
'''
plt.figure(figsize=(10,5))

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_train[i],cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")
    
plt.show
'''
#reshape and scaling
x_train=x_train.reshape((x_train.shape[0]),x_train.shape[1]*x_train.shape[2]).astype("float32")/255 #60000,784 yaptık 2d oldu
x_test=x_test.reshape((x_test.shape[0]),x_test.shape[1]*x_test.shape[2]).astype("float32")/255


#One hot encoding
y_train=to_categorical(y_train,len(np.unique(y_train)))
y_test=to_categorical(y_test,len(np.unique(y_test)))

#%%
model=Sequential()
#ilk katman
model.add(Dense(512,activation="relu",input_shape=(x_row_shape*x_col_shape,))) # 0 dan pozitif değerlere taşı daha hızlıdır
#ikinci katman
model.add(Dense(256,activation="tanh")) #tanh -1 ile 1 arasına sıkıştırır

#output layer 10 tane olmak zorunda
model.add(Dense(10,activation="softmax"))

model.summary()

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=[
              'accuracy',f1_score]
            )
#%%
#monitor demek val_loss u izler.
early_stopping=EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)

#en iyi modelin ağırlıklarını kaydeder.save_best_only ise en iyi performans gösteren modeli kaydeder
checkpoint=ModelCheckpoint("ann_best_model.h5",monitor="val_loss",save_best_only=True)


#%%
history=model.fit(x_train,y_train,
          epochs=10,
          batch_size=60,
          validation_split=0.2, #eğitim verisinin %20 si doğrulama verisi olarak kullanılıcak
          callbacks=[early_stopping,checkpoint])

history_dict = history.history
#%%
results=model.evaluate(x_test,y_test)

print(f"Test loss: {results[0]},Test accuracy: {results[1]},f1_score: {results[2]}")

plt.figure()
plt.plot(history.history["accuracy"],marker="o",label="Training Accuracy")# burada y verildi sadece x otomatik olarak oluşturdu accuracy den kaç tane varsa
plt.plot(history.history["val_accuracy"],marker="o",label="Validation Accuracy")
plt.plot(history.history["f1_score"],marker="o",label="Training f1_score")
plt.plot(history.history["val_f1_score"],marker="o",label="Validation f1_score")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()
plt.grid(True)
plt.show()
#%%

plt.figure()
plt.plot(history.history["loss"],marker="o",label="Traning loss")
plt.plot(history.history["val_loss"],marker="o",label="Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()
plt.grid(True)
plt.show()
#%%
model.save("model/final_minst_model.h5")
