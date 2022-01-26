import pandas as pd
from Dataset import Dataset

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential

dt = Dataset()
x_train, x_test, y_train, y_test = dt.get_train_test_dataset()

model = Sequential()
model.add(Input(shape=x_train.shape[1]))
model.add(Dense(35, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=15)

predictions = tf.round(model.predict(dt.test_dt).flatten())
dt = pd.read_csv('../datasets/test.csv')
dt = dt.drop(['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
dt['Survived'] = predictions
print(dt)

dt.to_csv('./final_predictions.csv')