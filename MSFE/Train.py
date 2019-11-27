import numpy as np
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Flatten, concatenate, Input
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard

# Load Data
Train = np.loadtxt('Dataset_Train_2019_1126_n.csv', delimiter=',', skiprows=1)
print('Load Finish : Train')

Test = np.loadtxt('Dataset_Test_2019_1126_n.csv', delimiter=',', skiprows=1)
print('Load Finish : Test')

# Split
xTrain = Train[:, :8]
yTrain = Train[:, 8:]

xTest = Test[:, :8]
yTest = Test[:, 8:]

# Reshape
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

print(xTrain.shape)
print(yTrain.shape)
print(xTest.shape)
print(yTest.shape)


# Modeling
DropoutRate = 0.5
# Conv1
input1 = Input(shape=(8, 1))
conv1 = Conv1D(32, 1, activation='relu', padding='same')(input1)
conv1 = Dropout(DropoutRate)(conv1)
conv1 = Conv1D(32, 3, activation='relu', padding='same')(conv1)

# Conv3
conv3 = Conv1D(32, 3, activation='relu', padding='same')(input1)
conv3 = Dropout(DropoutRate)(conv3)
conv3 = Conv1D(32, 3, activation='relu', padding='same')(conv3)
conv3 = Dropout(DropoutRate)(conv3)
conv3 = Conv1D(32, 3, activation='relu', padding='same')(conv3)

# Conv5
conv5 = Conv1D(32, 5, activation='relu', padding='same')(input1)
conv5 = Dropout(DropoutRate)(conv5)
conv5 = Conv1D(32, 3, activation='relu', padding='same')(conv5)
conv5 = Dropout(DropoutRate)(conv5)
conv5 = Conv1D(32, 3, activation='relu', padding='same')(conv5)

# model
model = concatenate([conv1, conv3, conv5], axis=-1)

# LSTM Layer
model = LSTM(32, return_sequences=True)(model)
model = LSTM(32, return_sequences=False)(model)
output = Dense(3, activation='softmax')(model)

model = Model(input1, output)
model.summary()

# Optimizer = optimizers.Adam(lr = 0.00001)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


callback_list = [
	ReduceLROnPlateau(
		# Model의 Val_loss를 Monitoring
		monitor = 'val_loss',
		# Callback 호출시 Learning rate를 1/10으로 줄임
		factor=0.1,
		# Val_loss가 5 Epoch동안 개선되지 않을 경우 CallBack 호출
		patience=5
		),

	EarlyStopping(
		# Monitoring Index
		monitor = 'val_loss',
		# 5 Epoch보다 더 길게(즉 6 Epoch 동안) 정확도 향상이 없을 경우 Early Stop
		patience = 5
		)
]

# ReduceLR = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, mode='auto', min_lr=1e-05)
model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=50, batch_size=1000, callbacks=callback_list)

model.save('MSFE.h5')
