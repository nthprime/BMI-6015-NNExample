import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

X = np.array([[.05, .10]], dtype=np.float32)
y = np.array([[.01, .99]], dtype=np.float32)

l = [
        [[.15, .25], [.20, .30]],
        [.35, .35],
        [[.40, .50], [.45, .55]],
        [.60, .60]
    ]

model = Sequential()
model.add(Dense(units=2,
                input_dim=2,
                activation='sigmoid'
                ))
model.set_weights(l)
model.add(Dense(units=2,
                activation='sigmoid'
                ))
model.set_weights(l)
model.compile(loss='mean_squared_error', optimizer=SGD(lr=.5))


#filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]


print(model.get_weights())
history = model.fit(X, y, nb_epoch=100)
history_dict = history.history
model.summary()
print(model.get_weights())


loss_values = history_dict['loss']
epochs = range(1, len(loss_values)+1)
plt.plot(epochs, loss_values, 'b', label='Training loss')
plt.title("Training and validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



#ann_viz(model, title="My first neural network", filename='hw4.1.gv')
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

'''
[array([[0.14978072, 0.24975115],
       [0.19956143, 0.2995023 ]], dtype=float32), array([0.3456143 , 0.34502286], dtype=float32),
 array([[0.3589165 , 0.5113013 ],
       [0.40866616, 0.56137013]], dtype=float32), array([0.53075075, 0.61904913], dtype=float32)]
'''

