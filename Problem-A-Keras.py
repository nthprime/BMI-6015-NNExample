import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

X = np.array([[1, 3, 7]], dtype=float)
X = np.array([[0.142857143,	0.428571429,	1]], dtype=float)


y = np.array([[1, 0, 0]], dtype=float)

l = [
        [[.2, .3, .5], [.3, .5, .7], [.6, .4, .8]],
        [1., 1., 1.],
        [[.1, .4, .8], [.3, .7, .2], [.5, .2, .9]],
        [1., 1., 1.]
    ]

model = Sequential()
model.add(Dense(units=3,
                input_dim=3,
                activation='sigmoid'
                ))
model.set_weights(l)
model.add(Dense(units=3,
                activation='softmax'
                ))
model.set_weights(l)
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=1))


#filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]

print(model.get_weights())
history = model.fit(X, y, nb_epoch=1)
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

