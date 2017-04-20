El presente paper, describe un metodo mejorado en Deep Learning para la reidentificacion de personas. 
Framework utilizado:  Keras-Tensorflow. Theano para la optimizacion en GPU.
Se adjunta los pesos para la prueba. (modelkeras_final_w10.h5).
Dataset utilizado: CUHK01

Salida
C:\Users\Lenovo\Miniconda3\python.exe "C:/Users/Lenovo/Desktop/MAESTRIA/Sistemas inteligentes/market/keras_load_prueba_reid.py"
Using TensorFlow backend.
(1000, 3, 160, 60)
(100, 3, 160, 60)
Size of:
- Training-set:		1000
- Test-set:		100
ini model
cross_input_shape: None   25   200   75
fin model
(3, 160, 60)
(3, 160, 60)
total imagenes:  100
probabilidad:  [[ 0.81679676]]

..............

Epoch 1997/2000
5468/5468 [==============================] - 125s - loss: 0.0175 - acc: 0.9916 - val_loss: 0.3259 - val_acc: 0.9376
Epoch 1998/2000
5468/5468 [==============================] - 125s - loss: 0.0141 - acc: 0.9894 - val_loss: 0.3277 - val_acc: 0.9485
Epoch 1999/2000
5468/5468 [==============================] - 125s - loss: 0.0155 - acc: 0.9905 - val_loss: 0.3132 - val_acc: 0.9429
Epoch 2000/2000
5468/5468 [==============================] - 124s - loss: 0.0174 - acc: 0.9914 - val_loss: 0.3180 - val_acc: 0.9371
