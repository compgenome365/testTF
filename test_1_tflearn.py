import tflearn

data, labels = tflearn.data_utils.load_csv('f3.csv', target_column=60,
                        categorical_labels=True, n_classes=2)



# Build neural network
net = tflearn.input_data(shape=[None, 60])
net = tflearn.fully_connected(net, 500)
net = tflearn.fully_connected(net, 500)
net = tflearn.fully_connected(net, 2, activation='tanh')
net = tflearn.regression(net,optimizer='rmsprop')

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

pred = model.predict(data)

print pred
