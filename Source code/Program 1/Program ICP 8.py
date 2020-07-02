import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

# fix random seed for reproducibility
seed = 155
numpy.random.seed(seed)

# load  dataset
dataset = pd.read_csv("C:/Users/Devna Chaturvedi/Desktop/ICP 8/DeepLearning_Lesson1/diabetes.csv", header=None).values
data=pd.DataFrame(dataset) #data is panda but dataset is something else
print(data.head())

# split into input (X ie dependent variables) and output (Y ie independent variables) variables
X = dataset[:,0:8]   #0-8 columns are dependent variables - remember 8th column is not included
Y = dataset[:,8]     #8 column is independent variable

# create model
my_first_nn = Sequential()
# my_first_nn.add(Dense(1000, input_dim=8, init='uniform', activation='relu')) # 1000 neurons
# my_first_nn.add(Dense(100, init='uniform', activation='tanh'))
my_first_nn.add(Dense(500, init='uniform', activation='relu')) # 500 neurons

my_first_nn.add(Dense(1, init='uniform', activation='sigmoid')) # 1 output neuron

# Compile model
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
my_first_nn.fit(X, Y, nb_epoch=150, batch_size=10,  verbose=2) # 150 epoch, 10 batch size, verbose = 2

# evaluate the model
scores = my_first_nn.evaluate(X, Y)
print("%s: %.2f%%" % (my_first_nn.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = my_first_nn.predict(X)    # predicting Y only using X
print(predictions)

# Round predictions
rounded = [int(numpy.round(x, 0)) for x in predictions]
print(rounded)

print("Rounded type: ", type(rounded)) # rounded is a 'list' class
print("Shape of rounded: ", len(rounded))
print("Dataset type: ", type(dataset)) # numpy array?
print("Shape of dataset: ", dataset.shape)

# Turn rounded from a 'list' class into a numpy array
newRounded = numpy.array(rounded)
print("Rounded type: ", type(newRounded))

# Add the rounded numpy array (called newRounded) to the end of the dataset numpy array
newDataset = numpy.column_stack((dataset, newRounded))


df_confusion = pd.crosstab(Y, newRounded, rownames=['Actual'], colnames=['Predicted'], margins=True)
df_conf_norm = df_confusion / df_confusion.sum(axis=1)

print(df_confusion)
print(df_conf_norm)