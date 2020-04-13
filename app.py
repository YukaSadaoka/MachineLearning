import pandas
import tensorflow
import keras
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import numpy as np
import pickle
from matplotlib import style, pyplot

# Read data from csv file
data = pandas.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

best_accuracy = 0

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
for x in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    # Training process starts....
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    # Get accuracy
    # This accuracy is a bit different everytime because it's trained
    accuracy = linear.score(x_test, y_test)
    print(f"accuracy: {accuracy}")
    # ### Training process ends ###
    if accuracy > best_accuracy:
        # Create pickle and load it into linear to save the data trained
        best_accuracy = accuracy
        with open("studentmodel.pickle", 'wb') as f:
            pickle.dump(linear, f)

# Loading pickle file(saved model)
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print(f"Coefficient: {linear.coef_}")
print(f"Intercept: {linear.intercept_}")
# make a prediction
predictions = linear.predict(y_test)

for index in range(len(predictions)):
    print(predictions[index],x_test[index], y_test[index])

p = 'G1'
style.use('ggplot')
pyplot.scatter(data[p], data['G3'])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
