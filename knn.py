import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# Read adults.txt
data = pd.read_csv('adults.txt', sep=',')

X = data[['race', 'occupation','education','sex']]
print("***The train (before convert the string labels to numeric labels)***")
print(X)
print()
# Convert the string labels to numeric labels
for label in ['race', 'occupation','education','sex']:
    data[label] = LabelEncoder().fit_transform(data[label])

# Take the fields that interest us and put them into  X
X = data[['race', 'occupation','education','sex']]
# Take the field that we want to learn/know and put him into Y
Y = data['salary'].values.tolist()

# Split the data into test and train (50% for each)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
print("***The train (after convert the string labels to numeric labels)***")
print(X_train)
print()
# Init the wanted classifier
clf = KNeighborsClassifier(n_neighbors=3)

# Train the classifier using the train data
clf = clf.fit(X_train, Y_train)

# accuracy and prediction
accuracy = clf.score(X_test, Y_test)
print ("***KNN Accuracy: " + str(accuracy)+"***")
prediction = clf.predict(X_test)
print()
print("***Classification report***")
print(classification_report(Y_test, prediction))

# Make a confusion matrix
cm = confusion_matrix(prediction, Y_test)
print ("Confusion matrix: ")
count=0
for i in cm:
    # print(i)
    for j in i:
        if (count == 0):
            print("True Positive = ", j)
        if (count == 1):
            print("False Positive = ", j)
        if (count == 2):
            print("False Negative = ", j)
        if (count == 3):
            print("True Negative = ", j)
        count=count+1