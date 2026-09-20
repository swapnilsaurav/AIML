import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = pd.read_csv("data\\data_weather.csv")
print("Data: \n",data.head())
print("Columns in Dataframe: ", data.columns)
del data["number"]
data["relative_humidity_3pm"] = (data["relative_humidity_3pm"] > 24.99)*1
print("Data at 3pm: \n", data["relative_humidity_3pm"])
data = data.dropna()
X=data.iloc[:,0:6].values #has to multi D
Y=data.iloc[:,9].values  #has to be single dimensional

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
classifier = DecisionTreeClassifier()
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
output_df = pd.DataFrame({"Actual": Y_test, "Predicted": y_pred})
print("Output Dataframe: \n",output_df)
print("Accuracy of the Model: ",accuracy_score(Y_test, y_pred))
