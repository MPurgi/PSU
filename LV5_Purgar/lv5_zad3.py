import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

df = pd.read_csv("occupancy_processed.csv")

#print(df.head())
#X ulazni i y izlazni podaci
X = df[['S3_Temp', 'S5_CO2']]
y = df['Room_Occupancy_Count']

#podjela na train i test, stratify je da odrzava omjer 0 i 1
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
#fit uci kako sklairati, transofrm primjenjuje skaliranje
X_train = scaler.fit_transform(X_train)
#na testu se ne uci nego samo primjenjujemo
X_test = scaler.transform(X_test)


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)


y_pred = dt.predict(X_test)


print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))

plt.figure(figsize=(12, 8))
plot_tree(dt, filled=True, feature_names=['Temperature', 'CO2'])
plt.title("Decision Tree")
plt.show()