# Importing Important Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importing Data Preprocessing Libraries
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Importing the Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Importing the Dataset
df = pd.read_csv('hotel_bookings.csv')

#----------------------------------------Data Preprocessing and Data Cleaning----------------------------------------

# Dropping Columns with High Amount of presence of Missing Values
for col in df.columns:
    per = df[col].isnull().sum() / df[col].shape[0]
    if per > 0.3:
        df = df.drop(col, axis=1)

# Since Country and Children have less than 1% of missing values so deleting rows would be a better choice instead of doing imputation for such a small amount of missing data
df = df.dropna(subset=['country'])
df = df.dropna(subset=['children'])

# Encoding Arrival Date Month Column
enc = {
    'arrival_date_month' : {'January' : 1, 'February' : 2, 'March' : 3, 'April' : 4, 'May' : 5, 'June' : 6, 'July' : 7, 'August' : 8, 'September' : 9, 'October' : 10, 'November' : 11, 'December' : 12}
}

# Replacing Categorical Value with Encoded Value
df = df.replace(enc)

# Replacing Remaining Categorical Feature Values on the basis of One Hot Encoder Criterion
df = pd.get_dummies(data=df, columns=[col for col in df.columns if df[col].dtype == 'O'])

#----------------------------------------Data Preprocessing and Data Cleaning----------------------------------------

#----------------------------------------Predicting whether or not the person is cancelling the booking----------------------------------------

# Separating Features and Class from the Dataset
X = df.drop('is_canceled', axis=1).values
y = df['is_canceled'].values

# Imputing Agent Column Values with Mean
imputer = Imputer(missing_values='NaN', strategy='mean')
X[:, 14:15] = imputer.fit_transform(X[:, 14:15])

# Splitting the dataset into Training Set and Testing Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#----------------------------------------Model Building and Training----------------------------------------

# Defining Lists to store the performance of Different Classifiers
classifiers = ['Decision Tree Classifier', 'Logistic Regression', 'Random Forest Classifier']
scores = []

# Training with Decision Tree Classifier
clf1 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)
y_pred = clf1.predict(X_test)
score = accuracy_score(y_test, y_pred)
scores.append(score)

# Training with Logistic Regression
clf2 = LogisticRegression()
clf2.fit(X_train, y_train)
y_pred = clf2.predict(X_test)
score = accuracy_score(y_test, y_pred)
scores.append(score)

# Training with Random Forest Classifier
clf3 = RandomForestClassifier(n_estimators=20)
clf3.fit(X_train, y_train)
y_pred = clf3.predict(X_test)
score = accuracy_score(y_test, y_pred)
scores.append(score)

#----------------------------------------Model Building and Training----------------------------------------

#----------------------------------------Predicting whether or not the person is cancelling the booking----------------------------------------

#----------------------------------------Model Evaluation----------------------------------------

# Evaluating Performance of all the Classifiers
sns.barplot(x=scores, y=classifiers)
plt.xlabel('Accuracy Score')
plt.ylabel('Classifier')
plt.title('Classifier Performance')
plt.show()

# Since All the Classifiers are doing great so we can proceed with anyone

#----------------------------------------Model Evaluation----------------------------------------