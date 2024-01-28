#this is the start of the project
#import the dataset and read it.
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('C:/Users/GHADAH/Desktop/PROJECT/Diabetes.csv')
print(df) #print the dataframe 
df.info()

#to check if there is any missing value
print(df.isnull().any())
#to assign all columns/features without the label to x 
X = df.drop(['Outcome'], axis=1)
#to assign the last column/label to y
y = df.Outcome

print("X (features):")
print(X) # this will print the features 

print("\ny (label):")
print(y) # this will print the label

#to check if the data set is balanced or imbalanced
class_distribution = df['Outcome'].value_counts()
print("\nClass distribution: ")
print(class_distribution)

#to get percentages:
class_percentage = df['Outcome'].value_counts(normalize=True) * 100
print("\nClass percentage: ")
print(class_percentage)

#to split the data into training set and a testing set
#test size is going to be 20% and the training set is going to be 80%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#Naive bayes model
#Import Gaussian Naive Bayes from sklearn
#Will build the model and train it.
from sklearn.naive_bayes import GaussianNB
model = GaussianNB() #create a model
model.fit(X_train, y_train); #train the model

#this is the model evaluation
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
   ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)

y_pred = model.predict(X_test) #this will predict on test data

#Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")

# print the model evaluation
print("\nModel evaluation for Naive Bayes: ")
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision score:",precision)
print("Recall score:",recall)

#ROC curve for naive bayes
probabilities = model.predict_proba(X_test)
pred_prob = probabilities[:, 1]  # probabilities for class 1 which is positive 
fpr, tpr, thresholds = roc_curve(y_test,pred_prob) #fpr (false positive) tpr (true positive) threshold (calculate the FPR and TPR)
auc_value = roc_auc_score(y_test,pred_prob,average = 'weighted')
plt.plot(fpr, tpr, color='green', label=f'ROC curve (area = {auc_value:0.2f})') 
plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Chance (area = 0.50)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
print('\nROC:', auc_value)
plt.legend()
plt.show()

#this is the confusion matrix
#based on the results, this model is the best out of the three models
#at identifying class 1
labels = ["Class 0", "Class 1"]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
print("Confusion matrix results\n",cm)
plt.show()
#this is the end of the naive bayes model

#Decision tree model
#will build the model and train it
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
#hyperparameters-tuning
param_grid_dt = {   
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
    }
dt_model = DecisionTreeClassifier(random_state=42)
grid_search_dt = GridSearchCV(dt_model, param_grid_dt, cv=5, scoring='accuracy')  #5 fold cross-validation is used here
grid_search_dt.fit(X_train, y_train)
best_dt_model = grid_search_dt.best_estimator_
print("\nBest parameters for decision tree:", grid_search_dt.best_params_)

#this is the model evaluation
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_curve,
    roc_auc_score,
)

dt_pred = best_dt_model.predict(X_test) #this will predict on test data

# Calculate evaluation metrics
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred, average="weighted")
dt_precision = precision_score(y_test,dt_pred, average="weighted")
dt_recall = recall_score(y_test, dt_pred, average="weighted")

#print the model evaluation
print("\nModel evaluation for Decision tree:")
print("Accuracy:", dt_accuracy)
print("F1 Score:", dt_f1)
print("Precision score:", dt_precision)
print("Recall score:", dt_recall)

#ROC curve for decision tree
dt_probabilities = best_dt_model.predict_proba(X_test)
dt_pred_prob = dt_probabilities[:, 1]  # probabilities for class 1 which is positive 
fpr, tpr, thresholds = roc_curve(y_test, dt_pred_prob)#fpr (false positive) tpr (true positive) threshold (calculate the FPR and TPR)
dt_auc_value = roc_auc_score(y_test, dt_pred_prob)
plt.plot(fpr, tpr, color='green', label=f'ROC curve (area = {dt_auc_value:0.2f})')  
plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Chance (area = 0.50)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
print('\nROC:', dt_auc_value)
plt.legend()
plt.show()

#this is the confusion matrix
dt_labels =["Class 0", "Class 1"]
dt_cm = confusion_matrix(y_test, dt_pred)
dt_disp = ConfusionMatrixDisplay(confusion_matrix=dt_cm, display_labels=dt_labels)
dt_disp.plot()
print("Confusion matrix results\n",dt_cm)
plt.show()
#this is the end of decision tree

#this is the start of the random forest
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
#hyperparameters-tuning 
param_grid_rf = {
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_model = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='accuracy')  #5 fold cross-validation is used here
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_
print("\nBest parameters for random forest:", grid_search_rf.best_params_)

#model evaluation
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_curve,
    roc_auc_score,
)
rf_pred = best_rf_model.predict(X_test) #this will predict on test data
rf_accuracy = accuracy_score(y_test,rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average="weighted")
rf_precision = precision_score(y_test,rf_pred, average="weighted")
rf_recall = recall_score(y_test, rf_pred, average="weighted")

#print the model evaluation
print("\nModel evaluation for Random forest:")
print("Accuracy:", rf_accuracy)
print("F1 Score:", rf_f1)
print("Precision score:", rf_precision)
print("Recall score:", rf_recall)

#ROC curve for the random forest
rf_probabilities = best_rf_model.predict_proba(X_test)
rf_pred_prob = rf_probabilities[:, 1]  # probabilities for class 1 which is positive 
fpr, tpr, thresholds = roc_curve(y_test, rf_pred_prob)#fpr (false positive) tpr (true positive) threshold (calculate the FPR and TPR)
rf_auc_value = roc_auc_score(y_test, rf_pred_prob)
plt.plot(fpr, tpr, color='green', label=f'ROC curve (area = {rf_auc_value:0.2f})')  
plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Chance (area = 0.50)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
print('\nROC: ', rf_auc_value)
plt.legend()
plt.show()

#this is the confusion matrix
rf_labels =["Class 0", "Class 1"]
rf_cm = confusion_matrix(y_test, rf_pred)
rf_disp = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=rf_labels)
rf_disp.plot()
print("Confusion matrix results\n",rf_cm)
plt.show()
#this is the end of random forest classifier






