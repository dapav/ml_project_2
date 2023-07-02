import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

#Διάβασμα των δεδομένων από το αρχείο Excle
file_path = 'C:\\Users\\Davor\\Desktop\\ml_p2.xlsx'
df = pd.read_excel(file_path)
rename_dict={
    "365* ( Β.Υ / Κοστ.Πωλ )": "sales_cost_365",
    "Λειτ.Αποτ/Συν.Ενεργ. (ROA)": "roa",
    "ΧΡΗΜ.ΔΑΠΑΝΕΣ / ΠΩΛΗΣΕΙΣ": "cost_to_sales_ratio",
    " ΠΡΑΓΜΑΤΙΚΗ ΡΕΥΣΤΟΤΗΤΑ :  (ΚΕ-ΑΠΟΘΕΜΑΤΑ) / Β.Υ": "liquidity",
    "(ΑΠΑΙΤ.*365) / ΠΩΛ.": "receivable_to_sales_ratio",
    "Συν.Υποχρ/Συν.Ενεργ": "payable_to_assets_ratio",
    "Διάρκεια Παραμονής Αποθεμάτων": "inventory_duration",
    "Λογαριθμος Προσωπικού": "num_employees_log",
    "ΕΝΔΕΙΞΗ ΕΞΑΓΩΓΩΝ": "exports_flag",
    "ΕΝΔΕΙΞΗ ΕΙΣΑΓΩΓΩΝ": "imports_flag",
    "ΕΝΔΕΙΞΗ ΑΝΤΙΠΡΟΣΩΠΕΙΩΝ": "repr_flag",
    "ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)": "bankrupt",
    "ΕΤΟΣ": "year",
}
df=df.rename(columns=rename_dict)

#Εμφάνιση αριθμού υγιών και χρεωκοπημένων επιχ ανα έτος

grouped = df.groupby('year')
# for year, group in grouped:
#     print("year",year)
#     print("Υγιείς επιχ:", group['bankrupt'].value_counts().get(1))
#     print("Χρεωκοπημένες επιχειρήσεις:", group['bankrupt'].value_counts().get(2))

df=df.drop(['year'], axis=1)


#Εκτύπωση των min, max και avg τιμών για κάθε δίκτη πέραν του bankrupt και του year
# for column in df.columns:
#     min_value = df[column].min()
#     max_value = df[column].max()
#     avg_value = (df[column].sum())/len(df[column])
#     print(f"Min-Max-AVG'{column}':'{'{:.2f}'.format(min_value)}"
#           f"-{'{:.2f}'.format(max_value)}-{'{:.2f}'.format(avg_value)}")

#Κανονικοποίηση των δεδομένων
values = df.values
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(values)
normalized_df = pd.DataFrame(normalized_data, columns=df.columns)

#Χωρισμός των δεδομένων
y = normalized_df['bankrupt'].values
x = normalized_df.drop(['bankrupt'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

train_healthy_count = len(y_train[y_train==0])
train_bankrupt_count = len(y_train[y_train == 1])
test_healthy_count = len(y_test[y_test == 0])
test_bankrupt_count = len(y_test[y_test == 1])
print(train_healthy_count, train_bankrupt_count, test_healthy_count, test_bankrupt_count)

# x_train_grouped = x_train.groupby('bankrupt')
# x_test_grouped = x_test.groupby('bankrupt')

# print("Υγιείς επιχ στο train-set και στο test:", x_train_grouped['bankrupt'].value_counts().get(1),
#       x_test_grouped['bankrupt'].value_counts().get(1))
# print("Χρεωκ εταιριες στο train-set και στο test:", x_train_grouped['bankrupt'].value_counts().get(2),
#       x_test_grouped['bankrupt'].value_counts().get(2))

models = {
    'Decision Trees': DecisionTreeClassifier(),
    'k_Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes':GaussianNB(),
    'Support Vector Machines':SVC()
}

#Λιστα για την αποθήκευση των δεδομένων
results = []
classnames=['not-bankrupt','bankrupt']
#Create an empty figure
fig,axes = plt.subplots(nrows=4,ncols=2, figsize=(13,15))
axes = axes.flatten()
plt.subplots_adjust(wspace=0.8)
for i, (name, model) in enumerate(models.items()):
    #Fit the model and make predictions
    model.fit(x_train, y_train)
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    # print("Model:", name)
    # print("Confusion Matrix(Train):")
    # print(confusion_matrix(y_train, train_predictions))
    # print("Confusion Matrix(Test):")
    # print(confusion_matrix(y_test, test_predictions))
    # print()

    train_cm = confusion_matrix(y_train, train_predictions)
    test_cm = confusion_matrix(y_test, test_predictions)


    #Train Set
    axes[i].imshow(train_cm, cmap='Blues')
    axes[i].set_title(f'{name}')
    tick_marks=np.arange(len(classnames))
    axes[i].set_xticks(tick_marks)
    axes[i].set_xticklabels(classnames)
    axes[i].set_yticks(tick_marks)
    axes[i].set_yticklabels(classnames)



    thresh = train_cm.max() / 2
    for j, k in itertools.product(range(train_cm.shape[0]), range(train_cm.shape[1])):
        axes[i].text(k, j, f'{train_cm[j, k]}', horizontalalignment='center', color='white' if train_cm[j, k] > thresh else 'black')




    axes[i+4].imshow(test_cm, cmap='Greens')
    axes[i+4].set_title(f'{name}')
    tick_marks=np.arange(len(classnames))
    axes[i+4].set_xticks(tick_marks)
    axes[i+4].set_xticklabels(classnames)
    axes[i+4].set_yticks(tick_marks)
    axes[i+4].set_yticklabels(classnames)

    thresh = test_cm.max() / 2
    for j, k in itertools.product(range(test_cm.shape[0]), range(test_cm.shape[1])):
        axes[i+4].text(k, j, f'{test_cm[j, k]}', horizontalalignment='center', color='white' if test_cm[j, k] > thresh else 'black')





    #Υπολογισμος των μετρικών για το train set
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_precision = precision_score(y_train, train_predictions, zero_division=1)
    train_recall = recall_score(y_train, train_predictions)
    train_f1 = f1_score(y_train, train_predictions)
    train_tp = confusion_matrix(y_train, train_predictions)[1, 1]
    train_tn = confusion_matrix(y_train, train_predictions)[0, 0]
    train_fp = confusion_matrix(y_train, train_predictions)[0, 1]
    train_fn = confusion_matrix(y_train, train_predictions)[1, 0]



    #Υπολογισμος των μετρικών για το test set
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(y_test, test_predictions,zero_division=1)
    test_recall = recall_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions)
    test_tp = confusion_matrix(y_test, test_predictions)[1, 1]
    test_tn = confusion_matrix(y_test, test_predictions)[0, 0]
    test_fp = confusion_matrix(y_test, test_predictions)[0, 1]
    test_fn = confusion_matrix(y_test, test_predictions)[1, 0]

    #Προσθήκη των αποτελεσμάτων στην λιστα
    results.append([name, 'Train', len(x_train),
                    train_bankrupt_count,train_tp, train_tn,
                    train_fp, train_fn, train_precision, train_recall,
                    train_f1, train_accuracy])

    results.append([name, 'Test', len(x_test),
                    test_bankrupt_count, test_tp, test_tn,
                    test_fp, test_fn, test_precision, test_recall,
                    test_f1, test_accuracy])




plt.tight_layout
plt.show()
#Εγγραφή των αποτελεσμάτων σε ένα αρχείο CSV
with open('results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Classifier Name', 'Training or test set', 'Number of training samples', 'Number of non-healthy companies in training sample',
                     'TP', 'TN', 'FP', 'FN', 'Precision', 'Recall', 'F1 score', 'Accuracy'])

    writer.writerows(results)



























