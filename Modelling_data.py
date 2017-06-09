import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy import spatial
from sklearn.metrics import mean_squared_error
from collections import Counter
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.naive_bayes import GaussianNB as GB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation

#Loading Data
job_info = pd.read_csv("job_tags_june_2017.csv", encoding='ISO-8859-1', error_bad_lines=False)
person_info = pd.read_csv("person_tags_june_2017.csv", encoding='ISO-8859-1', error_bad_lines=False)
similarity_score = pd.read_csv("processed_sim_scorev2.csv", encoding='utf-8', error_bad_lines=False)

#Getting unique Job & person id's
person_ids_unique = set(person_info.person_id.unique())
job_id_unique = set(job_info.job_id.unique())

#Filtering sim score data
df = similarity_score[similarity_score.person_id.isin(person_ids_unique)]
df = df[df.job_id.isin(job_id_unique)]

#get categories present both in job & person id
categories = set(person_info.category.unique()).intersection(set(job_info.category.unique()))
person_info = person_info[person_info.category.isin(categories)]
job_info = job_info[job_info.category.isin(categories)]

job_info = job_info.dropna()
person_info = person_info.dropna()

skill = list(set(job_info.name.unique()).union(set(person_info.name.unique())))
skills = list(set([i.lower() for i in skill]))

# To remove spaces in skills
skillss = [" ".join(i.split()) for i in skills]

job_id_dict = {}

#Combining name for person and jobs id's respectively
for index, row in job_info.iterrows():
    jID = row["job_id"]
    name = row["name"].lower()
    #name = " ".join(name.split())

    if jID not in job_id_dict.keys():
        job_id_dict[jID]=name
    else:
        i = job_id_dict[jID]
        i = i + " " + name
        job_id_dict[jID] = i

person_id_dict = {}
for index, row in person_info.iterrows():
    pID = row["person_id"]
    name = row["name"].lower()
    name = " ".join(name.split())
    if pID not in person_id_dict.keys():
        person_id_dict[pID]=name
    else:
        i = person_id_dict[pID]
        i = i + " " + name
        person_id_dict[pID] = i

cv = TfidfVectorizer(ngram_range=(1, 2))
cv.fit_transform(skillss)

#creating job and person feature vector 
person_vec_dict = {}
for i in person_id_dict:
    x = []
    x.append(person_id_dict[i])
    x = cv.transform(x).toarray()
    person_vec_dict[i] = x


job_vec_dict = {}
for i in job_id_dict:
    x = []
    x.append(job_id_dict[i])
    x = cv.transform(x).toarray()
    job_vec_dict[i] = x

pID = df["person_id"].tolist()
jID = df.job_id.tolist()

job_list = [job_vec_dict[i] for i in jID]
per_list = [person_vec_dict[i] for i in pID]

#new columns in same df
df["person_vec"] = per_list
df["job_vec"] = job_list

#Tried to use cosine sim as a feature initially
# Result = []
# for i,row in df.iterrows():
#     X = row["person_vec"]
#     Y = row["job_vec"]
#     result = 1 - spatial.distance.cosine(X, Y)
#     Result.append(result)

# df["cosine_sim"] = Result
# df["cosine_sim"] = df.cosine_sim.apply(lambda x: [x])

def flattenvector(list_of_lists):
    flattened = [val for sublist in list_of_lists for val in sublist]
    return flattened

df["flat_per"] = df.person_vec.apply(flattenvector)
df["flat_job"] = df.job_vec.apply(flattenvector)

# print (len(x), len(y))

#dividing data into train:test : 90:10
train, test = train_test_split(df, test_size = 0.10, random_state=5)
n = ["flat_per","flat_job"]
X_train = train.as_matrix(columns = n)
X_test = test.as_matrix(columns = n)
y_train = train.similarity_score.tolist()
y_test = test.similarity_score.tolist()

#flatterning train and test arrays
X_trainn = []
for i in X_train:
    list_of_lists = list(i)
    flattened = [val for sublist in list_of_lists for val in sublist]
    X_trainn.append(flattened)

X_testt = []
for i in X_test:
    list_of_lists = list(i)
    flattened = [val for sublist in list_of_lists for val in sublist]
    X_testt.append(flattened)

print (len(X_trainn), "training points")
print (len(X_testt), "testing points")

rus = RandomOverSampler()
print (Counter(y_train), "counts before over-sampling")
X_tr, y_tr = rus.fit_sample(X_trainn, y_train)
print (Counter(y_tr), "counts after over-sampling")

# models=["RandomForestClassifier", "Gaussian Naive Bays", "KNN", 
#         "Logistic_Regression", "Support_Vector", "Decision Trees"]
# Classification_models = [RandomForestClassifier(n_estimators=500, max_features='sqrt',
#                         random_state=5, n_jobs=-1), GB(),knn(n_neighbors=7, weights='uniform', algorithm='kd_tree'), 
#                         LogisticRegression(), SVC(), 
#                         tree.DecisionTreeClassifier()]
# Model_Accuracy = []
# confusion_matrix_list = []

# for clf in Classification_models:
    
#     clf.fit(X_tr, y_tr)
#     y_pred = clf.predict(X_testt)
#     acc = accuracy_score(y_test, y_pred)
#     Model_Accuracy.append(acc)
#     #print("\n Accuracy score of : ",acc)
#     confusion_matrix_list.append(confusion_matrix(y_test, y_pred))

# Accuracy_with_all_features = pd.DataFrame(
#     {"Accuracy with all features" : Model_Accuracy,
#      "Classification Model" : models,
#      "confusion_matrix" : confusion_matrix_list
     
#     })

# aaa= Accuracy_with_all_features.sort_values(by="Accuracy with all features",ascending=False).reset_index(drop=True)
# print (aaa)

rfc_parameters = {'n_estimators':[50,100,200],  'max_depth': [5, 10, 20, 40],  'min_samples_leaf': [2, 4, 6],
                 'min_samples_split': [2, 4], 'n_jobs':[-1]}
rfc = RandomForestClassifier()
clf = GridSearchCV(rfc, rfc_parameters)
clf.fit(X_tr, y_tr)

print ("The best accuracy for Random Forest classifier is: " + str(clf.best_score_))
print ("The best hyper-parameters for Random Forest classifier are: " + str(clf.best_params_))
'''
The best hyper-parameters for Random Forest classifier are: 
{'max_depth': 40, 'min_samples_split': 4, 'n_estimators': 50, 'n_jobs': -1, 'min_samples_leaf': 2}
'''
y_pred = clf.predict(X_testt)
acc = accuracy_score(y_test, y_pred)
print("\n Accuracy score  : ",acc)
print (confusion_matrix(y_test, y_pred))

#Gives us score for unseen data
print("Grid score function: ", clf.score(X_testt,y_test))

#Training model again with best parameters to cross verify the score!!
print("Train model again with best parameters: ")
clf2 = RandomForestClassifier(**clf.best_params_)
print("Model trained")
clf2.fit(X_tr, y_tr)
y_pred2 = clf2.predict(X_testt)
acc2 = accuracy_score(y_test, y_pred2)
print("\n Accuracy score  2 : ",acc2)
print (confusion_matrix(y_test, y_pred2))


