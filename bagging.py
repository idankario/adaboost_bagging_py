#This program predicts if a passenger will survive on the titanic

#Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

#create model base on desicion tree deep 2
def get_desicion_tree(s):
    #Split the data into undependent 'x' and dependent 'y'
    X = s.iloc[:, 4:7].values
    Y = s.iloc[:, 7].values
    stump = DecisionTreeClassifier(criterion="entropy", max_depth=2)
    #Build a decision tree classifier from the training set (X, y). retun self : DecisionTreeClassifier Fitted estimator.
    return stump.fit(X, Y)

#Crate sample model
def crate_models(data,m,models):   
    # 63.2% Has proven himself on average to have a good sample for esembler algorithem
    p=0.632
    for i in range(m):
        # Get random of 63.2% sample from data random state to get diffrent random for each
        t=data.sample(frac=p, random_state=i)
        # Get random of 36.8% sample duplicate of our sample
        subsample = t.sample(n=len(data)-len(t), replace=True)
        # Combine the two data set together
        s= pd.concat([t, subsample])
        # Combine the current model
        models.append(get_desicion_tree(s))
#Change atribute to numeric number
def change_table_val_from_category_to_numeric(data,num):
    labelEncoder = LabelEncoder()
    # Convert string column to int and append new coulmn
    for i in range(num):
        data["is_"+data.columns[i]] = labelEncoder.fit_transform(data.iloc[:,i].values)
        
def majority_key(k):
    u, c = np.unique(k, return_counts=True)
    return u[c.argmax()]

def prediction(models,dt):
    #list of all prediction of our model
    p = list()
    for m in models:
        #Predict class or regression value for X. For a classification model, the predicted class for each sample in X is returned. For a regression model, the predicted value based on X is returned.
        p.append(m.predict(dt.iloc[:, 4:7].values))
    #Combine the m resulting models and give majority key inside
    majority = np.apply_along_axis(majority_key, 0, np.array(p))
    #Change data from numeric to string
    result_test=np.sum(majority == dt.is_survived) *100/len(majority)
    print("Success {}%".format(result_test))
    return majority

def begin(num_m,num_f):
    #Load the data from file
    d = pd.read_csv("titanikData.csv")
    dt = pd.read_csv('titanikTest.csv', names=["pclass", "age", "gender", "survived"])
    change_table_val_from_category_to_numeric(d,num_f)
    change_table_val_from_category_to_numeric(dt,num_f)
    models =  list()
    #becouse of the bias in the training is better to get rid of duplicate to get better model
    crate_models(d.drop_duplicates(),num_m,models)
    p=prediction(models,dt)
    #Drop all numeric coulmn 
    dt = dt.iloc[:, :4]
    dt['prediction']=p
    #Change data from numeric to string
    dt['prediction'] = np.where(dt['prediction'] == 1, 'yes', 'no')
    dt.to_csv('titanik_bagging_reult.csv', index=False)
    print(dt)
    
if __name__ == "__main__":
    #adaBoost model =100 feture=4
    begin(100,4) 