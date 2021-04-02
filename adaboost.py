#This program predicts if a passenger will survive on the titanic

#Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

#create model base on desicion tree deep 2
def get_desicion_tree(X, Y,s):
    stump = DecisionTreeClassifier(criterion="entropy", max_depth=2)
    #Build a decision tree classifier from the training set (X, y). retun self : DecisionTreeClassifier Fitted estimator.
    return stump.fit(X, Y, sample_weight=np.array(s['weights']))
#Change atribute to numeric number
def change_table_val_from_category_to_numeric(data,num):
    labelEncoder = LabelEncoder()
    # Convert string column to int and append new coulmn
    for i in range(num):
        data["is_"+data.columns[i]] = labelEncoder.fit_transform(data.iloc[:,i].values)
def prediction(h,a,dt):
    #list of all prediction of our model
    p = list()
    for alpha, hypothesis in zip(a, h):
        #Predict class or regression value for X. For a classification model, the predicted class for each sample in X is returned. For a regression model, the predicted value based on X is returned.
        p.append(alpha*hypothesis.predict(dt.iloc[:, 4:7].values)) 
    #Change data from numeric to string
    is_survived = np.sign(np.sum(np.array(p), axis=0))
    #Change data from numeric to string
    result_test=np.sum(is_survived == dt.is_survived) *100/len(is_survived)
    print("Success {}%".format(result_test))
    #return predictions to survived base our model
    return is_survived

#Crate sample model
def crate_models(d,hypothesises,alphas,i):   
    #Initilize the weights to be equal
    d['weights'] = 1 / len(d)
    #Split the data into undependent 'x' and dependent 'y'
    X = d.iloc[:, 4:7].values
    Y = d.iloc[:, 7].values
    for i in range(i):
        # Create the current model
        hypothesises.append(get_desicion_tree(X, Y,d))
        # Get list of predictions
        predictions=hypothesises[i].predict(X)
        predictions=(Y != predictions)
        # Error = sum of weights of misclassified samples
        error = sum(d['weights'][predictions])
        # Check if error boost algoritem is ok
        if error > 0.5:
            break
        # calculate beta
        beta=error / (1 - error)
        # calculate alpha
        alpha = 0.5 * np.log(1 / beta)
        alphas.append(alpha)
        #weight calculation
        d['weights'] *= np.exp(alpha * Y*predictions)
        # Normalize to one
        d['weights'] /= np.sum(d['weights'])

def adaBoost(i,num_f):
    #Load the data from file
    d = pd.read_csv("titanikData.csv")
    dt = pd.read_csv('titanikTest.csv', names=["pclass", "age", "gender", "survived"])
    change_table_val_from_category_to_numeric(d,num_f)
    change_table_val_from_category_to_numeric(dt,num_f)
    hypothesises=  list()
    alphas=  list()
    crate_models(d,hypothesises,alphas,i)
    #Print grade to the model in percent and return predictions
    p=prediction(hypothesises,alphas,dt)
    #Drop all numeric coulmn 
    dt = dt.iloc[:, :4]
    dt['prediction']=p
    #Change data from numeric to string
    dt['prediction'] = np.where(dt['prediction'] == 1, 'yes', 'no')
    dt.to_csv('titanik_adaBoost_result.csv', index=False)
    print(dt)

if __name__ == '__main__':
     #adaBoost itration =3 feture=4
    adaBoost(3,4)
    