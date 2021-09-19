import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif,mutual_info_classif
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from sklearn import preprocessing

#%%
y = pd.read_csv("y_train.csv")
y = y.drop(['id'],axis=1)
X = pd.read_csv("X_train.csv")
X = X.drop(['id'],axis=1)
X_imp = pd.read_csv("X_test.csv")
X_test = X_imp.drop(['id'],axis=1)

#%%
#Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

#%%
#Feature Selection:
#Hyperparameters: k(200 perfors best in lda), classifier(f_classif or mutual_info_classif(takes 1min, but gives 1% better lda))
X_new = SelectKBest(f_classif,k=200).fit_transform(X_train,np.ravel(y_train))

#%%
#Outlier Detection
scaler = preprocessing.StandardScaler().fit(X_new)
X_scaled = scaler.transform(X_new)
pca = PCA(n_components=2)
pca.fit(X_scaled)
pca_X = pca.transform(X_scaled)
plt.scatter(pca_X[:,0],pca_X[:,1])
plt.show()
outliers = pca_X[:,0] > 25
outliers2 = pca_X[:,1] > 7.5
outliers3 = pca_X[:,1] < -7.5
totoutliers = outliers + outliers2 + outliers3
totoutliers = totoutliers*1
X_train_noout = X_train[totoutliers == 0]
y_train = y_train[totoutliers==0]

#%%
#Rerun Feature Selection without outlier (Same hyperparamters as above!)
selection = SelectKBest(mutual_info_classif,k=200).fit(X_train_noout,np.ravel(y_train))
X_train = selection.transform(X_train_noout)
X_val = selection.transform(X_val)
X_test = selection.transform(X_test)

#%%
#Scaling data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#%%
#Prediction:
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel,ConstantKernel, RBF
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import RidgeClassifier

#%%LDA:  0.67
lda = LinearDiscriminantAnalysis(solver="lsqr", store_covariance=True,shrinkage=0.88)
y_pred = lda.fit(X_train, y_train).predict(X_val)
y_test = lda.predict(X_test)

#%%QDA: 0.45
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
y_pred = qda.fit(X_train, y_train).predict(X_val)

#%%SVM: 71% (poly:66%,rbf:71%(->C=1),sigmoid:55%)
param_grid = {'C': [1], 
              'gamma': ["auto"],
              'coef0':[0]}
svc = SVC(kernel='rbf', random_state=0, max_iter=100000,class_weight="balanced")
grid = GridSearchCV(svc, param_grid=param_grid, cv=5, n_jobs=-1,scoring="balanced_accuracy")
poly = grid.fit(X_train, y_train)
print('Best CV accuracy: {:.4f}'.format(poly.best_score_))
print('Validation CV score:       {:.4f}'.format(poly.score(X_val, y_val)))
print('Best parameters: {}'.format(poly.best_params_))
#poly.cv_results_

y_pred = grid.predict(X_val)
y_test = grid.predict(X_test)

#%%RF:63%(min_samples_split:100,max_depth=50)
param_grid = {'min_samples_split': [10,50,100,150], 
              'max_depth': [25,50,100],
              'max_features': ["auto"],
              'n_estimators': [200]
              }
rf = RandomForestClassifier(random_state=0,class_weight="balanced")
grid = GridSearchCV(rf, param_grid=param_grid, cv=5, n_jobs=-1,scoring="balanced_accuracy")
poly = grid.fit(X_train, y_train)
print('Best CV accuracy: {:.4f}'.format(poly.best_score_))
print('Validation CV score:       {:.4f}'.format(poly.score(X_val, y_val)))
print('Best parameters: {}'.format(poly.best_params_))
#poly.cv_results_

y_pred = grid.predict(X_val)
y_test = grid.predict(X_test)

#%%NNet: Relu:56%, logistic:61%(hidden layer=40,max_iter=200),tanh=60%
param_grid = {"hidden_layer_sizes" : [25,40,50,100],
              'max_iter': [100,200,300,400],
              'activation':['tanh'],
              'alpha':[0.0001]}
mlp = MLPClassifier(random_state=0, warm_start= True)
grid = GridSearchCV(mlp, param_grid=param_grid, cv=5, n_jobs=-1,scoring="balanced_accuracy")
poly = grid.fit(X_train, y_train)
print('Best CV accuracy: {:.4f}'.format(poly.best_score_))
print('Validation CV score:       {:.4f}'.format(poly.score(X_val, y_val)))
print('Best parameters: {}'.format(poly.best_params_))
#poly.cv_results_

y_pred = grid.predict(X_val)
y_test = grid.predict(X_test)


#%%GP
kernel = RBF(length_scale=0.5)
gpc = GaussianProcessClassifier(kernel=kernel,random_state=0,max_iter_predict=10,warm_start=1).fit(X_train, np.ravel(y_train))
y_pred = gpc.predict(X_val)

#%%LogReg 57%
param_grid = {'penalty': ['elasticnet'], #elastic nets combines l1&l2
              'C':[0.01,0.05,0.07,0.1],
              'l1_ratio':[0,0.1]} #if 0, or 1 then l2 or l1 would be best. If between then the combination of both
lor = LogisticRegression(max_iter=100, tol=0.001,random_state=1, n_jobs=-1,solver='saga',warm_start=True) #increasing iterations to 1000 increases score by only 1% -> it is not worth the additional time
grid = GridSearchCV(lor, param_grid=param_grid, cv=5, n_jobs=-1,scoring="balanced_accuracy")
poly = grid.fit(X_train, y_train)
print('Best CV accuracy: {:.4f}'.format(poly.best_score_))
print('Validation CV score:       {:.4f}'.format(poly.score(X_val, y_val)))
print('Best parameters: {}'.format(poly.best_params_))

#%%
#Just do pipeline with stacked regressor to find hyperparamters
#%%Stacked Regressor 65%
estimators = [
  #   ('lr', RidgeCV()),
   #  ('svr', LinearSVR(random_state=42)),
     ('lda', LinearDiscriminantAnalysis(solver="lsqr", store_covariance=True,shrinkage=0.88)),
     ('svc', SVC(kernel='rbf', random_state=0, max_iter=100000,class_weight="balanced",C=1,gamma="auto")),
     ('rf', RandomForestClassifier(random_state=0,class_weight="balanced",min_samples_split=100,max_depth=50,n_estimators=200,max_features="auto")),
     ('mlp', MLPClassifier(random_state=0, warm_start= True,hidden_layer_sizes=40,max_iter=500,activation="tanh",alpha=0.0001))
     
     ]
reg = StackingClassifier(estimators=estimators,final_estimator=RidgeClassifier(alpha=0.1,random_state=0,class_weight="balanced"))
result = reg.fit(X_train, y_train)
print('Validation CV score:       {:.4f}'.format(result.score(X_val, y_val)))
y_pred = result.predict(X_val)
y_test = result.predict(X_test)


#%%
#Evaluation
from sklearn.metrics import balanced_accuracy_score
BMAC = balanced_accuracy_score(y_val, y_pred)
print(BMAC)


#%%
y_handin = X_imp['id']
y_handin = pd.DataFrame(y_handin)
y_handin[1] =  y_test
y_handin.columns = ['id', 'y']
y_handin.to_csv(r'y_test.csv', index = False)

