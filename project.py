import numpy as np
data = np.genfromtxt ('data.csv', delimiter =',')
X, y = data [:,:-1], data [:,-1]
countof1 = np.argwhere(y==1)
countof0 = np.argwhere(y==0)
print(countof1.shape)
print(countof0.shape)
print(y.shape)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
scaler.fit(data)
X, y = data [:,:-1], data [:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn import linear_model
clf = linear_model.Lasso(alpha=1,max_iter=10)
clf=clf.fit(X,y)
print(clf.coef_)
from sklearn.decomposition import PCA
pca=PCA().fit(X,y)
import seaborn as sns
correlation_matrix = np.corrcoef(data, rowvar=False)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
covariance_matrix = np.cov(data, rowvar=False)
sns.heatmap(covariance_matrix, annot=True, fmt=".2f", cmap='coolwarm')
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(random_state=1, max_iter=1000).fit(X_train, y_train)
clf.score(X_test,y_test)
print(clf.score)