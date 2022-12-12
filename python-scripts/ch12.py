#!/usr/bin/env python
# coding: utf-8

# In[1]:


#! ipython suppress id=379207b8b8f44b8c8e923bccddb74e1b
get_ipython().run_line_magic('pushd', 'book-materials')
import numpy as np
import pandas as pd
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_columns = 20
pd.options.display.max_rows = 20
pd.options.display.max_colwidth = 80
np.set_printoptions(precision=4, suppress=True)


# In[2]:


#! ipython id=6b91a5404025497cb8caca631d473d94
data = pd.DataFrame({
    'x0': [1, 2, 3, 4, 5],
    'x1': [0.01, -0.01, 0.25, -4.1, 0.],
    'y': [-1.5, 0., 3.6, 1.3, -2.]})
data
data.columns
data.to_numpy()


# In[3]:


#! ipython id=bd1bb62fd48f4d1d8c0fa538775d33fa
df2 = pd.DataFrame(data.to_numpy(), columns=['one', 'two', 'three'])
df2


# In[4]:


#! ipython id=eb72464446b24feab231e6d154a13d68
df3 = data.copy()
df3['strings'] = ['a', 'b', 'c', 'd', 'e']
df3
df3.to_numpy()


# In[5]:


#! ipython id=0284bd81d9734d62a0d2dcb8fc65e935
model_cols = ['x0', 'x1']
data.loc[:, model_cols].to_numpy()


# In[6]:


#! ipython id=7b7d579912624331ac4bd2bf08470533
data['category'] = pd.Categorical(['a', 'b', 'a', 'a', 'b'],
                                  categories=['a', 'b'])
data


# In[7]:


#! ipython id=361dec779b57482b94be56e2d062e793
dummies = pd.get_dummies(data.category, prefix='category')
data_with_dummies = data.drop('category', axis=1).join(dummies)
data_with_dummies


# In[8]:


#! ipython id=0f377070f0d14b769232b62284a278f4
data = pd.DataFrame({
    'x0': [1, 2, 3, 4, 5],
    'x1': [0.01, -0.01, 0.25, -4.1, 0.],
    'y': [-1.5, 0., 3.6, 1.3, -2.]})
data
import patsy
y, X = patsy.dmatrices('y ~ x0 + x1', data)


# In[9]:


#! ipython id=5252acd3c6db4a3d9af0ebbb309c28c1
y
X


# In[10]:


#! ipython id=03adb0f4a90247f6aba11e6763b806d0
np.asarray(y)
np.asarray(X)


# In[11]:


#! ipython id=a3a80cb56c54468b9e26c4cb052b5f6a
patsy.dmatrices('y ~ x0 + x1 + 0', data)[1]


# In[12]:


#! ipython id=db616433095b4c758545ec95c5ba56d4
coef, resid, _, _ = np.linalg.lstsq(X, y)


# In[13]:


#! ipython id=a02f213f2f10477b8b7907c83746bbc6
coef
coef = pd.Series(coef.squeeze(), index=X.design_info.column_names)
coef


# In[14]:


#! ipython id=0da24c4c7d374bb0b2cac75072a4dd04
y, X = patsy.dmatrices('y ~ x0 + np.log(np.abs(x1) + 1)', data)
X


# In[15]:


#! ipython id=eed36a4a619c4d9f9d65c0d8cb19d34e
y, X = patsy.dmatrices('y ~ standardize(x0) + center(x1)', data)
X


# In[16]:


#! ipython id=88739eb5eaa94ab0bb19779111888158
new_data = pd.DataFrame({
    'x0': [6, 7, 8, 9],
    'x1': [3.1, -0.5, 0, 2.3],
    'y': [1, 2, 3, 4]})
new_X = patsy.build_design_matrices([X.design_info], new_data)
new_X


# In[17]:


#! ipython id=766fcf4390d24922a9f4cbbf86da6870
y, X = patsy.dmatrices('y ~ I(x0 + x1)', data)
X


# In[18]:


#! ipython id=5fa5d310160a4cd7a5862986a6be9c3d
data = pd.DataFrame({
    'key1': ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'b'],
    'key2': [0, 1, 0, 1, 0, 1, 0, 0],
    'v1': [1, 2, 3, 4, 5, 6, 7, 8],
    'v2': [-1, 0, 2.5, -0.5, 4.0, -1.2, 0.2, -1.7]
})
y, X = patsy.dmatrices('v2 ~ key1', data)
X


# In[19]:


#! ipython id=d795962e76ab48f7bbd08fa3adc3f481
y, X = patsy.dmatrices('v2 ~ key1 + 0', data)
X


# In[20]:


#! ipython id=2619ef96053c43eeb4f0fbaa5aed3836
y, X = patsy.dmatrices('v2 ~ C(key2)', data)
X


# In[21]:


#! ipython id=6efd1f19da93449d8ae591df85a826c9
data['key2'] = data['key2'].map({0: 'zero', 1: 'one'})
data
y, X = patsy.dmatrices('v2 ~ key1 + key2', data)
X
y, X = patsy.dmatrices('v2 ~ key1 + key2 + key1:key2', data)
X


# In[22]:


#! ipython verbatim id=2c6c970ee0234156b7dea4ebb8b16790
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[23]:


#! ipython verbatim id=15d197f427b344fa9f456841b129b9e4
# To make the example reproducible
rng = np.random.default_rng(seed=12345)

def dnorm(mean, variance, size=1):
    if isinstance(size, int):
        size = size,
    return mean + np.sqrt(variance) * rng.standard_normal(*size)

N = 100
X = np.c_[dnorm(0, 0.4, size=N),
          dnorm(0, 0.6, size=N),
          dnorm(0, 0.2, size=N)]
eps = dnorm(0, 0.1, size=N)
beta = [0.1, 0.3, 0.5]

y = np.dot(X, beta) + eps


# In[24]:


#! ipython id=81c61bc21b6c409aa2aaa99a3f575bfb
X[:5]
y[:5]


# In[25]:


#! ipython id=8ae5538884bb4d05bff507be6a9c3d53
X_model = sm.add_constant(X)
X_model[:5]


# In[26]:


#! ipython id=5ee479f40b0d49f1b8725a3ff80d5794
model = sm.OLS(y, X)


# In[27]:


#! ipython id=dab4d3edbde744b0ae331d4fac14a62b
results = model.fit()
results.params


# In[28]:


#! ipython id=f6cacdd9108340b7b7888be9cccb8c66
print(results.summary())


# In[29]:


#! ipython id=e7dc7a51f95240d690847760a4bce30e
data = pd.DataFrame(X, columns=['col0', 'col1', 'col2'])
data['y'] = y
data[:5]


# In[30]:


#! ipython id=44341d4bc2984b1f9eefdf0518c00b32
results = smf.ols('y ~ col0 + col1 + col2', data=data).fit()
results.params
results.tvalues


# In[31]:


#! ipython id=c1e9fd6361f747fcbdd7054fdea0b3e0
results.predict(data[:5])


# In[32]:


#! ipython verbatim id=271bfcc9f59a4d2096cc41fdcf9f7fef
init_x = 4

values = [init_x, init_x]
N = 1000

b0 = 0.8
b1 = -0.4
noise = dnorm(0, 0.1, N)
for i in range(N):
    new_x = values[-1] * b0 + values[-2] * b1 + noise[i]
    values.append(new_x)


# In[33]:


#! ipython id=e576aacedffa4dbab3be7517065e0b78
from statsmodels.tsa.ar_model import AutoReg
MAXLAGS = 5
model = AutoReg(values, MAXLAGS)
results = model.fit()


# In[34]:


#! ipython id=a43500bc16854e43977c0a4c5fc2a2de
results.params


# In[35]:


#! ipython id=9e54440f129e4624889185d002432371
train = pd.read_csv('datasets/titanic/train.csv')
test = pd.read_csv('datasets/titanic/test.csv')
train.head(4)


# In[36]:


#! ipython id=ea68d69bba434e0c9665c794accee8d8
train.isna().sum()
test.isna().sum()


# In[37]:


#! ipython id=ca1197b7e36e4af8bb0c7087960dca2e
impute_value = train['Age'].median()
train['Age'] = train['Age'].fillna(impute_value)
test['Age'] = test['Age'].fillna(impute_value)


# In[38]:


#! ipython id=52d11da1850c463783b6e088f0f662d3
train['IsFemale'] = (train['Sex'] == 'female').astype(int)
test['IsFemale'] = (test['Sex'] == 'female').astype(int)


# In[39]:


#! ipython id=450811815c5f40d0af45346441c2e3c2
predictors = ['Pclass', 'IsFemale', 'Age']

X_train = train[predictors].to_numpy()
X_test = test[predictors].to_numpy()
y_train = train['Survived'].to_numpy()
X_train[:5]
y_train[:5]


# In[40]:


#! ipython id=8a75f2aaf25b449b9347d5a71a1fa643
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()


# In[41]:


#! ipython id=d15acba4d6674ea58746c2258c28bf44
model.fit(X_train, y_train)


# In[42]:


#! ipython id=27541b6b54c54a37b01dcb07921952b2
y_predict = model.predict(X_test)
y_predict[:10]


# In[43]:


#! ipython id=8a33d87df19f49ec9e62bcb26d88b3b7
from sklearn.linear_model import LogisticRegressionCV
model_cv = LogisticRegressionCV(Cs=10)
model_cv.fit(X_train, y_train)


# In[44]:


#! ipython id=36245c79ec40498f8bcb84258c1020cd
from sklearn.model_selection import cross_val_score
model = LogisticRegression(C=10)
scores = cross_val_score(model, X_train, y_train, cv=4)
scores


# In[45]:


#! ipython suppress id=ce2a8a40e5074d149cd991bc1062381c
get_ipython().run_line_magic('popd', '')


# In[46]:


#! ipython suppress id=de44cca014a54e88a10784f41350688a
pd.options.display.max_rows = PREVIOUS_MAX_ROWS

