
# coding: utf-8

# # Analysis in Medicare Provider Utilization and Payment Data: From the Prospectives of Average Difference between Submitted and Charged Medicare Amount from Physician in California

# ## Data Input

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


data = pd.read_csv("medical_insurance.csv")
data.head()


# In[3]:


data.info()


# In[4]:


# subset dataset to only Californa data
data_ca = data.loc[data['State Code of the Provider']=='CA']
display(data_ca.shape)


# We can see that number of samples are reduced from 9+ millions to 700 thousands. Let's start to dive more into the dataset

# ## Exploratory Data Analysis

# ### Missing Value

# Since there's small portion of missing values in the remaining columns, these missing values are just dropped out. 

# In[5]:


# function to calculate missing value percentage
def get_missing_percentage(column):
    num = column.isnull().sum()
    total_n = len(column)
    return round(num/total_n, 2)


# In[6]:


data_ca_drop = data_ca.drop(['Middle Initial of the Provider', 'National Provider Identifier','First Name of the Provider','Credentials of the Provider','Street Address 2 of the Provider','State Code of the Provider','Country Code of the Provider'], axis=1)


# In[7]:


print (data_ca['Gender of the Provider'].describe())


# we can see that gender in male accounts for almost 74% of the data. It can be inferred from the fact that generally there are more male physicians (provider) than female physicians. 

# In[8]:


data_ca_drop_null = data_ca_drop[data_ca_drop.columns[data_ca_drop.isnull().any()].tolist()]
get_missing_percentage(data_ca_drop_null)


# Since there's only missing value in gender of the provider with missing at around 6%, therefore missing values were dropped. 

# In[9]:


data_ca_drop = data_ca_drop.dropna()
data_ca_drop.isnull().sum()


# Now we don't have any missing value in the dataframe, let's look at more details in this data. First, let's create a column to show the difference between "Avergae submitted Charged Amount" and "Average Medicare Allowed Amount"

# In[10]:


data_ca_drop['Average Medicare Difference'] = data_ca_drop['Average Submitted Charge Amount']- data_ca_drop['Average Medicare Allowed Amount'] 


# ### Subset dataframe based on Provider Type with Top 5 highest Average Medicare Difference  

# Let's focus on the top 5 provider types that have the most average medicare difference. 

# In[11]:


data_pt_amaa = data_ca_drop.groupby('Provider Type')['Average Medicare Difference'].mean().reset_index().sort_values('Average Medicare Difference', ascending = False).head(5)
data_pt_amaa


# In[12]:


provider_list = ['Thoracic Surgery','Neurosurgery','Cardiac Surgery', 'Vascular Surgery', 'CRNA']
data_ca_drop_pro = data_ca_drop.loc[data_ca_drop['Provider Type'].isin(provider_list)]

# group by provider type and plot bar plot
data_ca_drop_pro.groupby('Provider Type')['Average Submitted Charge Amount','Average Medicare Allowed Amount'].mean().sort_values(by = 'Average Submitted Charge Amount', ascending = False).plot(kind = 'bar')



# In[13]:


data_ca_drop_pro.groupby('Provider Type')['Average Medicare Difference'].mean().sort_values(ascending = False)


# In[14]:


### Top 5 Average Medicare Difference Cities


# We may also curious about what would average medicare difference varies among cities, here, we extract top 5 cities that have most average medicare difference:

# In[15]:


data_ca_drop_pro.groupby('City of the Provider')['Average Medicare Difference'].mean().sort_values(ascending = False).head(5).plot(kind = 'bar', color = 'skyblue')
data_ca_drop_pro.groupby('City of the Provider')['Average Medicare Difference'].mean().sort_values(ascending = False).head(5)


# We can see that West Hills has the most difference in average medicare amount, with around $4500 Now, let's subset dataframe based on top 5 provider types.

# ### ZIPCODE
# This section of code is referred from:
# https://www.christianpeccei.com/zipmap/

# In[16]:


def read_ascii_boundary(filestem):
    '''
    Reads polygon data from an ASCII boundary file.
    Returns a dictionary with polygon IDs for keys. The value for each
    key is another dictionary with three keys:
    'name' - the name of the polygon
    'polygon' - list of (longitude, latitude) pairs defining the main
    polygon boundary
    'exclusions' - list of lists of (lon, lat) pairs for any exclusions in
    the main polygon
    '''
    metadata_file = filestem + 'a.dat'
    data_file = filestem + '.dat'
    # Read metadata
    lines = [line.strip().strip('"') for line in open(metadata_file)]
    polygon_ids = lines[::6]
    polygon_names = lines[2::6]
    polygon_data = {}
    for polygon_id, polygon_name in zip(polygon_ids, polygon_names):
        # Initialize entry with name of polygon.
        # In this case the polygon_name will be the 5-digit ZIP code.
        polygon_data[polygon_id] = {'name': polygon_name}
    del polygon_data['0']
    # Read lon and lat.
    f = open(data_file)
    for line in f:
        fields = line.split()
        if len(fields) == 3:
            # Initialize new polygon
            polygon_id = fields[0]
            polygon_data[polygon_id]['polygon'] = []
            polygon_data[polygon_id]['exclusions'] = []
        elif len(fields) == 1:
            # -99999 denotes the start of a new sub-polygon
            if fields[0] == '-99999':
                polygon_data[polygon_id]['exclusions'].append([])
        else:
            # Add lon/lat pair to main polygon or exclusion
            lon = float(fields[0])
            lat = float(fields[1])
            if polygon_data[polygon_id]['exclusions']:
                polygon_data[polygon_id]['exclusions'][-1].append((lon, lat))
            else:
                polygon_data[polygon_id]['polygon'].append((lon, lat))
    return polygon_data


# In[17]:


#data_ca_drop_pro['Zip Code of the Provider'] = data_ca_drop_pro['Zip Code of the Provider'].astype('str')
data_ca_drop_pro['Zip Code of the Provider'] = data_ca_drop_pro['Zip Code of the Provider'].apply(lambda x: str(x)[:5])


# In[18]:


# group by Zip Code of the Provider
provider_zipcode = data_ca_drop_pro.groupby('Zip Code of the Provider')['Average Medicare Difference'].mean().sort_values(ascending = False)


# In[19]:


from pylab import *
avg_med_diff = {}

# Add data for each ZIP code
for i in range(provider_zipcode.shape[0]):
    avg_med_diff[provider_zipcode.index[i]] = provider_zipcode[i]
max_avg_med_diff = max(avg_med_diff.values())


# In[20]:


# Read in ZIP code boundaries for California
d = read_ascii_boundary('zip5/zt06_d00')


# In[21]:


# Create figure and two axes: one to hold the map and one to hold
# the colorbar
figure(figsize=(15, 15), dpi=70)
map_axis = axes([0.0, 0.0, 0.8, 0.9])
cb_axis = axes([0.83, 0.1, 0.03, 0.6])

# Define colormap to color the ZIP codes.
# You can try changing this to cm.Blues or any other colormap
# to get a different effect
cmap = cm.GnBu

# Create the map axis
axes(map_axis)
axis([-125, -114, 32, 42.5])
gca().set_axis_off()

# Loop over the ZIP codes in the boundary file
for polygon_id in d:
    polygon_data = array(d[polygon_id]['polygon'])
    zipcode = d[polygon_id]['name']
    avg_med_diff_mean = avg_med_diff[zipcode] if zipcode in avg_med_diff else 0.
    
    # Define the color for the ZIP code
    fc = cmap(avg_med_diff_mean/max_avg_med_diff)
    
    # Draw the ZIP code
    patch = Polygon(array(polygon_data), facecolor=fc,
        edgecolor=(.3, .3, .3, 1), linewidth=.4)
    gca().add_patch(patch)
title('Average Medciare Difference per ZIP Code in California (2014)')

# Draw colorbar
cb = mpl.colorbar.ColorbarBase(cb_axis, cmap=cmap,
    norm = mpl.colors.Normalize(vmin=0, vmax=max_avg_med_diff))
cb.set_label('Average Medciare Difference')


# In[22]:


# group by city of the provider and zipcode of the provider
data_ca_drop_pro.groupby(['City of the Provider','Zip Code of the Provider'])['Average Medicare Difference'].mean().sort_values(ascending = False).head(5)


# ### Procedures in each Provider Type

# In[23]:


#subset dataframe based on provider type
data_crna = data_ca_drop_pro.loc[data_ca_drop_pro['Provider Type']=='CRNA']
data_vas = data_ca_drop_pro.loc[data_ca_drop_pro['Provider Type']=='Vascular Surgery']
data_cardiac = data_ca_drop_pro.loc[data_ca_drop_pro['Provider Type']=='Cardiac Surgery']
data_thora = data_ca_drop_pro.loc[data_ca_drop_pro['Provider Type']=='Thoracic Surgery']
data_neuro = data_ca_drop_pro.loc[data_ca_drop_pro['Provider Type']=='Neurosurgery']


# ### Top 10 Procedures in CRNA

# In[24]:


data_crna.groupby('HCPCS Description')['Average Medicare Difference'].mean().sort_values(ascending = False).head(10).plot(kind ="barh")
data_crna.groupby('HCPCS Description')['Average Medicare Difference'].mean().sort_values(ascending = False).head(10)


# It seems that anesthesia for procedures on hear and great blood vessel cost has the most differences in average medicare amount, which is around $5000 USD. The procedure includes heart-lung usage, re-operation after original procedures. Further details about the frequency of these re-operating procedures and risk factors that contirbute to the re-operation can be discussed. But these questions will be left opened in the project.

# ### Top 10 Vascular Surgery

# In[25]:


data_vas.groupby('HCPCS Description')['Average Medicare Difference'].mean().sort_values(ascending = False).head(10).plot(kind ="barh")
data_vas.groupby('HCPCS Description')['Average Medicare Difference'].mean().sort_values(ascending = False).head(10)


# It seems that cost in removal of plaque and insection of stents into artery has the most differences in average medicare amount, which is $23700 USD, followed by the procedures involving removal of plaque and insection of stents into arteriers. We can see that removel of plaque and insection of stents account for top 4 average medicare amount difference procedures in Vascular Surgery. 

# ### Top 10 Cardiac Surgery 

# In[26]:


data_cardiac.groupby('HCPCS Description')['Average Medicare Difference'].mean().sort_values(ascending = False).head(10).plot(kind ="barh")
data_cardiac.groupby('HCPCS Description')['Average Medicare Difference'].mean().sort_values(ascending = False).head(10)


# The bar chart shows that insection of vena cava by endovascular approach has the most difference in average medicare amount in cardiac surgery, which is $30700 USD.

# ### Top 10 Thoracic Surgery 

# In[27]:


data_thora.groupby('HCPCS Description')['Average Medicare Difference'].mean().sort_values(ascending = False).head(10).plot(kind ="barh")
data_thora.groupby('HCPCS Description')['Average Medicare Difference'].mean().sort_values(ascending = False).head(10)


# The procedures that involves heart surgery shows the most difference between submitted charged amount and allowed medicare amount, which is around $17200 USD. 

# ### Top 10 Neurosurgery

# In[28]:


data_neuro.groupby('HCPCS Description')['Average Medicare Difference'].mean().sort_values(ascending = False).head(10).plot(kind ="barh")
data_neuro.groupby('HCPCS Description')['Average Medicare Difference'].mean().sort_values(ascending = False).head(10)


# The largest difference in neruosurgery is the procedure of repairing of bulging of blood vessel in brain, which is around 11520 USD.

# ## Feature Exploration

# ### Correlation Matrix

# To investigate what factors would contribute to major impact on the dependent variable - average medicare difference, and understand if independent variables have correlations, correlation matrix is applied. 

# In[29]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(data_ca_drop_pro.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
plt.savefig("corrmatirx.png")


# We can see that Average Medicare Allowed Amount, Average Submitted Charge Amount, Average Medicare Payment Amount,and Average Medicare Standardized Amount have higher correlation with Average Medicare Difference. We will focus on visualizing these features.

# ### Feature Transformation

# #### Average Medicare Difference

# In[30]:


plt.hist(pow(data_ca_drop_pro['Average Medicare Difference'], 1/6))


# #### Average Medicare Allowed Amount

# In[31]:


# log transformation of Average Medicare Allowed Amount
plt.hist(np.log(data_ca_drop_pro['Average Medicare Allowed Amount']))


# In[32]:


# copy df
data_pro_copy = data_ca_drop_pro.copy()
# log transformation of 
data_pro_copy['Log Average Medicare Allowed Amount'] = np.log(data_pro_copy['Average Medicare Allowed Amount'])
sns.lmplot(x = 'Log Average Medicare Allowed Amount', y='Average Medicare Difference', data = data_pro_copy, fit_reg = False, hue = 'Provider Type', legend = False)
plt.legend(loc='upper left')


# We could see there's more variations in vascular surgery. With more allowed average medicare amoumt, there's a increasing trend in variations in average medicare difference and the amount of average medicare difference.  

# #### Average Medicare Payment Amount

# In[33]:


plt.hist(np.log(data_ca_drop_pro['Average Medicare Payment Amount']))


# In[34]:


# scatter plot 
sns.lmplot(x = 'Average Medicare Payment Amount', y='Average Medicare Difference', data = data_ca_drop_pro, fit_reg = False, hue = 'Provider Type', legend = False)
plt.legend(loc='upper left')


# The scatter plot showed that vascular surgery varied the most in average medicare payment amount that medicare covered after coinsurance amount deducted compared to other surgery, and it can be visualized that there's slightly positive correlation between average medicare payment amount and average medicare difference.

# ### Creating Dummy Variables

# Now, I want to know that what factors would affect the average submitted charge amount. To do so, I need to get dummies for each categorical value to save space and ease computatinal complexity. 

# In[35]:


# convert categorical variable to dummy variable
data_ca_drop_dummy = pd.get_dummies(data_ca_drop_pro[['Zip Code of the Provider','Entity Type of the Provider','Provider Type','Medicare Participation Indicator','HCPCS Description','HCPCS Drug Indicator','City of the Provider']])


# In[36]:


# build continuous variable dataframe
data_ca_drop_continue = data_ca_drop_pro[['Number of Services','Number of Medicare Beneficiaries','Number of Distinct Medicare Beneficiary/Per Day Services','Average Medicare Payment Amount','Average Medicare Standardized Amount','Average Medicare Difference']]


# In[37]:


# normalize continuous dataframe
data_ca_drop_continue = (data_ca_drop_continue-data_ca_drop_continue.min())/(data_ca_drop_continue.max()-data_ca_drop_continue.min())


# In[38]:


# concat binary dataframe and continuous dataframe
data_ca_drop_dummy = pd.concat([data_ca_drop_continue,data_ca_drop_dummy], axis = 1)


# In[39]:


from sklearn.preprocessing import StandardScaler
# run lineaqr regression model
data_ca_drop_dummy.reset_index(drop = True)
Y = data_ca_drop_dummy['Average Medicare Difference']
X = data_ca_drop_dummy.drop(['Average Medicare Difference'], axis = 1)


# In[40]:


from sklearn.model_selection import train_test_split


# In[41]:


# split data into training set and testing set 
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# ## Linear Regression Model as a baseline model

# In[42]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score


# In[43]:


# train baseline model: linear regression
model = linear_model.LinearRegression()
model.fit(x_train, y_train)
y_pred_lr = model.predict(x_test)
print ("R Squared: ", model.score(x_test,y_test))
print("Residual sum of squares: %.2f"
              % np.mean((model.predict(x_test) - y_test) ** 2))
print ("Linear Regression MSE: %.4f"%mean_squared_error(y_test, y_pred_lr))


# ## Elastic Regression Model

# In[44]:


cv_enet = ElasticNetCV(l1_ratio = np.linspace(0.1,1,40.), cv = 10, eps = 0.001, n_alphas = 100, fit_intercept = True, normalize = True, max_iter = 2000)


# In[45]:


cv_enet.fit(x_train, y_train)


# In[46]:


print ("optimal l1_ratio: %.3f"%cv_enet.l1_ratio_)
print ("optimal alpha: %.6f"%cv_enet.alpha_)
print ("number of iterations : %d"%cv_enet.n_iter_)


# The l1 ratio is 0.1, which means ridge regression accounts majority part in the elastic net. This is reasonable because we may have collinearity and ridge solve collinearity issue better than lasso regressor does. 

# In[47]:


# train elastic net model 
net_model = ElasticNet(l1_ratio = cv_enet.l1_ratio_, alpha = cv_enet.alpha_, max_iter = cv_enet.n_iter_, fit_intercept = True, normalize = True)


# In[48]:


net_model.fit(x_train, y_train)


# In[49]:


# MSE
y_pred_net = net_model.predict(x_test)
print("Elastic Net MSE: {}".format(mean_squared_error(y_test, y_pred_net)))

# R squared
print ("Elastic Net R-Squared: {}".format(r2_score(y_test, y_pred_net)))


# In[50]:


plt.scatter(np.sqrt(y_test), np.sqrt(y_pred_net))
plt.xlabel("True Values")
plt.ylabel("Predictions")


# ### Feature Importance from Elastic Net

# In[51]:


def get_feature_importance(x_train, model):
    
    feature_importance = pd.Series(index = x_train.columns, data = np.abs(model.coef_))
    selected_features = (feature_importance>0).sum()
    print('{0:d} features, reduction of {1:2.2f}%'.format(
        selected_features,(1-selected_features/len(feature_importance))*100))
    feature_importance.sort_values().tail(10).plot(kind = 'barh', figsize = (20,8))


# In[52]:


get_feature_importance(x_train, net_model)


# # Random Forest Training and Prediction

# In[53]:


from sklearn.ensemble import RandomForestRegressor

# random forest model
rf = RandomForestRegressor(n_estimators = 200, min_samples_leaf = 2, min_samples_split = 15)

# train random forest regressor
rf_model = rf.fit(x_train, y_train)

# Use the forest's predict method on the test data
y_pred_rf = rf.predict(x_test)


# In[54]:


# MSE
print("Random Forest MSE: {}".format(mean_squared_error(y_test, y_pred_rf)))

# R squared
print ("Random Forest R-Squared: {}".format(r2_score(y_test, y_pred_rf)))


# ### Feature Importance from Random Forest

# In[55]:


# feature importance
feature_importance = pd.Series(index = x_train.columns, data = np.abs(rf_model.feature_importances_))
selected_features = (feature_importance>0).sum()
print('{0:d} features, reduction of {1:2.2f}%'.format(
    selected_features,(1-selected_features/len(feature_importance))*100))
feature_importance.sort_values().tail(10).plot(kind = 'barh', figsize = (20,8))


# In[56]:


# scatter plot 
plt.scatter(np.log(y_test), np.log(y_pred_rf))
plt.xlabel("True Values")
plt.ylabel("Predictions")


# # eXtreme Gradient Boosting

# In[57]:


#!pip install xgboost
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[58]:


print (x_train.shape)
print(y_train.shape)


# In[59]:


# train xgboosting regressor 
params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],
'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [2,3,4]}
xgb = XGBRegressor(colsample_bytree=0.2, gamma=0.0, 
                             learning_rate=0.05, max_depth=6, 
                             min_child_weight=1.5, n_estimators=200,
                             reg_alpha=0.9, reg_lambda=0.6,
                             subsample=0.2,seed=42)
xgb_grid = GridSearchCV(xgb, params)
xgb.fit(x_train,y_train)
y_pred_xgb = xgb.predict(x_test)


# In[60]:


# MSE
print("Xgboost MSE: {}".format(mean_squared_error(y_test, y_pred_xgb)))

# R squared
print ("Xgboost R-Squared: {}".format(r2_score(y_test, y_pred_xgb)))


# In[61]:


# feature importance 
feature_importance = pd.Series(index = x_train.columns, data = np.abs(xgb.feature_importances_))
selected_features = (feature_importance>0).sum()
print('{0:d} features, reduction of {1:2.2f}%'.format(
    selected_features,(1-selected_features/len(feature_importance))*100))
feature_importance.sort_values().tail(10).plot(kind = 'barh', figsize = (20,8))


# In[62]:


# plot prediciton values vs true values 
plt.scatter(np.sqrt(y_test), np.sqrt(y_pred_xgb))
plt.xlabel("True Values")
plt.ylabel("Predictions")


# In[63]:


# import MLP classifier
from sklearn.neural_network import MLPRegressor

# initialize MLP classifier
mlp = MLPRegressor(hidden_layer_sizes=(30,30,30))

# train MLP classifier 
mlp.fit(x_train,y_train)

# predict
y_pred_mlp = mlp.predict(x_test)


# In[64]:


# MSE
print("Neural Network MSE: {}".format(mean_squared_error(y_test, y_pred_mlp)))

# R squared
print ("Neural Network R-Squared: {}".format(r2_score(y_test, y_pred_mlp)))


# In[65]:


# plot prediciton values vs true values 
plt.scatter(np.sqrt(y_test), np.sqrt(y_pred_mlp))
plt.xlabel("True Values")
plt.ylabel("Predictions")


# # Stacking 

# In[116]:


from sklearn.model_selection import StratifiedKFold
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.cross_validation import KFold
import plotly.graph_objs as go
import scipy.optimize as spo
import plotly.offline as py
py.init_notebook_mode(connected=True)


# In[67]:


# x's dimension and y's dimension 
print (x_train.shape)
print(y_train.shape)


# In[68]:


ntrain = x_train.shape[0]
ntest = x_test.shape[0]

SEED = 43 
kf = KFold(ntrain, n_folds= 10, random_state=SEED)


# In[72]:


def get_train_test_per_model(clf, x_train, y_train, x_test):
    train = np.zeros((ntrain,))
    test = np.zeros((ntest,))
    test_skf = np.empty((10, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train.iloc[train_index]
        y_tr = y_train.iloc[train_index]
        x_te = x_train.iloc[test_index]

        clf.fit(x_tr, y_tr)

        train[test_index] = clf.predict(x_te)
        test_skf[i, :] = clf.predict(x_test)

    test[:] = test_skf.mean(axis=0)
    return train.reshape(-1, 1), test.reshape(-1, 1)


# In[73]:


# train all models
rf_train_stack, rf_test_stack = get_train_test_per_model(rf, x_train, y_train, x_test) # Random Forest
xgb_train_stack, xgb_test_stack = get_train_test_per_model(xgb,x_train, y_train, x_test) # Xgboost
mlp_train_stack, mlp_test_stack = get_train_test_per_model(mlp, x_train, y_train, x_test) # Neural Netork 


# In[74]:


predictions_train_stack = pd.DataFrame( {'RandomForest': rf_train_stack.ravel(),
      'GradientBoost': xgb_train_stack.ravel(),
      'NeuralNetowrk': mlp_train_stack.ravel()
    })


# In[75]:


# referred from: https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
heatmap = [
    go.Heatmap(
        z= predictions_train_stack.astype(float).corr().values ,
        x= predictions_train_stack.columns.values,
        y= predictions_train_stack.columns.values,
          colorscale='YlGnBu',
            showscale=True,
            reversescale = True
    )
]
py.iplot(heatmap, filename='labeled-heatmap')


# In[119]:


### Do combinations of models (stacking)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn import cross_validation

#define parameter for stacking 
model_list = ['XGBoost','MLP','RandomForest']
model_len = len(model_list)
opt_dict = {}

# Change df to np.array
X1 = X
X2 = X
X3 = X
y = Y

pred= np.zeros((3,9180))
pred_xgb = pd.Series([])
pred_mlp = pd.Series([])
pred_rf = pd.Series([])
# Do 10-fold train & evaluate
cv = cross_validation.KFold(len(x_train), n_folds=10, shuffle=False, random_state=None)

for traincv, testcv in cv:

    X_train1, X_test1= (X1.iloc[traincv], X1.iloc[testcv])
    X_train2, X_test2= (X2.iloc[traincv], X2.iloc[testcv])
    X_train3, X_test3 = (X3.iloc[traincv], X3.iloc[testcv])
    y_train1, y_test1= (y.iloc[traincv], y.iloc[testcv])

    # XGB
    xgb= XGBRegressor(colsample_bytree=0.2, gamma=0.0, 
                             learning_rate=0.05, max_depth=8, 
                             min_child_weight=1.5, n_estimators=200,
                             reg_alpha=0.9, reg_lambda=0.6,
                             subsample=0.2,seed=42)
    xgb.fit(X_train1,y_train1)
    pred_xgb_temp = xgb.predict(X_test1)
    pred_xgb = pd.concat([pred_xgb,pd.Series(pred_xgb_temp)], axis = 1)
    
    # MLP classifier
    mlp = MLPRegressor(hidden_layer_sizes=(30,30,30))
    mlp.fit(X_train2,y_train1)
    pred_mlp_temp = mlp.predict(X_test2)
    pred_mlp = pd.concat([pred_mlp,pd.Series(pred_mlp_temp)], axis = 1)
    
    #random forest regressor
    rf = RandomForestRegressor(n_estimators = 200, min_samples_leaf = 2, min_samples_split = 15)
    rf_model = rf.fit(X_train3, y_train1)
    pred_rf_temp = rf.predict(X_test1)
    pred_rf = pd.concat([pred_rf,pd.Series(pred_rf_temp)], axis = 1)
    
    
    


# In[123]:


#average result from cross validation by row
xgb_result = pred_xgb.mean(axis = 1)
mlp_result = pred_mlp.mean(axis = 1)
rf_result = pred_rf.mean(axis = 1)

#combine three model
pred_combine = pd.concat([xgb_result, mlp_result, rf_result], axis = 1)
pred_combine.columns = ['xgb','mlp','rf']



# In[151]:


def get_optimal_MSE(allocs,pred_df, eps=1e-15):
    pred_df = np.clip(pred_df, eps, 1 - eps)
    pred_df_w = np.sum(pred_df*allocs,axis = 1)
    return mean_squared_error(y_test1, pred_df_w)


# In[152]:


model_list = ['XGBoost','MLP','RandomForest']
model_len = len(model_list)
init_vals= [1.0 / model_len] * model_len
cons = ({'type': 'ineq', 'fun': lambda x: 1.0-np.sum(x)})
bnds = [(0.0, 1.0)] * model_len

#optimized allocations
opts = spo.minimize(get_optimal_MSE, init_vals, args = (pred_combine,),method='SLSQP', bounds=bnds, constraints=cons, options = {'disp':True})
opt_allocs = opts.x


# In[153]:


# optimal wieghts for each learner 
display(opt_allocs)

# optimal result 
display(opts)


# In[154]:


y_stack_pred = 0.33333333*mlp.predict(x_test) + 0.33333333*xgb.predict(x_test) + 0.33333333*rf.predict(x_test)


# In[155]:


# MSE
print("Stacking MSE: {}".format(mean_squared_error(y_test, y_stack_pred)))

# R squared
print ("Stacking R-Squared: {}".format(r2_score(y_test, y_stack_pred)))


# In[158]:


# plot prediciton values vs true values 
plt.scatter(np.log(y_test), np.log(y_stack_pred))
plt.xlabel("True Values")
plt.ylabel("Predictions")


# ## Comparison between Models

# In[162]:


# Mean Squared Error
model_mse = [ 
              mean_squared_error(y_test, y_pred_net),\
              mean_squared_error(y_test, y_pred_rf),\
              mean_squared_error(y_test, y_pred_xgb),\
              mean_squared_error(y_test, y_pred_mlp),\
              mean_squared_error(y_test, y_stack_pred)]
bar_cat = ('elastic net','random forest', 'XGB','MLP', 'stacking')
y_pos = np.arange(len(bar_cat))
 
# Create bars
plt.bar(y_pos, model_mse)
 
# Create names on the x-axis
plt.xticks(y_pos, bar_cat)

# Add title and axis names
plt.title('Mean Squared Error')
plt.xlabel('models')
plt.ylabel('values')
 
# Show graphic
plt.show()


# In[165]:


# R-Squared
model_r_squared = [ 
              r2_score(y_test, y_pred_net),\
              r2_score(y_test, y_pred_rf),\
              r2_score(y_test, y_pred_xgb),\
              r2_score(y_test, y_pred_mlp),\
              r2_score(y_test, y_stack_pred)]
bar_cat_r = ('elastic net','random forest', 'XGB','MLP', 'stacking')
y_pos_r = np.arange(len(bar_cat_r))
 
# Create bars
plt.bar(y_pos_r, model_r_squared)
 
# Names on x-axis
plt.xticks(y_pos_r, bar_cat_r)
 
# title and axis names
plt.title('R-Squared')
plt.xlabel('models')
plt.ylabel('values')
    
# Show graphic
plt.show()


# The result showded that in addtion to baseline model linear regression, the rest five models significantly reduced mean squared error, at the same time, R-Sqaured significantly were increased among the rest five models. Among the five models, MLP model showed the lowest mean squared error and highest R-Squared. It is intersted to note that MLP perform the same or even slightly better than the stacking model. 
# 
# Noted that stacking model was built based on the mean value from cross validation and the training set and testing set were resampled, it is unlikely that stacking model did not perform the best because of overfitting issue. Instead, it may be explained by the fact that the model may be suboptimal. It can be inferred that neural network performed the best because random forest regressor and xgboosting regressor did not fit to the optimal parameters, grid search can be implemented to get the optimal value and increase performace in stacking model. 
# 
# Also, noted that random forest regressor also outperformed xgbkosting regressor, it is very likely that xgboosting regressor did not for to the optimal values. It is also possible that neural newtwork may better predict numerical value compared to xgboosting and random forest in this case. For the future work, optimal parameters should be achieved. In addition to MSE and R-Squared, other analysis can be also conducted to accurately define what is the "best model".  
