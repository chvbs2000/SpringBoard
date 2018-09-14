
# coding: utf-8

# # Analysis in Medicare Provider Utilization and Payment Data: From the Prospectives of Average Difference between Submitted and Charged Medicare Amount from Physician in California

# With the growth of older population, the need for Medicare is increasing. It is important to
# understand how Medicare works and what factors would contribute to the difference in
# Medicare. Medicare is the federal health insurance program for people who are 65 or older,
# specific younger people with disabilities, and people with End-Stage Renal Disease
# (permanent kidney failure requiring dialysis or a transplantIn this analysis. For Medicare
# beneficiaries, Medicare has different types of plans, beneficiaries would choose the plan
# based on their need. For providers, there are several options provided for charging medicare
# patients. Generally, Medicare has established a standard allowed payment for each
# procedure, if the amount of payment physicians submitted to the Medicare exceed the
# allowed amount, most of the time, physicians/ providers need to cover these extra expense
# (sometimes beneficiaries are required to pay the extra expense if they are out-of-network or if they opt to certain private plan). Therefore, in this analysis, I would like to focus on Average Medicare Difference (AMD), which is the difference that providers, private insurance, or beneficiaries possibly need to afford. AMD is difference between Average Medicare Allowed
# Amount established by Medicare and Average Submitted Charged Amount filed by providers.
# 
# To encourage providers to accept assignment with Medicare for all their patients and become
# participating providers, there are several incentives providing to participating providers. For example, Medicare payment rates for participating providers are approximately 5% or higher
# than the rates paid to non-participating providers. In addition, participating providers may
# receive Medicare’s reimbursement amount directly from Medicare compared to nonparticipating
# providers, who generally bill their Medicare patients according to their charges
# and may not receive payment from Medicare. Also, participating providers will have electronic
# access to Medicare beneficiaries insurance status and make them easy to file claims to get
# beneficiaries coinsurance. Formula for medicare is complicated and may depend on case by case. From the prospective of providers, it requires planning and strategies to choose optimal options to reduce the cost and maximize profit based on their entities, service, and pursuit. 
# 
# I would use statistical method and machine learning to investigate what factors would influence the difference between average Medicare submitted amount and average amount physicians charged in the provider type of Neurosurgery, Cardiac surgery, Vascular surgery, Nurse Anesthetist (CRNA),and Thoracic Surgery, in the state of California.
# 
# 

# ## Data Input
# 
# The data can be seen and downloaded at: https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Medicare-Provider-Charge-Data/Physician-and-Other-Supplier.html

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


# ## Dealing with Outliers

# In[23]:


# outliers in numerical features of interests
boxplot = data_ca_drop_pro.boxplot(column=['Average Medicare Allowed Amount','Average Submitted Charge Amount','Average Medicare Payment Amount','Average Medicare Standardized Amount'], rot=90, fontsize=15)


# In[24]:


# outliers in each procedure
bplot = sns.boxplot(y='Average Medicare Difference', x='Provider Type', 
                 data=data_ca_drop_pro, 
                 width=0.5,
                 palette="colorblind")
bplot.set_xticklabels(bplot.get_xticklabels(),rotation=30)


# We can see that there are a lot outliers in each numerical variables, yet there might be some medical meaning for these outliers, for example, maybe certain types of procedures would be very expensive. Thus we are not removing these outliers at this point. 

# ## Procedures in each Provider Type

# In[25]:


#subset dataframe based on provider type
data_crna = data_ca_drop_pro.loc[data_ca_drop_pro['Provider Type']=='CRNA']
data_vas = data_ca_drop_pro.loc[data_ca_drop_pro['Provider Type']=='Vascular Surgery']
data_cardiac = data_ca_drop_pro.loc[data_ca_drop_pro['Provider Type']=='Cardiac Surgery']
data_thora = data_ca_drop_pro.loc[data_ca_drop_pro['Provider Type']=='Thoracic Surgery']
data_neuro = data_ca_drop_pro.loc[data_ca_drop_pro['Provider Type']=='Neurosurgery']


# ### Top 10 Procedures in CRNA

# In[26]:


data_crna.groupby('HCPCS Description')['Average Medicare Difference'].mean().sort_values(ascending = False).head(10).plot(kind ="barh")
data_crna.groupby('HCPCS Description')['Average Medicare Difference'].mean().sort_values(ascending = False).head(10)


# It seems that anesthesia for procedures on hear and great blood vessel cost has the most differences in average medicare amount, which is around $5000 USD. The procedure includes heart-lung usage, re-operation after original procedures. Further details about the frequency of these re-operating procedures and risk factors that contirbute to the re-operation can be discussed. But these questions will be left opened in the project.

# ### Top 10 Vascular Surgery

# In[27]:


data_vas.groupby('HCPCS Description')['Average Medicare Difference'].mean().sort_values(ascending = False).head(10).plot(kind ="barh")
data_vas.groupby('HCPCS Description')['Average Medicare Difference'].mean().sort_values(ascending = False).head(10)


# It seems that cost in removal of plaque and insection of stents into artery has the most differences in average medicare amount, which is $23700 USD, followed by the procedures involving removal of plaque and insection of stents into arteriers. We can see that removel of plaque and insection of stents account for top 4 average medicare amount difference procedures in Vascular Surgery. 

# ### Top 10 Cardiac Surgery 

# In[28]:


data_cardiac.groupby('HCPCS Description')['Average Medicare Difference'].mean().sort_values(ascending = False).head(10).plot(kind ="barh")
data_cardiac.groupby('HCPCS Description')['Average Medicare Difference'].mean().sort_values(ascending = False).head(10)


# The bar chart shows that insection of vena cava by endovascular approach has the most difference in average medicare amount in cardiac surgery, which is $30700 USD.

# ### Top 10 Thoracic Surgery 

# In[29]:


data_thora.groupby('HCPCS Description')['Average Medicare Difference'].mean().sort_values(ascending = False).head(10).plot(kind ="barh")
data_thora.groupby('HCPCS Description')['Average Medicare Difference'].mean().sort_values(ascending = False).head(10)


# The procedures that involves heart surgery shows the most difference between submitted charged amount and allowed medicare amount, which is around $17200 USD. 

# ### Top 10 Neurosurgery

# In[30]:


data_neuro.groupby('HCPCS Description')['Average Medicare Difference'].mean().sort_values(ascending = False).head(10).plot(kind ="barh")
data_neuro.groupby('HCPCS Description')['Average Medicare Difference'].mean().sort_values(ascending = False).head(10)


# The largest difference in neruosurgery is the procedure of repairing of bulging of blood vessel in brain, which is around 11520 USD.

# ## Feature Exploration

# ### Correlation Matrix

# To investigate what factors would contribute to major impact on the dependent variable - average medicare difference, and understand if independent variables have correlations, correlation matrix is applied. 

# In[31]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(data_ca_drop_pro.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
plt.savefig("corrmatirx.png")


# We can see that Average Medicare Allowed Amount, Average Submitted Charge Amount, Average Medicare Payment Amount,and Average Medicare Standardized Amount have higher correlation with Average Medicare Difference. We will focus on visualizing these features.

# ### Feature Transformation

# #### Average Medicare Difference

# In[32]:


plt.hist(pow(data_ca_drop_pro['Average Medicare Difference'], 1/6))


# #### Average Medicare Allowed Amount

# In[33]:


# log transformation of Average Medicare Allowed Amount
plt.hist(np.log(data_ca_drop_pro['Average Medicare Allowed Amount']))


# In[34]:


# copy df
data_pro_copy = data_ca_drop_pro.copy()
# log transformation of 
data_pro_copy['Log Average Medicare Allowed Amount'] = np.log(data_pro_copy['Average Medicare Allowed Amount'])
sns.lmplot(x = 'Log Average Medicare Allowed Amount', y='Average Medicare Difference', data = data_pro_copy, fit_reg = False, hue = 'Provider Type', legend = False)
plt.legend(loc='upper left')


# We could see there's more variations in vascular surgery. With more allowed average medicare amoumt, there's a increasing trend in variations in average medicare difference and the amount of average medicare difference.  

# #### Average Medicare Payment Amount

# In[35]:


plt.hist(np.log(data_ca_drop_pro['Average Medicare Payment Amount']))


# In[36]:


# scatter plot 
sns.lmplot(x = 'Average Medicare Payment Amount', y='Average Medicare Difference', data = data_ca_drop_pro, fit_reg = False, hue = 'Provider Type', legend = False)
plt.legend(loc='upper left')


# The scatter plot showed that vascular surgery varied the most in average medicare payment amount that medicare covered after coinsurance amount deducted compared to other surgery, and it can be visualized that there's slightly positive correlation between average medicare payment amount and average medicare difference.

# ### Creating Dummy Variables, Normalization, and Data Merge

# Now, I want to know that what factors would affect the average submitted charge amount. To do so, I need to get dummies for each categorical value to save space and ease computatinal complexity. 

# In[37]:


# convert categorical variable to dummy variable
data_ca_drop_dummy = pd.get_dummies(data_ca_drop_pro[['Zip Code of the Provider','Entity Type of the Provider','Provider Type','Medicare Participation Indicator','HCPCS Description','HCPCS Drug Indicator','City of the Provider']])


# In[38]:


# build continuous variable dataframe
data_ca_drop_continue = data_ca_drop_pro[['Number of Services','Number of Medicare Beneficiaries','Number of Distinct Medicare Beneficiary/Per Day Services','Average Medicare Payment Amount','Average Medicare Standardized Amount','Average Medicare Difference']]


# In[39]:


# normalize continuous dataframe
data_ca_drop_continue = (data_ca_drop_continue-data_ca_drop_continue.min())/(data_ca_drop_continue.max()-data_ca_drop_continue.min())


# In[40]:


# concat binary dataframe and continuous dataframe
data_ca_drop_dummy = pd.concat([data_ca_drop_continue,data_ca_drop_dummy], axis = 1)


# After finalizing our dataframe of interest, to train, optimize, and validate the model with unseen data. I am going to divide the dataset into training dataset, validation dataset, and testing dataset, with 60%, 20%, 20%, respectively. Training dataset is used for training the data. While validation dataset is used for providing an unbiased evaluation of how training dataset fit to the data while tuning model parameters. Whereas testing dataset is used as an unseen data to evaluate the generalization of the model. 

# In[41]:


from sklearn.preprocessing import StandardScaler
# run lineaqr regression model
data_ca_drop_dummy.reset_index(drop = True)
Y = data_ca_drop_dummy['Average Medicare Difference']
X = data_ca_drop_dummy.drop(['Average Medicare Difference'], axis = 1)


# In[42]:


from sklearn.model_selection import train_test_split


# In[46]:


x_train, x_test, y_train, y_test    = train_test_split(X, Y, test_size=0.2, random_state=1)

x_train, x_val, y_train, y_val    = train_test_split(x_train, y_train, test_size=0.25, random_state=1)


# In[132]:


# print ratios of training, validation, and testingset
print('Ratio for each dataset: train: {}% | validation: {}% | test {}%'.format(round(len(y_train)/len(Y),2)*100,
                                                       round(len(y_val)/len(Y),2)*100,
                                                       round(len(y_test)/len(Y),2)*100))


# ## Linear Regression Model as a Baseline Model

# In[128]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


# In[71]:


# train baseline model: linear regression
model = linear_model.LinearRegression()
model.fit(x_train, y_train)

# train error 
print ("Linear Regression train error (RMSE): %.4f"%sqrt(mean_squared_error(y_train, model.predict(x_train))))

# validation error 
print ("Linear Regression validation error (RMSE): %.4f"%sqrt(mean_squared_error(y_val, model.predict(x_val))))

# test error 
print ("Linear Regression test error (RMSE): %.4f"%sqrt(mean_squared_error(y_test, model.predict(x_test))))      


# The validation error and test error is far more than train error, indicating that the model is horribly overfitting. Other methods will be used to improve the performance.

# ## Elastic Regression Model

# Due to the poor performace of linear regression, we used regularization to see if we can improve the result. Generally, Lasso (L1) or Ridge (L2) regression would be applied. Lassopenalizes the model by shrinking coefficients of irrelevant vairables to 0, while Ridge does not enforce coefficient of irrelevant variables to 0, instead, it minimize their impact on the model. Lasso provides sparsity, yet it might lose some relevant independent variables along the way. Ridge is often used when the independent variables are collinear by introducing bias to reduce the variance of parameter estimates, yet it does not reduce complexity. Here, I choose to use elastic regression because it solves limitations of both regularization methods, but also keep their special properties. 

# In[57]:


cv_enet = ElasticNetCV(l1_ratio = np.linspace(0.1,1,40.), cv = 10, eps = 0.001, n_alphas = 100, fit_intercept = True, normalize = True, max_iter = 2000)


# In[58]:


cv_enet.fit(x_train, y_train)


# In[60]:


print ("optimal l1_ratio: %.3f"%cv_enet.l1_ratio_)
print ("optimal alpha: %.6f"%cv_enet.alpha_)
print ("number of iterations : %d"%cv_enet.n_iter_)


# The l1 ratio is 0.1, which means ridge regression accounts majority part in the elastic net. This is reasonable because we may have collinearity and ridge solve collinearity issue better than lasso regressor does. 

# In[129]:


# train optinal elastic net with 10-fold cross validation
kf = KFold(n_splits = 10, shuffle=True) # 10 folds 
enet_train_error = []
enet_validate_error = []
enet_test_error = []
enet_score = []
for train_idx, test_idx in kf.split(X):
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]
    X_train, X_val, y_train, y_val    = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    
    net_model = ElasticNet(l1_ratio = cv_enet.l1_ratio_, alpha = cv_enet.alpha_, max_iter = cv_enet.n_iter_, fit_intercept = True, normalize = True)
    net_model.fit(X_train, y_train)
    
    enet_train_error.append(sqrt(mean_squared_error(y_train, net_model.predict(X_train))))
    enet_validate_error.append(sqrt(mean_squared_error(y_val, net_model.predict(X_val))))
    enet_test_error.append(sqrt(mean_squared_error(y_test, net_model.predict(X_test))))
    enet_score.append(net_model.score(X_test, y_test))


# In[130]:


# elastic net training error 
print ("Elastic Net average train error (RMSE): %.4f"%np.mean(enet_train_error))

# elastic net validation error 
print ("Elastic Net avergae validation error (RMSE): %.4f"%np.mean(enet_validate_error))

# elastic net testing error 
print ("Elastic Net avergae test error (RMSE): %.4f"%np.mean(enet_test_error))  


# In[131]:


# Accuracy for optinmal Elastic Net model
print("Accuracy for Random Forest after CV: ",round(np.mean(enet_score),4))


# In[75]:


plt.scatter(np.sqrt(y_test), np.sqrt(net_model.predict(x_test)))
plt.xlabel("True Values")
plt.ylabel("Predictions")


# With implementing Elastic Net Regularization, training error is slightly increased, from 0.026 to 0.027. While both validation error and testing error are drastically decreased, to 0.029 and 0.03, respectively. It has successfully overcome overfitting issue in the baseline linear regression model. And the accuracy of the model imporved to 0.578. Later on, I would like to see how it works by implementing other machine learning models. 

# ### Feature Importance from Elastic Net

# In[76]:


def get_feature_importance(x_train, model):
    
    feature_importance = pd.Series(index = x_train.columns, data = np.abs(model.coef_))
    selected_features = (feature_importance>0).sum()
    print('{0:d} features, reduction of {1:2.2f}%'.format(
        selected_features,(1-selected_features/len(feature_importance))*100))
    feature_importance.sort_values().tail(10).plot(kind = 'barh', figsize = (20,8))


# In[78]:


get_feature_importance(X, net_model)


# Based on the graph above, Elastic Net model indicated that Average Medicare Payment Amount, procedure related to using an endoscope to remove forign body from chest cavity have larger impact on Average Medicare Difference. 

# # Random Forest Training and Prediction

# Among many training algorithms, decision tree is regarded as one of the most robust algorithms to train data and is widely used in different applications. Yet a decision tree learner may easily underfitting or overfitting, and sometimes it is expensive for a decision tree to determine the best feature to split on. Therefore, random forest has been applied to
# improve a decision tree learner’s robustness and efficiency. Here, I will use grid search with 10-fold cross validation to optimize forst parameters. Average training error, average validation error, and average testing error are subsequently achieved. 

# In[83]:


from sklearn.ensemble import RandomForestRegressor

# random forest model
rf = RandomForestRegressor(random_state=4)

rf_param_grid = { 
    'n_estimators': np.arange(200,1200,200),
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : np.arange(4,10,2),
    'min_samples_leaf': np.arange(2,10,2),z
    'min_samples_split' : np.arange(6,15,3) 
}

cv_rf = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv= 10)
cv_rf.fit(x_train, y_train)


# In[84]:


cv_rf.best_params_


# In[120]:


# train optinal forest with 10-fold cross validation
kf = KFold(n_splits = 10, shuffle=True) # 10 folds 
train_error = []
validate_error = []
test_error = []
rf_score = []
for train_idx, test_idx in kf.split(X):
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]
    X_train, X_val, y_train, y_val    = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    
    opt_rf=RandomForestRegressor(random_state=4, max_features='auto', n_estimators= 200, max_depth=8, min_samples_split= 6, min_samples_leaf= 2)
    opt_rf.fit(X_train, y_train)
    
    train_error.append(sqrt(mean_squared_error(y_train, opt_rf.predict(X_train))))
    validate_error.append(sqrt(mean_squared_error(y_val, opt_rf.predict(X_val))))
    test_error.append(sqrt(mean_squared_error(y_test, opt_rf.predict(X_test))))
    rf_score.append(opt_rf.score(X_test, y_test))


# In[121]:


# random forest train error 
print ("RF average train error (RMSE): %.4f"%np.mean(train_error))

# random forest validation error 
print ("RF avergae validation error (RMSE): %.4f"%np.mean(validate_error))

# random forest test error 
print ("RF avergae test error (RMSE): %.4f"%np.mean(test_error))      


# In[127]:


# Accuracy for optinmal Random Forest model
print("Accuracy for Random Forest after CV: ",round(np.mean(rf_score),4))


# The result from the random forest model showed that it is indeed improved the model a little bit comparing to Elastic Net model based on the accuracy. the accuracy has been improved by 3%. Training error, validation error, and testing error are all reduced compared to the previous two models. 

# ### Feature Importance from Random Forest

# In[97]:


# feature importance
feature_importance = pd.Series(index = x_train.columns, data = np.abs(opt_rf.feature_importances_))
selected_features = (feature_importance>0).sum()
print('{0:d} features, reduction of {1:2.2f}%'.format(
    selected_features,(1-selected_features/len(feature_importance))*100))
feature_importance.sort_values().tail(10).plot(kind = 'barh', figsize = (20,8))


# It is intersting to see that the random forest model indicated the different important features comparing to Elastic Net. It showed that Average Medicare Payment Amount and Average Medicare Standardized Amount are two important features that impact Average Medicare Difference. Yet both model indicate that Average Medicare Payment Amount is the most important feature that influence Average Medicare Difference. 

# In[137]:


print (y_test.shape)
print (X_test.shape)


# In[149]:


# scatter plot on predictions
plt.scatter(y_test**(1/3), np.log(opt_rf.predict(X_test)))
plt.xlabel("True Values")
plt.ylabel("Predictions")

"""

# # eXtreme Gradient Boosting

# In[99]:


#!pip install xgboost
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[100]:


# make sure x_train and y_train have the same length
print (x_train.shape)
print(y_train.shape)


# In[ ]:


# train xgboosting regressor 
params = {'min_child_weight':[1, 3, 5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],
'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [4, 6, 8], 'learning_rate':[0.005, 0.01, 0.05, 0.1], 'n_estimators': [200, 500]}
xgb = XGBRegressor(nthread=-1)
xgb_grid = GridSearchCV(xgb, params, cv = 10)
xgb_grid.fit(x_train,y_train)
#y_pred_xgb = xgb.predict(x_test)


# In[ ]:


# train optinal xgboostingwith 10-fold cross validation
kf = KFold(n_splits = 10, shuffle=True) # 10 folds 
xgb_train_error = []
xgb_validate_error = []
xgb_test_error = []
xgb_score = []
for train_idx, test_idx in kf.split(X):
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]
    X_train, X_val, y_train, y_val    = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    
    xgb_opt=XGBRegressor(nthread=-1)
    xgb_opt.fit(X_train, y_train)
    
    xgb_train_error.append(sqrt(mean_squared_error(y_train, xgb_opt.predict(X_train))))
    xgb_validate_error.append(sqrt(mean_squared_error(y_val, xgb_opt.predict(X_val))))
    xgb_test_error.append(sqrt(mean_squared_error(y_test, xgb_opt.predict(X_test))))
    xgb_score.append(xgb_opt.score(X_test, y_test))


# In[ ]:


# xgboosting train error 
print ("XGB average train error (RMSE): %.4f"%np.mean(xgb_train_error))

# xgboosting validation error 
print ("XGB avergae validation error (RMSE): %.4f"%np.mean(xgb_validate_error))

# xgboosting test error 
print ("XGB avergae test error (RMSE): %.4f"%np.mean(xgb_test_error))  


# In[ ]:


# Accuracy for optinmal Random Forest model
print("Accuracy for XGB after CV: ",round(np.mean(xgb_score),4))


# In[ ]:


# feature importance 
feature_importance = pd.Series(index = x_train.columns, data = np.abs(xgb_opt.feature_importances_))
selected_features = (feature_importance>0).sum()
print('{0:d} features, reduction of {1:2.2f}%'.format(
    selected_features,(1-selected_features/len(feature_importance))*100))
feature_importance.sort_values().tail(10).plot(kind = 'barh', figsize = (20,8))


# In[ ]:


# plot prediciton values vs true values 
plt.scatter(np.sqrt(y_test), np.sqrt(xgb_grid.predict(x_test)))
plt.xlabel("True Values")
plt.ylabel("Predictions")


# # Stacking 

# In[ ]:


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


# In[ ]:


# x's dimension and y's dimension 
print (x_train.shape)
print(y_train.shape)


# In[ ]:


ntrain = x_train.shape[0]
ntest = x_test.shape[0]

SEED = 43 
kf = KFold(ntrain, n_folds= 10, random_state=SEED)


# In[ ]:


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


# In[ ]:


# train all models
rf_train_stack, rf_test_stack = get_train_test_per_model(rf, x_train, y_train, x_test) # Random Forest
xgb_train_stack, xgb_test_stack = get_train_test_per_model(xgb,x_train, y_train, x_test) # Xgboost
mlp_train_stack, mlp_test_stack = get_train_test_per_model(mlp, x_train, y_train, x_test) # Neural Netork 


# In[ ]:


predictions_train_stack = pd.DataFrame( {'RandomForest': rf_train_stack.ravel(),
      'GradientBoost': xgb_train_stack.ravel(),
      'NeuralNetowrk': mlp_train_stack.ravel()
    })


# In[ ]:


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


# In[ ]:


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
    
    
    


# In[ ]:


#average result from cross validation by row
xgb_result = pred_xgb.mean(axis = 1)
mlp_result = pred_mlp.mean(axis = 1)
rf_result = pred_rf.mean(axis = 1)

#combine three model
pred_combine = pd.concat([xgb_result, mlp_result, rf_result], axis = 1)
pred_combine.columns = ['xgb','mlp','rf']



# In[ ]:


def get_optimal_MSE(allocs,pred_df, eps=1e-15):
    pred_df = np.clip(pred_df, eps, 1 - eps)
    pred_df_w = np.sum(pred_df*allocs,axis = 1)
    return mean_squared_error(y_test1, pred_df_w)


# In[ ]:


model_list = ['XGBoost','MLP','RandomForest']
model_len = len(model_list)
init_vals= [1.0 / model_len] * model_len
cons = ({'type': 'ineq', 'fun': lambda x: 1.0-np.sum(x)})
bnds = [(0.0, 1.0)] * model_len

#optimized allocations
opts = spo.minimize(get_optimal_MSE, init_vals, args = (pred_combine,),method='SLSQP', bounds=bnds, constraints=cons, options = {'disp':True})
opt_allocs = opts.x


# In[ ]:


# optimal wieghts for each learner 
display(opt_allocs)

# optimal result 
display(opts)


# In[ ]:


y_stack_pred = 0.33333333*mlp.predict(x_test) + 0.33333333*xgb.predict(x_test) + 0.33333333*rf.predict(x_test)


# In[ ]:


# MSE
print("Stacking MSE: {}".format(mean_squared_error(y_test, y_stack_pred)))

# R squared
print ("Stacking R-Squared: {}".format(r2_score(y_test, y_stack_pred)))


# In[ ]:


# plot prediciton values vs true values 
plt.scatter(np.log(y_test), np.log(y_stack_pred))
plt.xlabel("True Values")
plt.ylabel("Predictions")

"""

# ## ## Linear Regression based on Most Important Features
# 

# Now we know whcih feaures are important for better predicting AMD, lets construct a linear regression model based on these features again.

# In[103]:


import statsmodels.api as sm

model_lr_aah = sm.OLS(data_ca_drop_dummy["Average Medicare Difference"], data_ca_drop_dummy[['Average Medicare Payment Amount', 'Average Medicare Standardized Amount','HCPCS Description_Heart surgery procedure', 'HCPCS Description_Reshaping of skull bone defect']])
result_lr = model_lr_aah.fit()
display(result_lr.params)
display(result_lr.summary())


# The result from linear regression showed that R-Squared was 0.57, which has been significantly improved compared to baseline linear regression model in prediction. It showed that with 0.42 unit decrease in Average Medicare Payment Amount, 0.97 unit increase in Average Medicare Standardized Amount, 0.27 increase in HCPCS Description_Heart surgery procedure, and 0.11 increase in HCPCS Description_Reshaping of skull bone defect would result in 1 unit increase in AMD. Noted that reamining other factors the same, one unit increase in procedure of heart surgery procedure would result in more than three unit increase in AMD. Similarily, one unit increase in procedure of Reshaping of skull bone defect would result in approximately 10 unit increase in AMD. 
