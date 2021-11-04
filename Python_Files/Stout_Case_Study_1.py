#!/usr/bin/env python
# coding: utf-8

# # Case Study 1

# # Understanding the Loan Dataset:

# In[116]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
import warnings

warnings.filterwarnings('ignore')


# In[117]:


loan_data = pd.read_csv(r"C:\Users\User\Downloads\loans_full_schema.csv")


# In[118]:


loan_data.head(5)


# # Size of the dataset

# In[119]:


loan_data.shape #Data Consists of 10k rows with 55 variables


# In[120]:


l_df = loan_data[loan_data.duplicated()]
print("Number of duplicate rows:" , l_df.shape)


# Data Description : The dataset consists of 10k rows with 55 features describing different features that are involved in availing a loan. It consists of object, float and integer datatypes. The dataset does not contain any dupicate value though it consists missing and NAN values. These values need to be handled for this data to be used to develop a prediction model. 

# In[121]:


loan_data.info()


# The above output indicates that there are about 5 columns that have more than 60% of missing data values 

# # Data Cleaning

# Data Cleaning :
# The following steps were taken to clean the dataset:
# 1)All those columns that had more than 60% of the data missing were removed.
# 2)All those columns that were of type object were encoded into numeric values and then converted to category type. 

# # Handling missing values

# In[122]:


perc = 60.0
min_count =  int(((100-perc)/100)*loan_data.shape[0] + 1)
loan_df = loan_data.dropna( axis=1,thresh=min_count)


# In[123]:


loan_df['emp_title'].fillna(loan_df['emp_title'].mode()[0], inplace = True)
#loan_df.apply(lambda x: x.fillna(x.mean())) 
loan_df = loan_df[loan_df['emp_length'].notna()]
loan_df = loan_df[loan_df['num_accounts_120d_past_due'].notna()]
loan_df = loan_df[loan_df['months_since_last_credit_inquiry'].notna()]
loan_df = loan_df[loan_df['debt_to_income'].notna()]
loan_df["months_since_last_delinq"].fillna(loan_df["months_since_last_delinq"].mean(), inplace=True)

print(loan_df.isna().sum())


# In[124]:


#This indicates no duplicate values are present
loan_rows_df = loan_df[loan_df.duplicated()]
loan_rows_df.shape


# # Encoding of categorical variable

# In[125]:


obj_df = loan_df.select_dtypes(include=['object']).copy()
obj_df.head()


# In[126]:


from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import LabelEncoder

cat_list=["emp_title","state","homeownership","verified_income","loan_purpose","application_type","grade","sub_grade","issue_month","loan_status","initial_listing_status","disbursement_method"]

for i in cat_list:
    le = LabelEncoder()
    encoded = le.fit_transform(loan_df[i])
    loan_df[i] = encoded


# In[127]:


loan_df.info()


# In[175]:


sn.heatmap(loan_df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Heatmap for Missing Data values', fontsize=20)
plt.figure(figsize=(12,10))


# In[128]:


loan_df.describe()


# # Outlier Detection

# In[140]:


import seaborn as sns
sns.boxplot(loan_df['loan_amount'])
plt.title("Visualizing Outlier for loan amount feature")


# In[136]:


sns.boxplot(loan_df['debt_to_income'])
plt.title("Visualizing Outlier for debt_to_income feature")


# Outliers are values that are varies significantly from the other set of values in the given dataset. Handling these values 
# improves in the overall accuracy and efficiency of the final model. One way to visualize outliers efficiently is through a 
# box plot. The above two visualizations are a sample of a box plot for two different columns in a dataset to determine outliers.
# The first visualization shows that there is not many outliers in the annual_income column where as all the points above 50, 
# which is third Quartile range are considered outliers.
# 

# In[131]:


Q1 = loan_df.quantile(0.25)
Q3 = loan_df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[132]:


print(loan_df < (Q1 - 1.5 * IQR)) |(loan_df > (Q3 + 1.5 * IQR))


# The below data shows the amount of skewness in each columns of the dataset. The ideal range of skewness is between 0 and 1. 
# The output for some of the column shows that either they are skewed severely to the right having very high values (eg:tax_liens) or skewed to the left extreme indicating very low values compared to the rest of the data. There are some columns that are left skewed but not to any extreme. This shows that the outlier presence in our dataset is right skewed overall.

# In[133]:


for i in loan_df.columns:
    print(i, ":", loan_df[i].skew())


# # Removing Outliers from dataset

# In[142]:


for i in loan_df.columns:
    a=loan_df[i].quantile(0.10)
    b=loan_df[i].quantile(0.90)
    loan_df[i] = np.where(loan_df[i] <a, a,loan_df[i])
    loan_df[i] = np.where(loan_df[i] >b, b,loan_df[i])
    
for i in loan_df.columns:
    print(i, ":", loan_df[i].skew())


# The above technique is called Quantile based flooring and capping. We calculate the 10th and 90th percentile of each column 
# and those vales will be used for this technique. The skewness report after doing this shows that data is less skewed 
# compared to the previous skewness report provided above.

# # Data Visualizations

# Five visualizations were made to determine the interest rate distribution, loan amount availed for different purposes,
# how the interest rate varies with the loan grade, debt to income ratio of different states and the relationship between the 
# loan amount received and the annual income of applicants.

# In[146]:


plt.figure(figsize=(14,8))
from jupyterthemes import jtplot
jtplot.style(theme='onedork')

sn.set_palette("husl")
plt.figure(figsize=(18,10))
sn.distplot(loan_df['interest_rate'], hist='density')
plt.axvline(x=loan_df.interest_rate.mean(), color='red', linestyle='-', lw=4, label='Mean of Interest Rate')
plt.title('Interest Rate Distribution', fontsize=20)
plt.xlabel('Interest Rate', fontsize=18)
plt.legend(fontsize=20)
plt.show()


# This graph shows how the density of interest rate distribution with an average of 12.5% 

# In[148]:


plt.figure(figsize=(14,8))
jtplot.style(theme='onedork')
ax=sn.barplot(obj_df.loan_purpose, loan_df.loan_amount)

plt.title('Loan Amount Availed for Different Purposes', fontsize=20)
plt.xlabel('Loan Purpose', fontsize=18)
plt.ylabel('Loan Amount', fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


# This helps us understand the business domains that mostly require loans. Small business have the highest requirement which 
# could be attributed to their scale and lack of background support 

# In[251]:


plt.figure(figsize=(30,12))
sn.boxplot(x=loan_df.interest_rate, y=obj_df.grade, palette="vlag")
plt.title('Box Plot for Interest Rate vs. Loan Grade', fontsize=35)
plt.xlabel('Rate of Interest', fontsize=30)
plt.ylabel('Loan Grade', fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()


# This graph shows how rate of interest varies with respect to the loan grade represented in the form of a boxplot

# In[174]:


plt.figure(figsize=(14,8))

ax=sn.boxplot(x=obj_df.state, y=loan_df.debt_to_income)
plt.title("Debt-to-income Ratio for Every State")
plt.tight_layout()


# This graph explains the debt to income ratio of different states. Idaho seems to be the state with the highest ratio

# In[251]:


plt.figure(figsize=(14,8))
jtplot.style(theme='onedork')
ax=sn.barplot(obj_df.loan_purpose, loan_df.interest_rate)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.title("Loan Purpose vs Interest Rate")
plt.tight_layout()
plt.show()


# This visulaization is to understand the relationship between purpose for which the loan was received vs the interest rate. This shows that renewable_energy has the highest rate of interest

# # Feature Selection & Machine Learning Model

# In[219]:


#Identifying the target and feature variables
target=loan_df["interest_rate"]
loan_features=loan_df.drop(columns=["interest_rate"])
loan_features.info()


# In[248]:


from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

numerical_features=["annual_income","debt_to_income","loan_amount","loan_purpose","term","grade","sub_grade"]

features_copy=loan_features
features_copy[numerical_features].describe().round(2)

df_scaled = scale(features_copy[numerical_features])
df2 = pd.DataFrame(df_scaled, columns=numerical_features)



sns.set(style="whitegrid")
sns.pairplot(df2[numerical_features], kind='reg', diag_kind='kde')
plt.title("Scatter Plot shwing correlation between some of the numerical columns")
plt.show()


# The above scatter plot show that for many mariable they is no corelation and they are randomly distributed.

# In[177]:


loan_features=loan_features.drop(columns=["emp_title","emp_length","inquiries_last_12m","issue_month","loan_status"])


# In[178]:


loan_features=loan_features.drop(columns=["disbursement_method","verified_income","num_accounts_30d_past_due"])


# In[179]:


from sklearn.model_selection import KFold, cross_val_score, train_test_split
X_train, X_test, y_train, y_test = train_test_split(loan_features, target, test_size=0.3, random_state=40) 


# In[180]:


from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train,y_train)


# In[181]:


print(lm.intercept_)


# In[205]:


predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.title("Scatter Plot to visualize the output")


# In[183]:


sn.distplot((y_test-predictions),bins=50);


# In[184]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

lin_mae=metrics.mean_absolute_error(y_test, predictions)
lin_mse=metrics.mean_squared_error(y_test, predictions)
lin_rmse=np.sqrt(metrics.mean_squared_error(y_test, predictions))


# Predicted Output vs Orginal Values

# In[204]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(df.head(10))


# In[210]:


print('Training score: {}'.format(lm.score(X_train, y_train)))
print('Test score: {}'.format(lm.score(X_test, y_test)))


# In[185]:


from sklearn.ensemble import RandomForestRegressor

rfg = RandomForestRegressor(n_estimators=45, max_depth=10, random_state=0)
rfg.fit(X_train, y_train)

y_pred=rfg.predict(X_test)


# In[206]:


plt.scatter(y_test,y_pred)
plt.title("Scatter Plot to visualize the output for Random Forest Regressor")


# In[186]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

reg_mae=metrics.mean_absolute_error(y_test, y_pred)
reg_mse=metrics.mean_squared_error(y_test, y_pred)
reg_rmse=np.sqrt(metrics.mean_squared_error(y_test, y_pred))


# In[208]:


df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df1.head(10))


# In[218]:


print('Training score: {}'.format(rfg.score(X_train, y_train)))
print('Test score: {}'.format(rfg.score(X_test, y_test)))


# In[217]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
metrics = ['lin_mse', 'lin_mae', 'reg_mse', 'reg_mae']
scores = [lin_mae, lin_mse, reg_mae, reg_mse]
width=0.40
ax.bar(metrics,scores)
plt.title("Accuracy metrics for Linear and Random Forest Regression")
plt.show()


# The above two graphs show that the ditribution is linear and normally distributed indicating that the Linear Regression model has performed well. The error values observed in both the models is very less indicating good performance. 

# Future Work : I would want to use more sophisticated techniques for feature selection and combine more models to improve performance.
