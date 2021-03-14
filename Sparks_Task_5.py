#!/usr/bin/env python
# coding: utf-8

# **The Sparks Foundation**

# **Data Science and Business Analytics Internship**

# **Task-5:plotting different graph patterns and possible reasons helping Covid-19 spread 
# with basic as well as advanced charts**

# **By-Priyanka Mohanta**

# <img src='COVID-19-visual.jpg'/>

# In[45]:


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[114]:


#load the dataset
data=pd.read_csv(r"C:\Users\kabya\Downloads\owid-covid-data.csv")


# In[115]:


#check the covid 19 dataset
data


# In[116]:


#check the shape of the dataset
data.shape


# In[117]:


data.info()


# In[118]:


#summery statistics for numerical columns
data.describe(include='all')


# In[119]:


#check the first 5 rows
data.head()


# In[120]:


#check the last 5 rows
data.tail()


# In[121]:


#check the null value of this dataset
data.isnull().sum()


# **cleaning the null value**

# 
# As we can see, there are many number of null values in columns-
# icu_patients,icu_patients_per_million,
# hosp_patients,hosp_patients_per_million,weekly_icu_admissions,weekly_icu_admissions_per_million,
# weekly_hosp_admissions,weekly_hosp_admissions_per_million,,new_tests,total_tests                              
# total_tests_per_thousand,new_tests_per_thousand,new_tests_smoothed,new_tests_smoothed_per_thousand,
# positive_rate,tests_per_case,tests_units,total_vaccinations,people_vaccinated,people_fully_vaccinated,
# new_vaccinations,new_vaccinations_smoothed,total_vaccinations_per_hundred,
# people_vaccinated_per_hundred,people_fully_vaccinated_per_hundred      
# new_vaccinations_smoothed_per_million high amount of null values,so how to replace nan value with nalvalue
# so i droped the high amount of null value columns

# In[122]:


data.drop(["icu_patients","icu_patients_per_million","hosp_patients","hosp_patients_per_million","weekly_icu_admissions",
           "weekly_icu_admissions_per_million","weekly_hosp_admissions_per_million",
           "new_tests","total_tests","total_tests_per_thousand","new_tests_per_thousand","new_tests_smoothed",
           "new_tests_smoothed_per_thousand","positive_rate","tests_per_case","tests_units","total_vaccinations",
           "people_vaccinated","people_fully_vaccinated","new_vaccinations","new_vaccinations_smoothed","total_vaccinations_per_hundred",
            "people_vaccinated_per_hundred","people_fully_vaccinated_per_hundred","new_vaccinations_smoothed_per_million"],axis=1,inplace=True)


# In[123]:


#Cheking the object values in this dataset
for i in data.columns:
    if data[i].dtype=="O":
        print(i,":",sum(data[i]=="?"))


# In[57]:


#check the data types
data.dtypes


# In[125]:


sns.heatmap(data.isnull())
plt.show()


# In[126]:


#drop the unrelevent columns
data.drop(['human_development_index','handwashing_facilities','stringency_index',
          'hospital_beds_per_thousand','reproduction_rate','weekly_hosp_admissions'],
          axis=1,inplace=True)


# In[127]:


null_val=data.isnull().sum()/data.shape[0]*100
null_val


# In[136]:


drop_columns=null_val[null_val>30].keys()
drop_columns


# In[138]:


data1=data.drop(columns=drop_columns)


# In[139]:


data1.shape


# In[140]:


data2=data2.dropna()


# In[141]:


data2.shape


# **After cleaning the null value check the dataset**

# In[143]:


plt.figure(figsize=(25,15))
sns.heatmap(data2.isnull())
plt.show()


# **In this above graph,we can see my dataset is fully clened.**

# In[144]:


#again checking the null values
data2.isnull().sum()


# **VISUALIZING**

# In[145]:


#Pearson's Correlation
cor=data2.corr()
plt.figure(figsize=(10,8))
sns.heatmap(cor,annot=True,cmap='coolwarm')
plt.show()


# In[27]:


#check the unique value of the female_smokers column 
data2.female_smokers.unique()


# In[37]:


data['continent'].value_counts()


# In[151]:


#plotting grapg between Continent wise male_smokers 
x = data2.groupby('continent')['male_smokers'].count()
x = x.sort_values(ascending = False)
x.plot(kind = 'barh', figsize = (15,8), color = 'c')
plt.title('Continent Vs Male Smoker', fontsize = 20)
plt.show()


# In[152]:


#plotting grapg between Continent wise female_smokers 
x = data2.groupby('continent')['female_smokers'].count()
x = x.sort_values(ascending = False)
x.plot(kind = 'barh', figsize = (15,8), color = 'b')
plt.title('Continent Vs Female Smoker', fontsize = 20)


# In[99]:


plt.figure(figsize=(15,10))
data2['continent'].value_counts().plot.pie(autopct="%.1f%%")
plt.show()


# In[153]:


#plotting grapg between Continent diabetes
x = data2.groupby('continent')['diabetes_prevalence'].count()
x = x.sort_values(ascending = False)
x.plot(kind = 'barh', figsize = (15,8), color = 'y')
plt.title('Continent wise diabetes_prevalence', fontsize = 20)
plt.show()


# In[43]:


plt.figure(figsize=(15,12))
sns.countplot(x='location',data=data2,palette='rocket_r',order=data2['location'].value_counts().index)
plt.xticks(rotation=90)
plt.show()


# In[74]:


data2['date'].value_counts()


# In[80]:


x = data2.groupby('date')['total_deaths'].sum()
x.plot(xticks = [], marker = '+', figsize = (10,4), color = 'b')
plt.title('total_deaths', fontsize = 20)
plt.show()


# In[82]:


x = data2.groupby('continent')['total_cases', 'new_cases', 'new_deaths'].sum()
x.plot(xticks = [], linestyle = '-', marker = '.', figsize = (10,4),color = ('b','r','g'))
plt.title('Covid 19\n total_cases v/s new_cases v/s new_deaths', fontsize = 20)
plt.show()


# In[81]:


y= data2.groupby('location')['total_cases', 'new_cases', 'date'].sum()
y.plot(xticks = [], linestyle = '-', marker = '.', figsize = (10,4),color = ('b','r','g'))
plt.title('Covid 19\n total_cases v/s new_cases v/s new_deaths', fontsize = 20)
plt.show()


# In[154]:


a= data2.pivot_table('new_cases', columns = 'date', index = 'continent', fill_value = 0, aggfunc = 'sum')
a[a.columns[-1]].sort_values(ascending = False)[:20].plot(kind = 'barh', figsize = (15,5), color = 'm')
plt.title('covid 19 maximum number of cases', fontsize = 20)
plt.show()


# In[92]:


a= data2.pivot_table('new_deaths', columns = 'date', index = 'location', fill_value = 0, aggfunc = 'sum')
a[a.columns[-1]].sort_values(ascending = False)[:20].plot(kind = 'barh', figsize = (15,5), color = 'g')
plt.title('Location Wise with maximum number of new_deaths cases', fontsize = 20)
plt.show()


# **THANK YOU**

# In[ ]:





# In[ ]:





# In[ ]:




