#!/usr/bin/env python
# coding: utf-8

# **Project description**
# Project is to prepare a report for a bank’s loan division. You’ll need to find out if a customer’s marital status and number of children have an impact on whether they will default on a loan. The bank already has some data on customers’ credit worthiness.
# Your report will be considered when building a credit score for a potential customer. A credit score is used to evaluate the ability of a potential borrower to repay their loan.

# ## Analyzing borrowers’ risk of defaulting
# 
#  to prepare a report for a bank’s loan division. We need to find out if a customer’s marital status and number of children has an impact on whether they will default on a loan. The bank already has some data on customers’ credit worthiness.
# 
# Our report will be considered when building a **credit scoring** of a potential customer. A ** credit scoring ** is used to evaluate the ability of a potential borrower to repay their loan.

# ### Step 1. Open the data file and have a look at the general information. 

# In[1]:


import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer 
english_stemmer = SnowballStemmer('english')
df=pd.read_csv('/datasets/credit_scoring_eng.csv')
df.info()


# <div class="alert alert-block alert-success">
# <b>Success:</b> Thank you for collecting all imports in the first cell!
# </div>

# In[2]:


df.head(13)


# ### Conclusion

# The data has 12 columns and 21525. In data two columns ['days_employed','total_income'] have missing values. This can be known by seeing the number of non null entries in these column. Also by seeing, head and tails it can be further ensured. Likewise, there are two columns having float64 datatype, five columns with int64, and five columns with 'object' datatype.

# <div class="alert alert-block alert-success">
# <b>Success:</b> Data loading and initial analysis are well done.
# </div>

# ### Step 2. Data preprocessing

# ### Processing missing values

# In[3]:


df1=df[df['days_employed'].isnull()]
print(df['education'].value_counts())
print('\n')
print(df1['education'].value_counts())


# In[4]:


print(df[df['days_employed'].isnull()].head(15))


# In[5]:


print(df['income_type'].value_counts())
print('\n')
print(df1['income_type'].value_counts())


# In[6]:


df['total_income'].mean()


# In[7]:


df.groupby('income_type')['total_income'].mean()


# In[8]:


df.groupby('family_status')['total_income'].mean()


# In[9]:


df.groupby('education')['total_income'].mean()


# In[10]:


df=df.dropna()


# ### Conclusion

# on analysisng data, missing values appeared  in 'days_employed' and'total_income' columns. On analysing missing values, it can be seen that they are random. The propertion of missing values ( days_employed and total_income) belonging to all subcategories of each column and total number of costomer belonging to that subcategory are roughly about 10 percents. Looking at overall mean of total_income and average total_income on different subcategories are different. In this condition, rows containing missing values can be deleted as the overall characteristic can expected to remain similar.

# <div class="alert alert-block alert-success">
# <b>Success: </b>Step was done not bad.. But in my opinion it would be better to fill missing values. But your decision is also okay. There is no single solution, this is partly creative work .</div>

# ### Data type replacement

# In[11]:


df['total_income']=df['total_income'].astype(int)
df['days_employed']=df['days_employed'].astype(int)
df['days_employed'] = df['days_employed'].abs()
df['children'] = df['children'].abs()
df.info()


# ### Conclusion

# 'total_income' and 'days_employed' are converted to integer type from floating type as it reduces data size and ease for eye. Also, number of children and days employed are converted to absolute value as there were some negetive values.

# ### Processing duplicates

# In[12]:


duplicateDFRow = df[df.duplicated()]
print(duplicateDFRow)


# In[13]:


df['education']= df['education'].str.lower()
df.head(10)


# In[14]:


print(df['education'].value_counts())


# ### Conclusion

# There was not any duplicated row but the string on 'education' column was written in different cases making different category for same level of study.
# This problem is solved by converting all string to lowercase in that column.

# ### Categorizing Data

# In[15]:


print(df.groupby('children')['debt'].sum().sort_values())


# In[16]:


matrix=pd.pivot_table(df , index=['children'], values=['dob_years', 'debt','education_id', 'family_status_id','days_employed','total_income'],
                    aggfunc={'dob_years': np.median,'debt':[np.median,np.mean],'education_id': [np.median,np.mean],'family_status_id': [np.median, np.mean], 'days_employed': np.mean, 'total_income':np.mean})
print(matrix)


# In[17]:


df['children']=df["children"].replace(20 , 2) 
#""" '20' could be resulted from typos, so it should be either '2' or '0'"""
## for 20, more of parameters means  are close to '2' compared to '0'.


# In[18]:


debt_children=df.groupby('children')['debt'].sum().sort_values()
print(debt_children)


# In[19]:


total_debt_children=df['children'].value_counts().sort_values()
print(total_debt_children)


# In[20]:


repayment_kids= (total_debt_children-debt_children)/total_debt_children*100
print(repayment_kids)


# ### Conclusion

# 

# ### Step 3. Answer these questions

# - Is there a relation between having kids and repaying a loan on time?

# From the above result, it can be clearly seen that having more kids increase the probability of paying loan on time.

# <div class="alert alert-block alert-success">
# <b>Success: </b> Correct. </div>

# In[21]:


pivot_family_status=pd.pivot_table(df , index=['family_status'], values=['debt'],
                    aggfunc={'debt':np.mean})
print(pivot_family_status.sort_values('debt'))


# ### Conclusion

# 

# - Is there a relation between marital status and repaying a loan on time?

# From the above result, the co-relation between 'family status' and the average defaulted debt on that category. So, it can be cocluded that the widow/widower have higher chance of repaying loan on time and unmarried have lower repaying rate.

# In[22]:


print(df['total_income'].min())
print(df['total_income'].max())
print(df['total_income'].mean())


# In[23]:


low_income=df[df['total_income']< 20000]
low_income.shape


# In[24]:


def income_group(row):
    
    total_income = row['total_income']
        
    if total_income <= 40000:
        return 'low'   
        
        
    if (total_income >= 40000 and total_income <= 60000):
        return 'medium'    
    if (total_income >= 60000):
        return 'high'

df['income_group'] = df.apply(income_group, axis=1)
df.head()


# In[25]:


pivot_income_group=pd.pivot_table(df , index=['income_group'], values=['debt'],
                    aggfunc={'debt':np.mean})
print(pivot_income_group.sort_values('debt'))


# ### Conclusion

# 

# - Is there a relation between income level and repaying a loan on time?

# From the result, it can be seen that lower income group has higher default debt. So, there is co-relation between income and repaying loan on time.

# In[29]:


#Split the sentences to lists of words.
df['category'] = df['purpose'].str.split()

# Make sure we see the full column.
pd.set_option('display.max_colwidth', -1)
df['stemmed']=df['category'].apply(lambda x: [english_stemmer.stem(y) for y in x])
   
# Stem every word.
df = df.drop(columns=['category']) # Get rid of the unstemmed column.
def debt_purpose(row):
    purpose = row['stemmed']
    
    
    for query in purpose:
        for word in query.split(" "):
            stemmed_word = english_stemmer.stem(word)
            if 'real' in stemmed_word:
                return 'Housing'           
    
            if 'hous' in stemmed_word:
                return  'Housing'            

            if 'properti' in stemmed_word:
                return  'Housing'
           
            if 'car' in stemmed_word:
                return  'car'
       
            if'wed' in stemmed_word:
                return  'Wedding'
        
            if 'educ' in stemmed_word:
                return  'Education'
            

df['debt_purpose'] = df.apply(debt_purpose, axis=1)               
df.tail(15)


# In[30]:


pivot_purpose=pd.pivot_table(df , index=['debt_purpose'], values=['debt'],
                    aggfunc={'debt':np.mean})
print(pivot_purpose.sort_values('debt'))


# <h1> Conclusion </h1>

# 

# <h2> How do different loan purposes affect on-time repayment of the loan? </h2>

# From this result more chances of repayment is on housing, property and real state related loan. Then, the repayment probability on time is on 'wedding' related loans followed by 'education'and 'car' related debt respectively.

# <h1> Conclusion </h2>

# overall, the repayment probability on time depends on various factor, namely: having kids, family_status, income level and debt purpose.

# <h1> Step 4. General conclusion </h2>

# In general, the data has some missing value, which were evenly distributed in all categories in each row. Missing value are random 
# in nature. Since, missing value were quantitavive values so one possibility would be replace by average value. But, the average value do not concide with average value obtained by grouping subcategories in different columns.
# So, average value could change the average value of some categories. on the other hand, deleting missing rows does not have severe impact on any categories. Same educational level 
# were written in different format making confusion. so, all values in 'education column' are converted to lowercase. Likewise, there was typos on number of children. '20' was written in some rows
# which is not realistic. so, proper replacement for that was found by comparing other values related to every value of 'childern'. Lastly, 
# the repayment probability are determined by pivot table for different indices and 'debt' value. 
# 

# ### Project Readiness Checklist
# 
# 

# - [x]  file open;
# - [ x]  file examined;
# - [x ]  missing values defined;
# - [x ]  missing values are filled;
# - [x ]  an explanation of which missing value types were detected;
# - [x ]  explanation for the possible causes of missing values;
# - [x ]  an explanation of how the blanks are filled;
# - [x ]  replaced the real data type with an integer;
# - [ x]  an explanation of which method is used to change the data type and why;
# - [x ]  duplicates deleted;
# - [x ]  an explanation of which method is used to find and remove duplicates;
# - [x ]  description of the possible reasons for the appearance of duplicates in the data;
# - [ x]  data is categorized;
# - [x ]  an explanation of the principle of data categorization;
# - [x ]  an answer to the question "Is there a relation between having kids and repaying a loan on time?";
# - [x ]  an answer to the question " Is there a relation between marital status and repaying a loan on time?";
# - [x ]   an answer to the question " Is there a relation between income level and repaying a loan on time?";
# - [x ]  an answer to the question " How do different loan purposes affect on-time repayment of the loan?"
# - [x ]  conclusions are present on each stage;
# - [x ]  a general conclusion is made.

# <h3>  Thank yo very much!! </h3>
