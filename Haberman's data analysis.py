#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

COVID_DATA = pd.read_csv(r'C:\Users\Hello\Downloads\haberman (1).csv')
COVID_DATA


# In[2]:


COVID_DATA.head(3)


# In[7]:


COVID_DATA.shape


# In[33]:


COVID_DATA.columns


# In[34]:


COVID_DATA.count()


# In[35]:


COVID_DATA.isnull().sum()


# In[3]:


column_names = ['Age', 'Year', 'Axillary_Nodes', 'Survival_chances']
COVID_DATA.columns = column_names
COVID_DATA.head()


# In[24]:


COVID_DATA.columns


# In[4]:


COVID_DATA['Survival_chances'] = COVID_DATA['Survival_chances'].map({1: 'yes', 2: 'No'})
COVID_DATA['Survival_chances'] = COVID_DATA['Survival_chances'].astype('category')
COVID_DATA.head()


# In[4]:


COVID_DATA.tail()


# In[5]:


COVID_DATA.describe()


# In[6]:


COVID_DATA["Survival_chances"].value_counts(1)


# In[7]:


COVID_DATA.groupby('Survival_chances')['Age'].sum()


# In[27]:


COVID_DATA['Survival_chances'].describe()


# In[8]:


COVID_DATA.corr()


# In[9]:


sns.heatmap(COVID_DATA.corr())


# In[45]:


sns.set_style = ("greengrid")
sns.FacetGrid(COVID_DATA, hue="Survival_chances", size=5).map(sns.distplot, "Age").add_legend()


# In[19]:


import seaborn as sns 
sns.FacetGrid(COVID_DATA, hue="Survival_chances", height = 5).map(plt.scatter, "Age", "Axillary_Nodes").add_legend()


# In[64]:


sns.pairplot(COVID_DATA, hue = 'Survival_chances', vars = ['Age','Year', 'Axillary_Nodes'], height = 5)


# In[11]:


count_classes = pd.value_counts(COVID_DATA['Survival_chances'])
count_classes.plot(kind = 'bar')
plt.title("Class distribution Histogram")
plt.xlabel("Survival_chances")
plt.ylabel("Frequency")
plt.show()


# In[74]:


sns.FacetGrid(COVID_DATA, hue='Survival_chances', size=8).map(sns.distplot,'Axillary_Nodes').add_legend()
plt.show()


# In[70]:


sns.distplot(COVID_DATA['Age'], kde = True, bins = 10)


# In[38]:


sns.countplot('Survival_chances', data = COVID_DATA)


# In[34]:


sns.boxplot(x = 'Survival_chances', y = 'Age', hue = "Survival_chances", data = COVID_DATA)
plt.show()
sns.boxplot(x = 'Survival_chances', y = 'Year', hue = "Survival_chances",  data = COVID_DATA)
plt.show()
sns.boxplot(x = 'Survival_chances', y = 'Axillary_Nodes', hue = "Survival_chances",  data = COVID_DATA)
plt.show()
sns.boxplot(data = COVID_DATA, orient = 'V')
plt.show()


# In[29]:


sns.violinplot(x = 'Survival_chances', y = 'Age', hue = "Survival_chances", data = COVID_DATA)
plt.show()
sns.violinplot(x = 'Survival_chances', y = 'Year', hue = "Survival_chances",  data = COVID_DATA)
plt.show()
sns.violinplot(x = 'Survival_chances', y = 'Axillary_Nodes', hue = "Survival_chances",  data = COVID_DATA)
plt.show()


# In[59]:


mean = np.mean(COVID_DATA['Age'])
print(mean)
var = np.var(COVID_DATA['Age'])
print(var)
std = np.std(COVID_DATA['Age'])
print(std)
median = np.median(COVID_DATA['Age'])
print(median)


# In[80]:


counts, bin_edges = np.histogram(COVID_DATA[['Age']], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.show()
counts, bin_edges = np.histogram(COVID_DATA[['Year']], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.show()
counts, bin_edges = np.histogram(COVID_DATA[['Axillary_Nodes']], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.show()


# In[ ]:





# In[ ]:




