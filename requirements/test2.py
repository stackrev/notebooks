#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


data = pd.read_csv("./500_Person_Gender_Height_Weight_Index.csv")
data.sample(3)


# In[4]:


data.drop("Index", inplace=True, axis=1)
data


# In[5]:


data.shape


# In[6]:


data.isna().sum()


# In[7]:


data["Height"].min()


# In[8]:


def get_median(series):
    median = None
    ordered = series.sort_values()
    size = len(series)
    if size % 2 == 0:
        median = (ordered[size/2 - 1] + ordered[size/2])/2
    else:
        median =  ordered[size/2]
    return median

med =  get_median(data["Weight"])
med


# In[9]:


data["Weight"].median()


# In[10]:


plt.figure(figsize=(5,4))
data["Weight"].hist(bins=30)
plt.axvline(data["Weight"].mean(), color='r', label="mean")
plt.legend()


# In[11]:


h = data[["Height"]].copy()
h["c"] = 1
h = h.groupby("Height", as_index=False).count()
h["cc"] = h["c"].cumsum()
h.head(5)


# In[12]:


q1 = h["Height"].quantile(.25)
q3 = h["Height"].quantile(.75)
print(f"{q1} -> {q3}: {q3 - q1}")

plt.figure(figsize=(6,4))
data["Height"].hist(bins=20)

plt.axvline(q1, color="r", label="q1")
plt.axvline(q3, color="g", label="q3")

plt.legend()


# In[13]:


plt.scatter(data["Weight"], data["Height"], s=100)
plt.axvline(data["Weight"].median(), color="r", label="median")
plt.axhline(data["Height"].median(), color="g", label="median")


# In[14]:


plt.bar(h["Height"], h["cc"])
plt.axvline(q1, color="y", label="q1")
plt.axvline(q3, color="g", label="q3")


# In[15]:


data.describe()


# In[16]:


import seaborn as sns
sns.set_theme(style="whitegrid")
sns.countplot(x=data["Gender"])
plt.show


# In[17]:


data["Height"].plot(kind="hist", title="heights")


# In[18]:


data["Height"].plot(kind="box", title="Boxplot")


# In[19]:


data["Height"].plot(kind="kde")


# In[20]:


sns.scatterplot(x ="Height", y="Weight", data=data, hue="Gender")


# In[41]:


g = sns.FacetGrid(data, hue="Gender")
g.map(sns.kdeplot, "Height", fill=True, common_norm=False, palette="crest", alpha=.5, linewidth=0)
g.add_legend()

