
# coding: utf-8

# In[2]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[3]:


digits = load_digits()
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
logisticRegression = KNeighborsClassifier()
logisticRegression.fit(x_train, y_train)
predictions = logisticRegression.predict(x_test)
scores = logisticRegression.score(x_test, y_test)
print(scores)


# In[5]:


file1 = open("result.txt", "w")
file1.write(str(scores*100))
file1.close()


# In[ ]:




