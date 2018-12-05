import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loading the CSV dataset
titanic_df = pd.read_csv("titanic.csv")
# print(titanic_df)

# dropping the columns which are not very useful
titanic_df.drop(['parch', 'ticket', 'fare', 'cabin', 'boat', 'body', 'home.dest'], axis=1, inplace=True)
# print(titanic_df.head(5))

# filling in embarked = "S" i.e. Southampton for people with embarked field as blank
titanic_df['embarked'] = titanic_df['embarked'].fillna("S")

# renaming the embarked values to make it understandable
titanic_df["embarked"] = titanic_df['embarked'].map({
    "C": "Cherbourg",
    "Q": "Queenstown",
    "S": "Southampton"
})

# renaming the survived values to make it understandable
titanic_df["survived"] = titanic_df["survived"].map({
    0: "Not Survived",
    1: "Survived"
})

# print(titanic_df.head(10))

# using sns for better visualization
# visualization1 : passenger class vs number of people survived or dead
ax = sns.countplot(x="pclass", hue="survived", palette="Set1", data=titanic_df)
ax.set(title="Passengers survived according to class", xlabel="Passenger_Class", ylabel="Total")
plt.show()

# visualization2 : passenger embarked point vs number of people survived or dead
print(pd.crosstab(titanic_df["embarked"], titanic_df["survived"]))
ax = sns.countplot(x="embarked", hue="survived", data=titanic_df, palette="Set1")
ax.set(title="Embarked vs Survival Number", xlabel="Embarked", ylabel="Survived")
plt.show()

# visualization3: sex vs number of people survived or dead
ax = sns.countplot(x="sex", data=titanic_df, hue="survived", palette="Set2")
ax.set(title="Sex vs Survived", xlabel="Sex", ylabel="Survived")
plt.show()

# creating a list of ages to be used for categorization
age_intervals = [0, 18, 30, 60, 120]
# cretaing the age category types
categories = ['Children', 'Adult', 'Middle-Aged', 'Old']
# setting the categories of ages based on age_intervals 
titanic_df['Age_Categories'] = pd.cut(titanic_df.age, age_intervals, labels=categories)
# visualization4: Age categories vs number of people survived or dead
ax = sns.countplot(x="Age_Categories", hue="survived", palette="Set3", data=titanic_df)
ax.set(title="Age categories vs Number of survivors", xlabel="Age Groups", ylabel="Survived")
plt.show()
