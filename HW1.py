import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

# Q1
print(list(train_df.columns))

# Q2
train_df.select_dtypes(include=object)

# Q3
train_df.select_dtypes(include=np.number)

# Q4
# answer The Tickets column
# Create a sample DataFrame
# Find columns with mixed datatypes

mixed_columns = []

for column in train_df.columns:
    unique_types = train_df[column].apply(type).unique()
    print(f'unique types for {column} feature: {unique_types}')
    if len(unique_types) > 1:
        mixed_columns.append(column)

print("Columns with mixed datatypes:")
print(mixed_columns)


# Q5
print(train_df.isna().any())
print(test_df.isna().any())

# Q6
train_df.dtypes

# Q7
train_df.describe(include=[float, int])

# Q8
train_df.astype('category').describe()

# Q9
# There is a correlation between the Pclass=1 and Survived:
upper_class = train_df.loc[train_df['Pclass'] == 1]['Pclass']
survived = train_df.loc[train_df['Survived'] == 1]['Survived']
print(upper_class.corr(survived)) # prints nan

# and there is a negative correlation of 33%:
print(train_df['Pclass'].corr(train_df['Survived'])) # yields -0.33848103596101586

# Q10
female_survived_cnt = train_df[(train_df.Sex == 'female') & (train_df.Survived == 1)].Survived.count()
# print(female_survived_cnt)
female_total = train_df[(train_df.Sex == 'female')].Sex.count()
# print(female_total)
print('percentage of female that survived: {0:.3g}%'.format((female_survived_cnt / female_total) * 100))

male_survived_cnt = train_df[(train_df.Sex == 'male') & (train_df.Survived == 1)].Survived.count()
# print(male_survived_cnt)
male_total = train_df[(train_df.Sex == 'male')].Sex.count()
# print(male_total)
print('percentage of male that survived: {0:.3g}%'.format((male_survived_cnt / male_total) * 100))


# Q11
pclass1_deaths = train_df[(train_df.Pclass == 1) & (train_df.Survived == 0)]['Age']
pclass2_deaths = train_df[(train_df.Pclass == 2) & (train_df.Survived == 0)]['Age']
pclass3_deaths = train_df[(train_df.Pclass == 3) & (train_df.Survived == 0)]['Age']

pclass1_survivors = train_df[(train_df.Pclass == 1) & (train_df.Survived == 1)]['Age']
pclass2_survivors = train_df[(train_df.Pclass == 2) & (train_df.Survived == 1)]['Age']
pclass3_survivors = train_df[(train_df.Pclass == 3) & (train_df.Survived == 1)]['Age']



train_died = train_df.query('Survived == 0') # subset of data for records that did not survive
train_survived = train_df.query('Survived == 1') # subset of data for records that survived

# histogram of survived vs deaths by ages
plt.figure('Ttianic surival vs death') # create and name the figure
plt.subplot(121) # subplot 1 (rows 1, columns 2, plot number)
plt.hist(train_died['Age'], color='r', edgecolor='black') # create histogram of age from query (did not survive)
plt.title('Survived = 0') # title subplot
plt.xlabel('Age') # label X-Axis
plt.ylabel('Frequency') # label y-Axis

plt.subplot(122) # subplot 2 (rows 1, columns 2, plot number)
plt.hist(train_survived['Age'], color='g', edgecolor='black') # create histogram of age from query (survived)
plt.title('Survived = 1') # title subplot
plt.xlabel('Age') # label X-Axis
plt.ylabel('Frequency') # label y-Axis
plt.show() # display histograms



# Q12
plt.figure('Ttianic surival vs death') # create and name the figure

plt.hist(pclass1_deaths, edgecolor='black', bins=20)
plt.title('Pclass = 1 | Survived = 0') # title subplot
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.xlim(0, 80)
plt.ylim(0, 60)
plt.show()

plt.hist(pclass1_survivors, edgecolor='black', bins=20)
plt.title('Pclass = 1 | Survived = 1') # title subplot
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.xlim(0, 80)
plt.ylim(0, 60)
plt.show()

plt.hist(pclass2_deaths, edgecolor='black', bins=20)
plt.title('Pclass = 2 | Survived = 0') # title subplot
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.xlim(0, 80)
plt.ylim(0, 60)
plt.show()

plt.hist(pclass2_survivors, edgecolor='black', bins=20)
plt.title('Pclass = 2 | Survived = 1') # title subplot
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.xlim(0, 80)
plt.ylim(0, 60)
plt.show()

plt.hist(pclass3_deaths, edgecolor='black', bins=20)
plt.title('Pclass = 3 | Survived = 0') # title subplot
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.xlim(0, 80)
plt.ylim(0, 50)
plt.show()

plt.hist(pclass3_survivors, edgecolor='black', bins=20)
plt.title('Pclass = 3 | Survived = 1') # title subplot
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.xlim(0, 80)
plt.ylim(0, 60)
plt.show()

# Q13
survived = [0, 1]
embarked = ['S', 'C', 'Q']


sex_ES_MD = train_df[(train_df.Embarked == 'S') & (train_df.Survived == 0) & (train_df.Sex == 'male')]['Fare']
sex_ES_FD = train_df[(train_df.Embarked == 'S') & (train_df.Survived == 0) & (train_df.Sex == 'female')]['Fare']

sex_EC_MD = train_df[(train_df.Embarked == 'C') & (train_df.Survived == 0) & (train_df.Sex == 'male')]['Fare']
sex_EC_FD = train_df[(train_df.Embarked == 'C') & (train_df.Survived == 0) & (train_df.Sex == 'female')]['Fare']

sex_EQ_MD = train_df[(train_df.Embarked == 'Q') & (train_df.Survived == 0) & (train_df.Sex == 'male')]['Fare']
sex_EQ_FD = train_df[(train_df.Embarked == 'Q') & (train_df.Survived == 0) & (train_df.Sex == 'female')]['Fare']



sex_ES_MS = train_df[(train_df.Embarked == 'S') & (train_df.Survived == 1) & (train_df.Sex == 'male')]['Fare']
sex_ES_FS = train_df[(train_df.Embarked == 'S') & (train_df.Survived == 1) & (train_df.Sex == 'female')]['Fare']

sex_EC_MS = train_df[(train_df.Embarked == 'C') & (train_df.Survived == 1) & (train_df.Sex == 'male')]['Fare']
sex_EC_FS = train_df[(train_df.Embarked == 'C') & (train_df.Survived == 1) & (train_df.Sex == 'female')]['Fare']

sex_EQ_MS = train_df[(train_df.Embarked == 'Q') & (train_df.Survived == 1) & (train_df.Sex == 'male')]['Fare']
sex_EQ_FS = train_df[(train_df.Embarked == 'Q') & (train_df.Survived == 1) & (train_df.Sex == 'female')]['Fare']

# print(train_df['Fare'].sum())


# deaths
plt.figure()
genders = ['Male', 'Female']
# averages are stored in fares
# must be list of numeric datatypes
fares = [sex_ES_MD.sum() / len(sex_ES_MD), sex_ES_FD.sum() / len(sex_ES_FD)]
plt.bar(genders, fares, color='green')
plt.title('Embarked = S | Survived = 0')
plt.xlabel('Sex')
plt.ylabel('Fare')
plt.show()

plt.figure()
genders = ['Male', 'Female']
fares = [sex_EC_MD.sum() / len(sex_EC_MD), sex_EC_FD.sum() / len(sex_EC_FD)]
plt.bar(genders, fares, color='green')
plt.title('Embarked = C | Survived = 0')
plt.xlabel('Sex')
plt.ylabel('Fare')
plt.show()


plt.figure()
genders = ['Male', 'Female']
fares = [sex_EQ_MD.sum() / len(sex_EQ_MD), sex_EQ_FD.sum() / len(sex_EQ_FD)]
plt.bar(genders, fares, color='green')
plt.title('Embarked = Q | Survived = 0')
plt.xlabel('Sex')
plt.ylabel('Fare')
plt.show()


# # survivors
plt.figure()
genders = ['Male', 'Female']
fares = [sex_ES_MS.sum() / len(sex_ES_MS), sex_ES_FS.sum() / len(sex_ES_FS)]
plt.bar(genders, fares, color='green')
plt.title('Embarked = S | Survived = 1')
plt.xlabel('Sex')
plt.ylabel('Fare')
plt.show()

plt.figure()
genders = ['Male', 'Female']
fares = [sex_EC_MS.sum() / len(sex_EC_MS), sex_EC_FS.sum() / len(sex_EC_FS)]
plt.bar(genders, fares, color='green')
plt.title('Embarked = C | Survived = 1')
plt.xlabel('Sex')
plt.ylabel('Fare')
plt.show()

plt.figure()
genders = ['Male', 'Female']
fares = [sex_EQ_MS.sum() / len(sex_EQ_MS), sex_EQ_FS.sum() / len(sex_EQ_FS)]
plt.bar(genders, fares, color='green')
plt.title('Embarked = Q | Survived = 1')
plt.xlabel('Sex')
plt.ylabel('Fare')
plt.show()

# higher paying passengers generally do have a better chance of survival but not always. For instance
# the graph of Embarked = S had a lot of male passenger deaths with high paying fares
# I think we should drop fare feature because of the outlier graph


# Q14
# total number of duplicate tickets / total number of tickets
print('{} %'.format((train_df['Ticket'].duplicated().sum() / train_df['Ticket'].count()) * 100))
# ticket is mixed data type and survival is just integers, so no correlation
# we should drop bc theres no correlation
# print(train_df['Ticket'].duplicated().sum())
# print(train_df['Ticket'].count())



# Q15
# number of null values in cabin feature of both training and testing data set:
# 687 + 327 = 1014

# We should drop the cabin feature. There are too many null values to help train the model

print(train_df['Cabin'].isna().sum())
print(train_df['Cabin'].count())
print(len(train_df['Cabin']))

print()

print(test_df['Cabin'].isna().sum())
print(test_df['Cabin'].count())
print(len(test_df['Cabin']))

# Q16
def gender_to_int(s: str) -> int:
  if s == 'male': return 0
  if s == 'female': return 1
  
  print('invalid string')
  return -1

train_df['Sex_in_numerical'] = train_df['Sex'].apply(gender_to_int)

print(train_df['Sex_in_numerical'])



# Q17
age_mean = train_df['Age'].mean()

# print(age_mean)
# print(train_df['Age'].isna().values.any())

# might have to put inplace = True
train_df['Age'].fillna(value=age_mean)


# Q18
# shows the number of null values in the embarked column
# res = filter(lambda b: b, train_df['Embarked'].isna().values)
# print(len(list(res)))

embarked_mode = train_df['Embarked'].mode()
# print(embarked_mode[0])
# print(train_df['Embarked'].isna().values.any())
train_df['Embarked'].fillna(value=embarked_mode[0], inplace=True)
# print(train_df['Embarked'].count())
# print(train_df['Embarked'].isna().values.any())
# train_df.dtypes


# Q19

# number of null values:
# len(test_df['Fare']) - test_df['Fare'].count()


test_df_fare_mode = test_df['Fare'].mode()
print(test_df_fare_mode[0])

print(test_df['Fare'].isna().any())
test_df['Fare'].fillna(value=test_df_fare_mode[0], inplace=True)
print(test_df['Fare'].isna().any())



# Q20
# 0 in train_df['Fare'].values # checks whether the Fare column contains 0
# train_df['Embarked'].isna().sum()  #counts all null values in the embarked column

def fare_to_ordinal_fare(fare: float) -> int:
  if fare <= -0.001: return -1

  if fare > -0.001 and fare <= 7.91: return 0
  if fare > 7.91 and fare <= 14.454: return 1
  if fare > 14.454 and fare <= 31.0: return 2
  if fare > 31.0 and fare <= 512.329: return 3

# train_df['Fare'].head()

train_df['Ordinal_Fare'] = train_df['Fare'].apply(fare_to_ordinal_fare)

train_df['Ordinal_Fare'].isna().any()
train_df['Ordinal_Fare'].dropna().astype(int)

