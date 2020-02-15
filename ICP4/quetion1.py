import pandas as pd

dataset = pd.read_csv('train.csv')

dataset['Sex'] =dataset['Sex'].map( {'female': 1, 'male': 0} )


print(dataset['Survived'].corr(dataset['Sex']))