from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('../../data/Jester-Dataset-ratings.csv', delimiter=',')

train, test = train_test_split(df, test_size=0.2, random_state=1)

print("Train samples: " + str(len(train)))
print("Test samples: " + str(len(test)))

train.to_csv('../../data/Jester-train.csv')
test.to_csv('../../data/Jester-test.csv')
