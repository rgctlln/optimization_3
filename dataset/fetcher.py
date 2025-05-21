import pandas as pd

train_data = pd.read_csv('poker-hand-training-true.data', header=None)
test_data = pd.read_csv('poker-hand-testing.data', header=None)

columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'CLASS']
train_data.columns = columns
test_data.columns = columns

full_data = pd.concat([train_data, test_data], axis=0)

print(full_data.head())

full_data.to_csv('poker_hand_full_dataset.csv', index=False)
