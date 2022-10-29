import pandas as pd
import sys
import pickle
from sklearn.neighbors import KNeighborsClassifier
import warnings


train_data = pd.read_csv('training.csv')

train_data = train_data.drop_duplicates()
train_data = train_data.reset_index(drop = True)
y_train = train_data['cancer']
x_train = train_data.iloc[:,1:11]

model = KNeighborsClassifier()
model.fit(x_train, y_train)


if __name__ == "__main__":
    main()
