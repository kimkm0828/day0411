import pandas as pd
import numpy as np
from sklearn import linear_model,model_selection
names = ['age','workclass','fnlwgt','education','education_num','marital-status',
         'occupation','relationship','race','sex','capital_gain'
    ,'capital_loss','hours_per_week','native_country','income']

data = pd.read_csv('../Data/adult.data.txt',header=None, names=names)
# print(data.head(1))


data = data[['age','workclass','education','sex'
            ,'hours_per_week','occupation','income']]


# b_data = pd.get_dummies(data)
# print(b_data.head)

new_df = pd.get_dummies(data)

x = new_df.iloc[:,:-2]
y = new_df.iloc[:,-1]

# 문제와 답을 훈련에 참여시킬 데이타와 검증을 위한 데이터로 분리
train_x, test_x,train_y, test_y = model_selection.train_test_split(x,y)

# print(len(train_x))
# print(len(train_y))

lr = linear_model.LogisticRegression()
lr.fit(train_x,train_y) # 훈련용 문제와 답
r = lr.predict(test_x)  # 검증용 답을 제시
result = r == test_y    # 검증용 답과 실제 답을 비교

result = result.values  # 검증결과가 시리즈라 values만 뽑아온다
print(lr.score(test_x,test_y))















