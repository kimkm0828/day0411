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
print(new_df.columns)
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
# print(lr.score(test_x,test_y))



# 45, Private, 386940, Bachelors, 13, Divorced, Exec-managerial, Own-child, White, Male, 0, 1408, 40, United-States, <=50K
sample_data = [[45, ' Private', 386940, ' Bachelors', 13, ' Divorced', ' Exec-managerial', ' Own-child',
          ' White', ' Male', 0, 1408, 40, ' United-States', ' <=50K']]

sample_df = pd.DataFrame(sample_data,columns=names)

sample_df = sample_df[['age','workclass','education','sex'
            ,'hours_per_week','occupation','income']]



data2 =data.append(sample_df)

one_hot = pd.get_dummies(data2)

pred_x = np.array( one_hot.iloc[-1,:-2] ).reshape(1,-1)
pred_y = lr.predict(pred_x)
msg = '대출 가능'
if pred_y == 1:
    msg = '대출불가능'

print(msg)






