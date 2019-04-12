import pandas as pd
import numpy as np
from sklearn import linear_model,model_selection

# sklean  ==>> preprocessing.Binarizer()
# 2차원 배열상대


# preprocessing.LabelBinarizer()
# 1차원 배열 상대 , 문자도 가능

# pandas의           get_dummies 1,2차원, 컬럼 이름까지 생성
# 배열 인코딩을 요구하면 숫자인 feature는 one hot encoding에서 제외



names = ['age','workclass','fnlwgt','education','education_num','marital-status',
         'occupation','relationship','race','sex','capital_gain'
    ,'capital_loss','hours_per_week','native_country','income']

data = pd.read_csv('../Data/adult.data.txt',header=None, names=names)
# print(data.head(1))


data = data[['age','workclass','education','sex'
            ,'hours_per_week','occupation','income']]

# cnt_workclass = data['workclass'].unique()
# print(len(cnt_workclass))

# b_data = pd.get_dummies(data)
# print(b_data.head)



new_df = pd.get_dummies(data)
# 원래 있는 데이터 프레임에 feature를 주는 새로운 데이터 프레임을 생성
# print(new_df.head())
# print(new_df.columns)


# 위의 데이터로 부터 문제는 x에 답은 y에 ㄱ
# x = new_df.loc[:,'age':'sex_ Male']
# y = new_df.loc[:,'income_ >50K']

x = new_df.iloc[:,:-2]
y = new_df.iloc[:,-1]
# print(x.columns)
# print(y.head())


# 문제와 답 차수 확인
# print(x.shape)          #(32561, 44)    2차원
# print(y.shape)          #(32561,)       1차원

# 문제와 답의 차수가 동일 한가요?
# 동일하지 않으면
# 차수를 변경해야 할수도 있다리
# print(y.reshape(-1))

# 문제와 답을 훈련에 참여시킬 데이타와 검증을 위한 데이터로 분리
train_x, test_x,train_y, test_y = model_selection.train_test_split(x,y)
# print("test_x",test_x) # 문제
# print("test_y",test_y) # 답
# print("train_x",train_x) # 문제
# print("train_y",train_y) # 답


# print(len(train_x))
# print(len(train_y))

lr = linear_model.LogisticRegression()
lr.fit(train_x,train_y) # 훈련용 문제와 답
r = lr.predict(test_x)  # 검증용 답을 제시
result = r == test_y    # 검증용 답과 실제 답을 비교

result = result.values  # 검증결과가 시리즈라 values만 뽑아온다
print(result[0])
b = len(result[result == True])
print(b) # 총갯수8141     /// 맞은 갯수 6617

print("정답률",b/len(test_y)*100)

# 정답률 알기 위해 LogisticRegression의 score함수를 이용
print(lr.score(test_x,test_y))














