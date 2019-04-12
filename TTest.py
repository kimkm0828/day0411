from day0411 import oneHotUtil
import numpy as np
info = '38, Private, 215646, HS-grad, 9, Divorced, Handlers-cleaners, Not-in-family, White, Male, 0, 0, 40, United-States, <=50K'
    #   0      1       2       3      4   5         6                   7              8     9    10 11 12   13             14
# a = np.array(info.split(','))
# a[0] = int(a[0])
# a[2] = int(a[2])
# a[4] = int(a[4])
# a[10] = int(a[10])
# a[11] = int(a[11])
# a[12] = int(a[12])
# x = []
# for i  in a:
#     x.append(i)
# print(x)
# print(a[[0,2,4,10,11,12]])




# age = 38
# workclass = 'Private'
# fnlwgt = 215646
# education = 'HS-grad'
# education_num = 9
# marital_status = 'Divorced'
# occupation = 'Handlers-cleaners'
# relationship = 'Not-in-family'
# race = 'White'
# sex = 'Male'
# capital_gain = 0
# capital_loss = 0
# hours_per_week = 40
# native_country = 'United-States'
# income = '<=50K'
# msg = oneHotUtil.oneHotTest(age, workclass, fnlwgt, education, education_num, marital_status,occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country, income)
# # msg = oneHotUtil.oneHotTest(info)
#
# print(msg)


domain = oneHotUtil.getDomain()
# print(workclass)
# print(education)
# print(occupation)
# print(race)
# print(sex)
print(domain[0])