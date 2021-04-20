import pandas as pd
import math


#                                       数据读取与预处理
path = 'NBA_Season_Stats(1984-2015)-new.csv'
data = pd.read_csv(path, encoding='gbk')
data = data.dropna()
data = data.reset_index(drop=True)
# 去掉没用属性
X = data.drop(['Pos'], 1)
X1 = data
y = data['Pos']
# 将Pos属性的值变为一个
for i in range(len(y)):
    x = str(y[i])
    index = x.find('-')
    if(index != -1):
        y[i] = x[0:index]
# 划分数据集
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=2)
X = Xtrain
y = ytrain
X = pd.DataFrame(X).reset_index(drop=True)
y = y.reset_index(drop=True)

            ############## 训练 ######
            ############## 计算先验概率与似然函数
# 先验概率：pri
count_each = {'C': 0, 'PF': 0, 'PG': 0, 'SF': 0, 'SG': 0}
pri = {'C': 0, 'PF': 0, 'PG': 0, 'SF': 0, 'SG': 0}
len_y = len(y)
for i in range(len(y)):
    print(y[i])
    count_each[y[i]] += 1
for i in pri:
    pri[i] = count_each[i]/len_y
#                                     似然 likelihold
likelihold = {'C': {}, 'PF': {}, 'PG': {}, 'SF': {}, 'SG': {}}
for pos in likelihold:
    X['Pos'] = y
    X2 = X[(X['Pos']) == pos]
    X2 = X2.drop(['Pos'], 1)
    for attr, row in X2.iteritems():
        attr_prob = {}
        attr_prob['mean'] = X2[attr].mean()
        attr_prob['std'] = X2[attr].std()
        likelihold[pos][attr] = attr_prob
print(likelihold)


# 测试
#
def test_all(Xtest, ytest):
    Xtest = pd.DataFrame(Xtest).reset_index(drop=True)
    ytest = ytest.reset_index(drop=True)
    print(Xtest, ytest)
    attrs = {}
    for attr, row in Xtest.iteritems():
        attrs[attr] = 0
    print(attrs)
    result = {'C': 0, 'PF': 0, 'PG': 0, 'SF': 0, 'SG': 0}
    res = 0
    for i in range(len(Xtest)):
        result = {'C': 0, 'PF': 0, 'PG': 0, 'SF': 0, 'SG': 0}
        row = Xtest.loc[i]
        # print(Xtest.loc[i])
        # for attr in Xtest.loc[i]:
                # print(attr)
        # print(row[]) # 输出每行的索引值
        for cla in result:
            prob = pri[cla]
            for attr in attrs:
                mean = likelihold[cla][attr]['mean']
                std = likelihold[cla][attr]['std']
                prob *= 1/math.sqrt(2*math.pi)/std*math.exp(-1*pow((row[attr]-mean),2)/2/pow(std,2))
            result[cla] = prob
            #print(i, "   ", result)
        print("第", i, "个测试的分类为", max(result, key=result.get))
        if ytest.loc[i] == max(result, key=result.get):
            res += 1
    print("正确率为     ", res/len(ytest))
        # for attr, row in Xtest.iteritems():
            # prob *= likelihold[cla][attr]

#                                             test
def test():

    # 0.58 0.7 7 3 1 22
    # input
    test_class = input("请输入类别")
    arr = list(map(float, input("请输入各属性值").split()))
    testcase = {'2P%': arr[0], 'FT%': arr[1], 'TRB': arr[2], 'AST': arr[3], 'STL': arr[4], 'PTS': arr[5]}

    #               直接填写
    # #testcase = {'2P%': 0.58, 'FT%': 0.7, 'TRB': 7, 'AST': 3, 'STL': 1, 'PTS': 22}
    #
    # testcase = {'2P%': 0.53, 'FT%': 0.7, 'TRB': 6, 'AST': 4, 'STL': 1, 'PTS': 30}
    # test_class = 'SF'
    result = {'C': 0, 'PF': 0, 'PG': 0, 'SF': 0, 'SG': 0}
    prob = 0
    for cla in result:
        prob = pri[cla]
        for attr in testcase:
            mean = likelihold[cla][attr]['mean']
            std = likelihold[cla][attr]['std']
            prob *= 1 / math.sqrt(2 * math.pi) / std * math.exp(-1 * pow((testcase[attr] - mean), 2) / 2 / pow(std, 2))
        result[cla] = prob
        # print(i, "   ", result)
    print(result)
    print("测试的分类为", max(result, key=result.get), "实际分类为",test_class)

test_all(Xtest, ytest)
# test()

# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()
# model.fit(Xtrain, ytrain)
# y_model = model.predict(Xtest)
#
# from sklearn.metrics import accuracy_score
# print(accuracy_score(ytest, y_model))
