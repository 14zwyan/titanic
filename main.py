import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score

'''
    ALl code is from https://blog.csdn.net/Flying_sfeng/article/details/77725890

'''



trainData=pd.read_csv('train.csv')
testData=pd.read_csv('test.csv')
combine=pd.concat( [ trainData.drop('Survived',axis=1),testData],axis=0)
print(trainData.describe())
print(trainData.info())
#print(trainData.head())
#print(trainData.tail())

# data analysis
survived = trainData[ trainData[ 'Survived' ] ==1]
non_survived= trainData[ trainData[ 'Survived' ] ==0]

surv_color='blue'
non_surv_color='red'

#Age
plt.figure(figsize=[12,10])
plt.subplot(331)
# 画连续特征可使用sns.distplot()
sns.distplot( survived['Age'].dropna().values, bins=range(0,80,1),
            kde=False, color=surv_color,axlabel='surv_Age')
plt.subplot(332)
sns.distplot( non_survived['Age'].dropna().values,bins=range(0,80,1),
            kde=False, color=non_surv_color, axlabel='non_surv_age')

#在某个特征下的幸存率
plt.subplot(333)
sns.barplot('Sex','Survived',data=trainData)
plt.subplot(334)
sns.barplot('Pclass','Survived',data=trainData)
plt.subplot(335)
sns.barplot('Embarked','Survived',data=trainData)
plt.subplot(336)
sns.barplot('Parch','Survived',data=trainData)
plt.subplot(337)
sns.barplot('SibSp','Survived',data=trainData)
# 因为Fare是连续值且分布范围广，可以对Fare取对数后再进行显示
plt.subplot(338)
sns.distplot( np.log10( survived['Fare'].dropna().values+1), kde=False,color=surv_color )
sns.distplot( np.log10( non_survived['Fare'].dropna().values + 1 ), kde=False, color=non_surv_color,
            axlabel='Fare')
#统计乘客的家庭成员个数并显示不同家庭成员数量下存活率
trainData['Family']=trainData['SibSp'] + trainData[ 'Parch' ]
plt.subplot(339)
sns.barplot( 'Family', 'Survived',data=trainData)
#plt.show()


combine['Family']=combine['SibSp'] + combine[ 'Parch' ]
#大于0小鱼4的值设为0，其他设为1
combine['FamilyBins']=np.where(  combine['Family']==0,1
        ,np.where(combine['Family']<4,0,1))
#由于Age 存在很多缺失值，使用corr看一下与年龄相关的特征由那些
plt.figure()
corr=sns.heatmap( trainData.drop('PassengerId',axis=1).corr(),
    vmax=0.8, annot=True)
#plt.show()


#与年龄相关的特征由‘Pclass’,'SIbSp','Parch'
group= combine.groupby( ['Pclass','SibSp','Parch']).Age
#tranform会将一个函数应用到各个分组，然后将结果放在适当的位置
combine['Age']=group.transform( lambda x: x.fillna( x.median()))

#探索在不同性别下，用户在不同年龄下的生存情况
msurv=trainData[ trainData['Survived']==1  & ( trainData['Sex']=='male')]
mnosurv=trainData[ trainData['Survived']==0  & (trainData['Sex']=='male')]
fsurv=trainData[ trainData['Survived']==1 & ( trainData['Sex']=='female')]
fnonsurv=trainData[ trainData['Survived']==0 & ( trainData['Sex']=='female')]
#分别显示在不同性别下，乘客在不同年龄下的幸存率
plt.figure()
plt.subplot(121)
sns.distplot( msurv['Age'].dropna().values, bins=range(0,80,1),kde=False,
    color=surv_color)
sns.distplot(mnosurv['Age'].dropna().values,bins=range(0,80,1),kde=False,
    color=non_surv_color,axlabel='maleAge')
plt.subplot(122)
sns.distplot( fsurv['Age'].dropna().values, bins=range(0,80,1),kde=False,color=surv_color)
sns.distplot( fnonsurv['Age'].dropna().values,bins=range(0,80,1),kde=False,
    color=non_surv_color,axlabel='feamaleAge')
#plt.show()

#在青中年阶段(18-40)，男性乘客的存活率明显偏低，女性乘客存活率明显片偏高
#因此，可以构造新特征，将年龄在18-40的男乘客归为0，女乘客为1，其他归为2
age_male_name= combine[ (combine['Age'] >=18 ) & (combine['Age']<40)
    & (combine['Sex']=='male')]['Name'].values
age_female_name= combine[  (combine['Age']>=18) & (combine['Age']<40)
    & (combine['Sex']=='female')]['Name'].values
combine['AgeClass']=np.where( combine['Name'].isin(age_male_name),0,
    np.where( combine['Name'].isin(age_female_name),1,2))

#统计不同Pclass,
plt.figure()
sns.violinplot(x='Pclass',y='Age',hue='Survived',data=trainData,split=True)
#splt=True能够把每个Pclass下，取Survived=0和Survived=1的每个图左右各一半
#这一句是标定了两条分界线（图中虚线），方便观察
#年龄比较小时，Pclass=1,2的孩子基本上都存货下来了，PcLASS=3时，有部分孩子没有存货下来
plt.hlines([0,12],xmin=-1,xmax=3,linestyles='dotted')
#plt.show()

#统计不同港口（Embarked)，不同Pclass下男性和女性的幸存率
sns.factorplot(x='Pclass',y='Survived',hue='Sex',col='Embarked',data=trainData)
#如果想将折线图转换为柱状图，可以加入kind参数，kind='bar'
sns.factorplot(x='Pclass',y='Survived',hue='Sex',col='Embarked',data=trainData,kind='bar')
#plt.show()

#可以看到pclass=1,2时，embarked=Qs时，sex=male的乘客没有存货的，sex=female的乘客全部存货
#可以分谢得到，这是因为在Q embark上pclass=2的乘客非常少
trainData[ (trainData['Pclass']==1) & ( trainData['Embarked']=='Q') ][ ['Sex','Survived']]
trainData[ (trainData['Pclass']==2) & ( trainData[ 'Embarked']=='Q')][['Sex','Survived']]

#将上图上幸存率较高和较低的情况分开，构成新的特征
PSM_name= combine[ ((combine['Pclass']<3) & (combine['Sex']=='female'))
                | ( (combine['Pclass']==3) & (combine['Sex']=='female')& (combine['Embarked']!='S'))]['Name'].values
#因为乘客中没有出现重名的情况，这里借用了名字的唯一性，使用Name作为中间变量
combine['PSM']=np.where( combine['Name'].isin(PSM_name),0,1 )

#统计不同港口，各Pclass等级乘客的比例，使用pd.crosstab()构造表格
tab=pd.crosstab( combine['Embarked'],combine['Pclass'] )
pic=tab.div(tab.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
pic=plt.xlabel('Embarked')
pic=plt.ylabel('Percent')
#plt.show()

#接下来分析Fare
#因为Fare的数值范围比较大[0:600]，所以我们可以先对Fare取对数，在进行分析
plt.figure()
sns.distplot( np.log10( survived['Fare'].dropna().values+1 ),kde=False,color=surv_color )
sns.distplot( np.log10( non_survived['Fare'].dropna().values+1),kde=False,color=non_surv_color,axlabel='Fare')
#可以看到，Fare越大，乘客幸存率越高
#plt.show()

#统计不同Pclass登记下Fare水平的幸存率，使用sns.boxplot()函数
sns.boxplot(x='Pclass',y='Fare',hue='Survived',data=trainData).set_yscale('log')
#plt.show()

#补全缺失值
#由上述相关关系表知道Fare相关性较大的特征由Pclass，Parch,SibSp，因此
nullFares= combine[ combine.Fare.isnull() ].index.values
combine.loc[nullFares,'Fare']= combine[ (combine['Pclass']==3)
 & (combine['Parch']==0) & (combine['SibSp']==0) ].Fare.median()

#补全Embarked的缺失值
#查明缺失值的信息
combine[ combine['Embarked'].isnull()]
#与缺失值相同特征的乘客数量
combine.where( (combine['Pclass']==1) & (combine['Sex']=='female') ).groupby(['Embarked','Pclass','Sex','Parch','SibSp']).size()
#观察可知，可以使用Embarked='C'补全缺失值
###finn nan values in Embarked
nullEmbarkeds= combine[ combine.Embarked.isnull()].index.values
combine['Embarked'].iloc[ nullEmbarkeds ] = 'C'

#构建分类模型

####offline model ###
## 交叉验证得到线下训练的准确率
'''
x_train=trainData['Fare','Age','Family','Embarked',
                'Sex','Pclass','AgeClass','SibSp','PSM',
                'Parch','FamilyBins']
'''



'''
x_train=np.concat(trainData['Fare'],trainData['Age'],trainData['Family'],
                  trainData['Embarked'],trainData['Sex'],trainData['Pclass'],
                  trainData[''])
'''
y_train=trainData['Survived']
x_train=trainData['Fare']
model= XGBClassifier( max_depth=6, n_estimator=1000, learning_rate=0.01 )
scores= cross_val_score(model,x_train,y_train,cv=3)
print('accuracy:{0:.5f}'.format(np.mean(scores)))
#使用xgboost的get_fscore得到特征的重要性并排序
model.fit(x_train,y_train)
importance= model.booster().get_fscore()
sort_importance= sorted( importance.items(), key=operator.itemgetter(1),reverse=True)
df=pd.DataFrame(sort_importance,columns=['feature','fscore'])
print(df)
