import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statistics import mean


a=pd.read_csv("geo_a/results", sep=' ', header=None)
a['geo']="a"
temp=pd.read_csv("geo_a/data.csv",header=None, sep='\t')
a=pd.merge(a, temp, how='left', left_on=[0], right_on=[1])
temp=pd.read_csv("geo_a/ansys.csv",header=None, sep=',')
a=pd.merge(a, temp, how='left', left_on=["0_y"], right_on=[0])
a.drop(["0_x","1_y","0_y",0,2], axis=1, inplace=True)
a.sort_values(by=[1], inplace=True)


"""

#plot y+

temp=pd.read_csv("results7_y15", sep=' ', header=None)
a=pd.merge(a, temp, how='left', left_on=["key_0"], right_on=[0])
a.rename(columns = {'1_y':'y+ = 15'}, inplace = True)
temp=pd.read_csv("results7_y5", sep=' ', header=None)
a=pd.merge(a, temp, how='left', left_on=["key_0"], right_on=[0])
a.rename(columns = {1:'y+ =  5'}, inplace = True)
temp=pd.read_csv("results7_y30", sep=' ', header=None)
a=pd.merge(a, temp, how='left', left_on=["key_0"], right_on=[0])
a.rename(columns = {1:'y+ = 30'}, inplace = True)
temp=pd.read_csv("results7_y60", sep=' ', header=None)
a=pd.merge(a, temp, how='left', left_on=["key_0"], right_on=[0])
a.rename(columns = {1:'y+ = 60'}, inplace = True)

temp=pd.read_csv("results7_y10", sep=' ', header=None)
a=pd.merge(a, temp, how='left', left_on=["key_0"], right_on=[0])
a.rename(columns = {1:'y+ = 10'}, inplace = True)


new_columns = a.columns.values
new_columns[0] = 'order'
new_columns[3] = 3
#new_columns[8] = 12
a.columns = new_columns




fig, ax = plt.subplots()
fig.set_size_inches(13,7)

plt.plot(a[3],a["y+ =  5"],c="red")
plt.plot(a[3],a["y+ = 10"],c="orange")
plt.plot(a[3],a["y+ = 15"],c="springgreen")
plt.plot(a[3],a["y+ = 30"],c="black")
plt.plot(a[3],a["y+ = 60"])

ax.legend()

plt.scatter(a[3], a["y+ =  5"],4,c="red",marker="x")
plt.scatter(a[3], a["y+ = 10"],4,c="orange",marker="x")
plt.scatter(a[3], a["y+ = 15"],4,c="springgreen",marker="x")
plt.scatter(a[3], a["y+ = 30"],4,c="black",marker="x")
plt.scatter(a[3], a["y+ = 60"],4,c="blue",marker="x")

plt.title('Współczynnik oporu dla geometrii A - porównanie wartosci Y+', fontsize=18)
plt.xlabel('Parametr A1 [mm]', fontsize=14)
plt.ylabel('Współczynnik oporu [1]', fontsize=14)
plt.xticks(a[3], range(25,51), rotation=0)
plt.grid(color='gray', linestyle='-', linewidth=0.7)
ax.set_facecolor('xkcd:white')

"""





"""

#fig=plt.figure(figsize=(10,6))
fig, ax = plt.subplots()
fig.set_size_inches(10,6)
plt.plot(a[3],a["1_x"])
plt.scatter(a[3], a["1_x"],4,c="blue",marker="x")
plt.title('Współczynnik oporu dla geometrii A', fontsize=15)
plt.xlabel('Parametr A1 [mm]')
plt.ylabel('Współczynnik oporu [1]')
plt.xticks(a[3], range(25,51), rotation=0)
#plt.yticks(a["1_x"], np.arange(8.6,9.6,0.10), rotation=20)
plt.grid(color='gray', linestyle='-', linewidth=0.7)
ax.set_facecolor('xkcd:white')

plt.savefig('foo_a.png')
"""

b=pd.read_csv("geo_b/results", sep=' ', header=None)
b.sort_values(by=[0], inplace=True)
b['geo']="b"
temp=pd.read_csv("geo_b/data.csv",header=None, sep='\t')
b=pd.merge(b, temp, how='left', left_on=[0], right_on=[1])
temp=pd.read_csv("geo_b/ansys.csv",header=None, sep=',')
b=pd.merge(b, temp, how='left', left_on=["0_y"], right_on=[0])
b.drop(["0_x","1_y","0_y",0,3], axis=1, inplace=True)
"""
#plot 





b_graph=b.pivot(1,2,"1_x")
ax = plt.axes()
sns.set()
sns.heatmap(b_graph, linewidths=.5, annot=True, fmt=".4f", cmap='jet'
            , xticklabels=range(25,46,5), yticklabels=range(5,66,5), center=0.225)
ax.set_title("Współczynnik oporu dla geometrii B", fontsize=15)
ax.set_xlabel("Parametr B2 [mm]")
ax.set_ylabel("Parametr B1 [mm]")

fig = heatmap.get_figure()
fig.savefig('heatmap.pdf')
"""


c=pd.read_csv("geo_c/results", sep=' ', header=None)
c.sort_values(by=[0], inplace=True)
c['geo']="c"
temp=pd.read_csv("geo_c/data.csv",header=None, sep='\t')
c=pd.merge(c, temp, how='left', left_on=[0], right_on=[1])
temp=pd.read_csv("geo_c/ansys.csv",header=None, sep=',')
c=pd.merge(c, temp, how='left', left_on=["0_y"], right_on=[0])
c.drop(["0_x","1_y","0_y",0,3], axis=1, inplace=True)

"""
#plot c 


c_graph=c.pivot(2,1,"1_x")
ax = plt.axes()
sns.set()
#for x1 in range(930,970,1):
fig=sns.heatmap(c_graph, linewidths=.5, annot=True, fmt=".4f", cmap='jet', xticklabels=range(35,66,10),
            yticklabels=range(20,60,3), cbar=True, robust=False, center=0.1)
ax.set_title("Współczynnik oporu dla geometrii C", fontsize=15)
ax.set_xlabel("Parametr C1 [mm]")
ax.set_ylabel("Parametr C2 [°]")

fig = fig.get_figure()
fig.savefig('temp/'+str(int(x1))+'.png')
    #fig.close()
"""

df=a.append(b, ignore_index=True)
df=df.append(c, ignore_index=True)



def stats(x,y):
    temp=pd.read_csv("geo_"+y+"/"+str(x)+"0.csv", sep=',')
    X=temp["Points:0"]
    Y=temp["Points:1"]

    numPoints=len(X)


    derivI=[]
    derivII=[]
    arr=[]
    lenght=0
        
    for i in range(0, numPoints-1):
        derivI.append((Y[i+1]-Y[i])/(X[i+1]-X[i]))
        lenght+=(Y[i+1]-Y[i])**2+(X[i+1]-X[i])**2
    lenght=lenght**0.5
    
    for j in range(0, numPoints-2):
        derivII.append((Y[j+2]-2*Y[j+1]+Y[j])/(X[j+1]-X[j])**2)
        

    deriv=[derivI,derivII]  
    arr.append(lenght)
    arr.append(max(Y))
    arr.append(max(X)-min(X))
    
    
        
    for i in deriv:      
        
        arr.append(max(i))
        arr.append(min(i))
        arr.append(mean(i))
        arr.append(sum(i))
        
        for j in range(10,91,10):
            arr.append(np.percentile(i, j))
        
        delta=round(len(i)/10)
        ranges=[0]
        for j in range(1,10):
            ranges.append(j*delta)
        ranges.append(len(i))
        
        for j in range(1,11):
            arr.append(mean(i[j:j+1]))
            
    
    return arr


data=[]
for index, row in df.iterrows():
    
    data.append(stats(df["key_0"][index],df["geo"][index]))
    
dataAx=data[0:26]
dataAy=df['1_x'][0:26].values

#testAx=data[12]
#testAy=df['1_x'][12]


dataBx=data[26:39]
dataBy=df['1_x'][26:39].values#58

testBx=data[39:91]
testBy=df['1_x'][39:91].values

dataCx=data[123:124]
dataCy=df['1_x'][123:124].values#117

testCx=data[91:123]+data[124:]
testCy=df['1_x'][np.r_[91:123,124:147]].values

    



from sklearn.model_selection import train_test_split

X_train=dataAx+dataBx+dataCx
y_train=list(dataAy)+list(dataBy)+list(dataCy)


X_train, X_test, y_train, y_test= train_test_split(X_train, y_train, test_size = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
testBx = sc.transform(testBx)
testCx = sc.transform(testCx)



from keras.models import Sequential
from keras.layers import Dense






from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

def create_model(optimizer="Adamax", activation='relu', hidden=1):

    regressor = Sequential()
    
    regressor.add(Dense(output_dim = 49, init = 'uniform', activation = activation, input_dim = 49))
    
    for i in range(hidden):
        regressor.add(Dense(output_dim = 49, init = 'uniform', activation = activation))
    
    
    regressor.add(Dense(output_dim = 1, init = 'uniform', activation = 'relu'))
    
    regressor.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['mean_squared_error'])
    
   
    
    return regressor


my_regressor = KerasRegressor(build_fn=create_model, verbose=0)


param_grid = {'optimizer': ['rmsprop', 'adam', 'Adamax'],'activation': ['relu', 'elu', 'selu','softplus','tanh', 'sigmoid', 'linear'], 'hidden':range(2,8)}
grid = GridSearchCV(estimator=my_regressor, param_grid=param_grid, scoring='neg_mean_squared_error',verbose=3, cv=3)
grid_result = grid.fit(X_train, y_train, batch_size = 100, epochs = 20000)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
results=grid_result.cv_results_['mean_test_score']

from statistics import mean

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']



file = open('opti.py', 'w')



for mean, stdev, param in zip(means, stds, params):
    file.write("%f (%f) with: %r \n" % (mean, stdev, param))
    
file.close()












y_predB = regressor.predict(testBx)
y_predB=[item[0] for item in y_predB]
zeros=np.zeros(len(y_predB))>1
resultB = np.vstack((testBy,y_predB, zeros,zeros,zeros))
resultB=resultB.T



resultBtoDrop=[]
resultBdropped=[]
resultBwrong=[]


for i in range(35,91,5):
    for j in range(5,31,5):

        
        testB_boundaryValule=np.percentile(testBy, j)
        y_predB_boundaryValule=np.percentile(y_predB, i)
        sumB=0
        sumB1=0
        sumB2=0
        for item in resultB:
            item[2]=item[0]<testB_boundaryValule
            item[3]=item[1]<y_predB_boundaryValule
            item[4]=item[3]>=item[2]
            sumB+=item[4]==0
            sumB1+=item[3]==0
            sumB2+=item[2]==0
        resultBwrong.append(sumB)
        resultBdropped.append(sumB1)
        resultBtoDrop.append(sumB2)
lenB=(len(y_predB))
    
resultBdropped=np.asarray(resultBdropped)
resultBwrong=np.asarray(resultBwrong)
resultBtoDrop=np.asarray(resultBtoDrop)



temp=[]

xxx=pd.DataFrame()
xxx["resultBdropped"]=resultBdropped
xxx["resultBwrong"]=resultBwrong
xxx["resultBtoDrop"]=resultBtoDrop

xxx["ylabel"]=xxx["resultBdropped"]
xxx["xlabel"]=xxx["resultBtoDrop"]
xxx["resultBdropped"]*=100/lenB
xxx["resultBtoDrop"]*=100/lenB


def helper(a):
    return str(round(a/lenB*100,1))+"% "+str(a)

xxx["ylabel"]=xxx.ylabel.apply(helper)
xxx["xlabel"]=xxx.xlabel.apply(helper)

xxx=xxx.round(1)

    
resultBgraph=xxx.pivot("resultBdropped" ,"resultBtoDrop","resultBwrong")
ax = plt.axes()
sns.set()
#for x1 in range(930,970,1):
fig=sns.heatmap(resultBgraph, linewidths=.5, annot=True, fmt="d", cmap='jet',
            cbar=True, robust=False, yticklabels=xxx["ylabel"].drop_duplicates().iloc[::-1]
            ,xticklabels=xxx["xlabel"].drop_duplicates().iloc[::-1])
ax.set_title("Liczba błędnie odrzuconych opływowowych konfiguracji \n dla zestawu geometrii B, liczba konfiguracji: "+str(lenB), fontsize=13)
ax.set_xlabel("Odsetek konfiguracji odrzuconych jako nieopływowe na postawie \n wyników CFD [%, ilość konfiguracji]", fontsize=12)
ax.set_ylabel("Odsetek konfiguracji odrzuconych jako nieopływowe \n na postawie przybliżenia sztucznej inteligencji  [%, ilość konfiguracji]", fontsize=12)

fig = fig.get_figure()









result2=pd.DataFrame(resultB)
result2.sort_values(by=1, inplace=True)
fig, ax = plt.subplots()
fig.set_size_inches(10,6)
plt.plot(range(1,53),result2[0])
plt.scatter(a[3], a["1_x"],4,c="blue",marker="x")
plt.title('Współczynnik oporu dla geometrii A', fontsize=15)
plt.xlabel('Parametr A1 [mm]')
plt.ylabel('Współczynnik oporu [1]')
plt.xticks(a[3], range(25,51), rotation=0)
#plt.yticks(a["1_x"], np.arange(8.6,9.6,0.10), rotation=20)
plt.grid(color='gray', linestyle='-', linewidth=0.7)
ax.set_facecolor('xkcd:white')





y_predC = regressor.predict(testCx)
y_predC=[item[0] for item in y_predC]
zeros=np.zeros(len(y_predC))>1
resultC = np.vstack((testCy,y_predC, zeros,zeros,zeros))
resultC=resultC.T

testC_boundaryValule=np.percentile(testCy, 20)
y_predC_boundaryValule=np.percentile(y_predC, 10)
sumC=0
sumC1=0
for item in resultC:
    item[2]=item[0]<testC_boundaryValule
    item[3]=item[1]<y_predC_boundaryValule
    item[4]=item[3]>=item[2]
    sumC+=item[4]==0
    sumC1+=item[3]==0
print (sumC, sumC1, len(y_predC))






resultCtoDrop=[]
resultCdropped=[]
resultCwrong=[]


for i in range(34,76,5):
    for j in range(5,33,5):

        
        testC_boundaryValule=np.percentile(testCy, j)
        y_predC_boundaryValule=np.percentile(y_predC, i)
        sumC=0
        sumC1=0
        sumC2=0
        for item in resultC:
            item[2]=item[0]<testC_boundaryValule
            item[3]=item[1]<y_predC_boundaryValule
            item[4]=item[3]>=item[2]
            sumC+=item[4]==0
            sumC1+=item[3]==0
            sumC2+=item[2]==0
        resultCwrong.append(sumC)
        resultCdropped.append(sumC1)
        resultCtoDrop.append(sumC2)
lenC=(len(y_predC))
    
resultCdropped=np.asarray(resultCdropped)
resultCwrong=np.asarray(resultCwrong)
resultCtoDrop=np.asarray(resultCtoDrop)



temp=[]

xxx=pd.DataFrame()
xxx["resultCdropped"]=resultCdropped
xxx["resultCwrong"]=resultCwrong
xxx["resultCtoDrop"]=resultCtoDrop

xxx["ylabel"]=xxx["resultCdropped"]
xxx["xlabel"]=xxx["resultCtoDrop"]
xxx["resultCdropped"]*=100/lenC
xxx["resultCtoDrop"]*=100/lenC


def helper(a):
    return str(round(a/lenC*100,1))+"% "+str(a)

xxx["ylabel"]=xxx.ylabel.apply(helper)
xxx["xlabel"]=xxx.xlabel.apply(helper)

xxx=xxx.round(1)


    
resultCgraph=xxx.pivot("resultCdropped" ,"resultCtoDrop","resultCwrong")

ax = plt.axes()
sns.set()
#for x1 in range(930,970,1):
fig=sns.heatmap(resultCgraph, linewidths=.5, annot=True, fmt="d", cmap='jet',
            cbar=True, robust=False, yticklabels=xxx["ylabel"].drop_duplicates().iloc[::-1]
            ,xticklabels=xxx["xlabel"].drop_duplicates().iloc[::-1])
ax.set_title("Liczba błędnie odrzuconych opływowowych konfiguracji \n dla zestawu geometrii C, liczba konfiguracji: "+str(lenC), fontsize=13)
ax.set_xlabel("Odsetek konfiguracji odrzuconych jako nieopływowe na postawie \n wyników CFD [%, ilość konfiguracji]", fontsize=12)
ax.set_ylabel("Odsetek konfiguracji odrzuconych jako nieopływowe \n na postawie przybliżenia sztucznej inteligencji  [%, ilość konfiguracji]", fontsize=12)

fig = fig.get_figure()






result3=pd.DataFrame(resultC)
result3.sort_values(by=1, inplace=True)
fig, ax = plt.subplots()
fig.set_size_inches(10,6)
plt.plot(range(1,56),result3[0])
plt.scatter(a[3], a["1_x"],4,c="blue",marker="x")
plt.title('Współczynnik oporu dla geometrii A', fontsize=15)
plt.xlabel('Parametr A1 [mm]')
plt.ylabel('Współczynnik oporu [1]')
plt.xticks(a[3], range(25,51), rotation=0)
#plt.yticks(a["1_x"], np.arange(8.6,9.6,0.10), rotation=20)
plt.grid(color='gray', linestyle='-', linewidth=0.7)
ax.set_facecolor('xkcd:white')
