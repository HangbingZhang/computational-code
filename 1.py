import numpy as np
import sympy
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import pandas as pd
pi=3.1415926
##########################
n1=16
n2=81
n=n1+n2
def f(x,y):
    return np.sin(3*x+1)-np.cos(2*y-1.5)+5
#边界点
X1=[-2,-1,0,1,2,2,2,2,2,1,0,-1,-2,-2,-2,-2]
Y1=[2,2,2,2,2,1,0,-1,-2,-2,-2,-2,-2,-1,0,1]
X1= np.array(X1)
Y1= np.array(Y1)
x=[-1.81910356e+00,	-1.07093340e+00,	-7.67801982e-01,	-7.73401981e-01,
1.02546382e-01,	3.36022776e-01,	7.92470951e-01,	1.13474756e+00,
1.38073941e+00,	-1.76994213e+00,	-8.72804577e-01,	-7.31717266e-01,
-3.80079782e-01,	8.11878714e-02,	4.02639052e-01,	8.33174277e-01,
1.19188008e+00,	1.42466143e+00,	-1.68273866e+00,	-1.15002858e+00,
-7.76913040e-01,	-3.29777285e-01,	-5.54816820e-04,	3.64046741e-01,
8.15232121e-01,	1.14253294e+00,	1.50070234e+00,	-1.88893531e+00,
-1.05104327e+00,	-7.69495243e-01,	-4.16760425e-01,	4.16171474e-02,
4.01900409e-01,	8.94000117e-01,	1.31972061e+00,	1.97952227e+00,
-1.78084243e+00,	-1.03783181e+00,	-7.87080866e-01,	-4.36222962e-01,
-2.58611121e-03,	3.33955957e-01,	1.10435644e+00,	1.18086938e+00,
1.46159347e+00,	-1.41836295e+00,	-1.06179993e+00,	-8.18957311e-01,
-6.33672835e-01,	2.90823388e-02,	3.27654362e-01,	1.11221649e+00,
1.41607233e+00,	1.99047970e+00,	-1.73269430e+00,	-1.06123748e+00,
-7.79578721e-01,	-6.10958101e-01,	6.71634451e-02,	3.22739573e-01,
9.70961554e-01,	1.22572007e+00,	1.49905765e+00,	-1.73546377e+00,
-1.06233082e+00,	-8.50275313e-01,	-6.90044594e-01,	1.82217985e-02,
3.57879151e-01,	1.06279390e+00,	1.21495306e+00,	1.48512151e+00,
-1.72697108e+00,	-9.99312783e-01,	-8.31302218e-01,	-7.47927986e-01,
8.68789631e-02,	3.79837212e-01,	9.10516251e-01,	1.18501695e+00,
1.46082229e+00]	
y=[-1.96843231e+00,	-1.57780445e+00,	-1.75018012e+00,
-1.99897892e+00,	-1.44179019e+00,	-1.99924265e+00,	-1.53235691e+00,
-1.73398180e+00,	-1.99727668e+00,	-1.16829207e+00,	-1.17616231e+00,
-1.22115055e+00,	-1.18701832e+00,	-1.11735855e+00,	-1.27001373e+00,
-1.10602164e+00,	-1.24936682e+00,	-1.14965044e+00,	-7.39069294e-01,
-8.00577654e-01,	-7.78588666e-01,	-7.68096904e-01,	-8.53051469e-01,
-8.21767691e-01,	-8.30386741e-01,	-7.62780748e-01,	-8.08225177e-01,
-5.12420673e-01,	-4.44697339e-01,	-3.28889868e-01,	-4.38336997e-01,
-4.96959637e-01,	-4.42789268e-01,	-5.16854446e-01,	-2.99646633e-01,
-5.93885968e-01,	-2.43023478e-01,	3.44951403e-02,	1.04727412e-01,
-4.87127904e-02,	-1.03306483e-01,	-1.32597685e-01,	2.49532596e-02,
7.46492829e-02,	1.50442969e-01,	4.64536947e-01,	4.56383979e-01,
4.99747359e-01,	3.64673335e-01,	4.33511047e-01,	3.90436122e-01,
4.31998756e-01,	4.24861614e-01,	5.22303239e-01,	7.53955668e-01,
8.10397156e-01,	8.27511628e-01,	7.43303361e-01,	7.95904816e-01,
7.80396703e-01,	7.85596351e-01,	7.86961540e-01,	7.21115014e-01,
1.07711890e+00, 1.16853952e+00,	1.13309671e+00,	1.11668678e+00,
1.23732801e+00,	1.21910752e+00,	1.08279256e+00,	1.15270335e+00,
1.15182531e+00,	1.96640628e+00,	1.55149846e+00,	1.57515829e+00,
1.99566703e+00,	1.73998606e+00,	1.94255691e+00,	1.59600947e+00,
1.57758737e+00,	1.99964129e+00]	
x_=[-1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,
    -1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,
    -1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,
    -1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,
    -1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,
    -1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,
    -1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,
    -1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,
    -1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6]
y_=[-1.6,-1.6,-1.6,-1.6,-1.6,-1.6,-1.6,-1.6,-1.6,
    -1.2,-1.2,-1.2,-1.2,-1.2,-1.2,-1.2,-1.2,-1.2,
    -0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,
    -0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,
    0,0,0,0,0,0,0,0,0,
    0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,
    0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,
    1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,
    1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6]
X2=np.array(x_)
Y2=np.array(y_)
C=np.append(X1,X2) 
D=np.append(Y1,Y2) 
B=[[ 0 for i in range(2) ] for j in range(n)]
for i in range(n):
    B[i][0]=C[i]
    B[i][1]=D[i]
point_0=np.array(B)
X=np.array(x)
Y=np.array(y)
C1=np.append(X1,X) 
D1=np.append(Y1,Y) 
B1=[[ 0 for i in range(2) ] for j in range(n)]
for i in range(n):
    B1[i][0]=C1[i]
    B1[i][1]=D1[i]
point_1=np.array(B1)
###########################
#求f(x,y)
a1=[]
a2=[]
for i in range(n2):
    n=f(x[i],y[i])
    n_=f(x_[i],y_[i])
    a1.append(n)
    a2.append(n_)
###########################
#求梯度模
b1=[]
b2=[]
def part_1(index,x,y):
    e=10 ** (-4)
    n=f(x,y)
    if index == 0:
        x_1=x+e
        m=f(x_1,y)
        k=(m-n)/e
    else:
        y_1=y+e
        m=f(x,y_1)
        k=(m-n)/e
    return k
def delta_length(x,y):
    return np.sqrt((part_1(0,x,y)**2+(part_1(1,x,y)**2)))
for i in range(n2):
    nn=delta_length(x[i],y[i])
    nn_=delta_length(x_[i],y_[i])
    b1.append(nn)
    b2.append(nn_)
##########################
#求hessin的范数-2
c=[]
def part_2(index1,index2,x,y):
    e=10 ** (-4)
    n=f(x,y)
    if index1==0 and index2==0:
        m1=f(x+e,y)
        m2=f(x-e,y)
        k=(m1+m2-2*n)/e**2
    elif index1==1 and index2==1:
        m1=f(x,y+e)
        m2=f(x,y-e)
        k=(m1+m2-2*n)/e**2
    else:
        m1=f(x+e,y+e)
        m2=f(x,y+e)
        m3=f(x+e,y)
        k=(m1+n-m2-m3)/e**2
    return k
s1=[]
s2=[]
for p in range(n2):
    hess=[[0,0],[0,0]]
    hess_=[[0,0],[0,0]]
    for i in range(2):
        for j in range(2):
            hess[i][j]=part_2(i,j,x[p],y[p])
            hess_[i][j]=part_2(i,j,x_[p],y_[p])
    nnn=np.linalg.norm(hess, ord=2, axis=None, keepdims=False)
    nnn_=np.linalg.norm(hess_, ord=2, axis=None, keepdims=False)
    s1.append(nnn)
    s2.append(nnn_)
    nnn_c=nnn-nnn_
    c.append(nnn_c)
print("hessin的范数-2变化范围：{}~{}".format(min(c),max(c)))
'''
fig=plt.figure(figsize=(10, 10))
plt.plot(a2,a2,'o',color='g')
plt.title('优化后hessin范数散点分布')
plt.show()
fig=plt.figure(figsize=(10, 10))
plt.plot(a1,a1,'o',color='b')
plt.title('优化前hessin范数散点分布')
plt.show()
'''
m=np.linspace(min(s1)-1,max(s1)+1,100)  
mm=m
m1=np.linspace(min(a1)-1,max(a1)+1,100)  
mm1=m1
m2=np.linspace(min(b1)-1,max(b1)+1,100)  
mm2=m2
fig=plt.figure(figsize=(10, 10))
plt.plot(mm,m,color='r')
plt.plot(s2,s1,'o',color='b')
plt.ylabel('优化后')
plt.xlabel('优化前')
plt.title('hessin范数散点分布')
plt.show()
fig=plt.figure(figsize=(10, 10))
plt.plot(mm1,m1,color='r')
plt.plot(a2,a1,'o')
plt.ylabel('优化后')
plt.xlabel('优化前')
plt.title('函数值散点分布')
plt.show()
fig=plt.figure(figsize=(10, 10))
plt.plot(mm2,m2,color='r')
plt.plot(b2,b1,'o')
plt.ylabel('优化后')
plt.xlabel('优化前')
plt.title('梯度模散点分布')
plt.show()
#########################################################
'''
a1.sort()
a2.sort()
fig=plt.figure(figsize=(10, 10))
plt.hist(a1,color='b')
plt.title('1-优化后的函数值')
plt.show()
fig=plt.figure(figsize=(10, 10))
plt.hist(a2,color='b')
plt.title('1-优化前的函数值')
plt.show()
b1.sort()
b2.sort()
fig=plt.figure(figsize=(10, 10))
plt.hist(b1,color='b')
plt.title('1-优化后的梯度模')
plt.show()
fig=plt.figure(figsize=(10, 10))
plt.hist(b2,color='b')
plt.title('1-优化前的梯度模')
plt.show()
######################点数变化#######################
s1.sort()
s2.sort()
s1=np.array(s1/max(s1))
s2=s2/max(s2)
h1=0
h2=0
h3=0
h4=0
h5=0
h=[]
for i in range(n2):
    if  s1[i] >0.8:
        h5+=1
    elif s1[i] >0.6:
        h4+=1
    elif s1[i] >0.4:
        h3+=1
    elif s1[i] >0.2:
        h2+=1
    else:
        h1+=1
h.append(h1)
h.append(h2)
h.append(h3)
h.append(h4)
h.append(h5)
h1_=0
h2_=0
h3_=0
h4_=0
h5_=0
h_=[]
for i in range(n2):
    if s2[i] >0.8:
        h5_+=1
    elif s2[i] >0.6:
        h4_+=1
    elif s2[i] >0.4:
        h3_+=1
    elif s2[i] >0.2:
        h2_+=1
    else:
        h1_+=1
h_.append(h1_)
h_.append(h2_)
h_.append(h3_)
h_.append(h4_)
h_.append(h5_)
fig=plt.figure(figsize=(10, 10))
index=['第1','第2','第3','第4','第5']
index_1=np.arange(len(index))
index_2=index_1 + 0.3
plt.bar(index,h_,width=0.3,color='g',label='优化前')
plt.bar(index_2,h,width=0.3,color='b',label='优化后')
plt.title('优化前后不同等级Hessian范数的点数变化')
plt.ylabel('点数')
plt.xlabel('Hessin范数等级')
plt.rcParams['font.sans-serif']=['FangSong'] 
plt.rcParams['axes.unicode_minus']=False 
plt.legend() 
plt.show()
######################计算相关系数#########################
h=np.array(h)
h_=np.array(h_)
h_c=[0,0,0,0,0]
for i in range(5):
    if h_[i]==0:
        h_c[i]=1
    else:
        h_c[i]=(h[i]-h_[i])/h_[i]
hes=[0.1,0.3,0.5,0.7,0.9]
print('点数与Hessian范数之间的相关系数：{}'.format(np.corrcoef(h_c,hes)[0][1]))
#####################连接线与梯度的角度##########################
#角度的cos值
def angle_co(x1,y1,x2,y2):
    a1=abs(x1*x2+y1*y2)
    a2=(np.sqrt(x1**2+y1**2))*(np.sqrt(x2**2+y2**2))
    return a1/a2
#########################
tri= Delaunay(point_0)
d0=[]
d1=[]
for vertice in tri.vertices:
    i=vertice[0]
    j=vertice[1]
    k=vertice[2]
    #1 i j
    xx=0.5*(point_0[j][0]+point_0[i][0])
    yy=0.5*(point_0[j][1]+point_0[i][1])
    n_0=[part_1(0,xx,yy),part_1(1,xx,yy)]
    n_1=[point_0[j][0]-point_0[i][0],point_0[j][1]-point_0[i][1]]
    co_0=angle_co(n_0[0],n_0[1],n_1[0],n_1[1])
    xx_=0.5*(point_1[j][0]+point_1[i][0])
    yy_=0.5*(point_1[j][1]+point_1[i][1])
    n__0=[part_1(0,xx_,yy_),part_1(1,xx_,yy_)]
    n__1=[point_1[j][0]-point_1[i][0],point_1[j][1]-point_1[i][1]]
    co_1=angle_co(n__0[0],n__0[1],n__1[0],n__1[1])
    co_c1=co_1-co_0
    d1.append(co_1)
    d0.append(co_0)
    #2 j k
    xx=0.5*(point_0[j][0]+point_0[k][0])
    yy=0.5*(point_0[j][1]+point_0[k][1])
    n_0=[part_1(0,xx,yy),part_1(1,xx,yy)]
    n_1=[point_0[j][0]-point_0[k][0],point_0[j][1]-point_0[k][1]]
    co_0=angle_co(n_0[0],n_0[1],n_1[0],n_1[1])
    xx_=0.5*(point_1[j][0]+point_1[k][0])
    yy_=0.5*(point_1[j][1]+point_1[k][1])
    n__0=[part_1(0,xx_,yy_),part_1(1,xx_,yy_)]
    n__1=[point_1[j][0]-point_1[k][0],point_1[j][1]-point_1[k][1]]
    co_1=angle_co(n__0[0],n__0[1],n__1[0],n__1[1])
    co_c2=co_1-co_0
    d1.append(co_1)
    d0.append(co_0)
    #3 i k
    xx=0.5*(point_0[k][0]+point_0[i][0])
    yy=0.5*(point_0[k][1]+point_0[i][1])
    n_0=[part_1(0,xx,yy),part_1(1,xx,yy)]
    n_1=[point_0[k][0]-point_0[i][0],point_0[k][1]-point_0[i][1]]
    co_0=angle_co(n_0[0],n_0[1],n_1[0],n_1[1])
    xx_=0.5*(point_1[k][0]+point_1[i][0])
    yy_=0.5*(point_1[k][1]+point_1[i][1])
    n__0=[part_1(0,xx_,yy_),part_1(1,xx_,yy_)]
    n__1=[point_1[k][0]-point_1[i][0],point_1[k][1]-point_1[i][1]]
    co_1=angle_co(n__0[0],n__0[1],n__1[0],n__1[1])
    co_c3=co_1-co_0
    d1.append(co_1)
    d0.append(co_0)
d1 = np.unique(d1)
d0 = np.unique(d0)
d1.sort()
g1=0
g2=0
g3=0
g4=0
g5=0
g=[]
g1_=0
g2_=0
g3_=0
g4_=0
g5_=0
g_=[]
for i in range(len(d1)):
    if d1[i] > 0.8:
        g5+=1
    elif d1[i] > 0.6:
        g4+=1
    elif d1[i] > 0.4:
        g3+=1
    elif d1[i] > 0.2:
        g2+=1
    else:
        g1+=1
g.append(g1)
g.append(g2)
g.append(g3)
g.append(g4)
g.append(g5)
for i in range(len(d0)):
    if d0[i] > 0.8:
        g5_+=1
    elif d0[i] > 0.6:
        g4_+=1
    elif d0[i] > 0.4:
        g3_+=1
    elif d0[i] > 0.2:
        g2_+=1
    else:
        g1_+=1
g_.append(g1_)
g_.append(g2_)
g_.append(g3_)
g_.append(g4_)
g_.append(g5_)
fig=plt.figure(figsize=(10, 10))
index=['0.1','0.3','0.5','0.7','0.9']
index_1=np.arange(len(index))
index_2=index_1 + 0.3
plt.bar(index,g_,width=0.3,color='g',label='优化前')
plt.bar(index_2,g,width=0.3,color='b',label='优化后')
plt.title('优化前后的点数变化')
plt.ylabel('点数')
plt.xlabel('连接线与梯度的角度cos值等级')
plt.legend() 
plt.show()
######################计算相关系数#########################
g=np.array(g)
g_=np.array(g_)
g_c=(g-g_)
g_cos=[0.1,0.3,0.5,0.7,0.9]
print('点数与连接线与梯度的角度cos值之间的相关系数：{}'.format(np.corrcoef(g,g_cos)[0][1]))
g_aver=sum(d1)/len(d1)
c_aver=((np.arccos(g_aver))*180)/pi
print('优化后平均角度：{}'.format(c_aver))
g_aver0=sum(d0)/len(d0)
c_aver0=((np.arccos(g_aver0))*180)/pi
print('优化前平均角度：{}'.format(c_aver0))
'''