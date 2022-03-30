#以分布生成随机数
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import integrate
from scipy.spatial import Delaunay
import os
from mpl_toolkits.mplot3d import Axes3D
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
###########################################
start = time.time()
'''
#13
def f(x,y):
    return x**5+2*x**4-x**2-4*np.sin(3*y**2-y)
#14 
def f(x,y):
    return 10*(4-y**2+np.cos(2*x))
#15
def f(x,y):
    return 10*(3*np.cos(1.6*x)+0.3*np.sin(4*y)+y)
#16
def f(x,y):
    return 10*(0.1*np.sin(7*x)+0.2*np.cos(6*y)+10-x**2-y**3+x)
#17
def f(x,y):
    return 10*(np.arctan(x+2*y)+0.1*np.sin(5*x)-np.cos(y))
#27
def f(x,y):
    return 10*(0.1*x**4-0.4*x**2+0.5*np.sin(4*y)+0.7*y)
#28
def f(x,y):
    return 10*(0.1*np.exp(x-0.5*y)+np.cos(3*y)-np.sin(2*x))
#29
def f(x,y):
    return 10*(np.arctan(x*y)+x-y**2+0.6*np.cos(3*y)+np.sin(1.5*x))
'''
#30
def f(x,y):
    return 10*(0.5*x+np.arctan(-y)+np.sin(2*x+y))
def cel_area(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1,p2,p3
    return 0.5*abs((x2*(y3-y1) + x1*(y2-y3) + x3*(y1-y2)))
def hr(x):
    return 1-x
##########################################
alpha_i = 50   #0.1,1,10,50,100
a = 5
n1 = 16
n2 = 81
n = n1+n2  #总点数
#################二维区域边界###############
p1=[-2,-1,0,1,2,2,2,2,2,1,0,-1,-2,-2,-2,-2,-2]
p2=[2,2,2,2,2,1,0,-1,-2,-2,-2,-2,-2,-1,0,1,2]
p1= np.array(p1)
p2= np.array(p2)
'''
fig=plt.figure(figsize=(10, 10))
plt.rcParams['font.sans-serif']=['SimHei']
plt.plot(p2, p1, label='原曲线',color='r')
plt.show()
plt.legend()
'''
##############生成边界线上均匀点#############
X1=[-2,-1,0,1,2,2,2,2,2,1,0,-1,-2,-2,-2,-2]
Y1=[2,2,2,2,2,1,0,-1,-2,-2,-2,-2,-2,-1,0,1]
X1= np.array(X1)
Y1= np.array(Y1)
'''
plt.scatter(X1, Y1,label='线上均匀点',color='b')
plt.show()
plt.legend()
'''
###############区域内初始点#################
t3=[-1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,
    -1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,
    -1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,
    -1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,
    -1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,
    -1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,
    -1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,
    -1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,
    -1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6]
t4=[-1.6,-1.6,-1.6,-1.6,-1.6,-1.6,-1.6,-1.6,-1.6,
    -1.2,-1.2,-1.2,-1.2,-1.2,-1.2,-1.2,-1.2,-1.2,
    -0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,
    -0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,
    0,0,0,0,0,0,0,0,0,
    0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,
    0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,
    1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,1.2,
    1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6]
X2=np.array(t3)
Y2=np.array(t4)
'''
plt.plot(X2,Y2,'*',color="blue")
plt.title("original_points_in_circle")
plt.legend()
plt.show()
'''
###############初始点的目标函数值##############
C=np.append(X1,X2) 
D=np.append(Y1,Y2) 
B=[[ 0 for i in range(2) ] for j in range(n)]
for i in range(n):
    B[i][0]=C[i]
    B[i][1]=D[i]
point_0=np.array(B)
tri_0= Delaunay(point_0)
Gap_0=0
for vertice in tri_0.vertices:
    h_0=0
    for i in range(3):
        M=point_0[vertice[i]]
        h_0+=1/3*(f(M[0],M[1]))
    cel_S=cel_area(point_0[vertice[0]],point_0[vertice[1]],point_0[vertice[2]])
    cel_v=cel_S*h_0
    def g(x,y):
        m=(point_0[vertice[0]][0]-point_0[vertice[2]][0])*x+(point_0[vertice[1]][0]
            -point_0[vertice[2]][0])*y+point_0[vertice[2]][0]
        n=(point_0[vertice[0]][1]-point_0[vertice[2]][1])*x+(point_0[vertice[1]][1]
            -point_0[vertice[2]][1])*y+point_0[vertice[2]][1]
        return f(m,n)
    v_0,err_0=integrate.dblquad(g,0,1,lambda x:0,hr)
    V_0=2*cel_S*v_0
    Gap_0+=(V_0-cel_v)**2
print('初始点的lossfunction值为：',Gap_0)
point_1=[[ 0 for i in range(2) ] for j in range(n1)]
for i in range(n1):
    point_1[i][0]=X1[i]
    point_1[i][1]=Y1[i]
#############初始点的三维分布图################
'''
fig=plt.figure(figsize=(10, 10))
ax1=plt.axes(projection='3d')
Zzz=f(point_0[:,0],point_0[:,1])
ax1.scatter3D(point_0[:,0],point_0[:,1],Zzz,label='初始点', cmap='Blues')
ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')
ax1.set_zlabel('Z Label')
plt.title('3D Random Scatter')
plt.legend()
plt.show()
plt.rcParams['font.sans-serif']=['FangSong']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  #用来正常显示负号
'''
#定义三维数据
xxx = np.arange(-2,2,0.1)
yyy = np.arange(-2,2,0.1)
Xx, Yy = np.meshgrid(xxx, yyy)
Zzzz = f(Xx,Yy)
#作图
'''
ax1.plot_surface(Xx,Yy,Zzzz,cmap='rainbow') 
#改变cmap参数可以控制三维曲面的颜色组合, 一般我们见到的三维曲面就是 rainbow 的
plt.show()
'''
##############初始点三角剖分图################
'''
fig=plt.figure(figsize=(8, 8))
plt.triplot(point_0[:,0], point_0[:,1], tri_0.simplices.copy())
plt.plot(point_0[:,0], point_0[:,1], 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Delaunay triangulation_1')
plt.show()
'''
#################等高线图1####################
#建立步长为0.01，即每隔0.01取一个点
step = 0.01
x_p = np.arange(-2,2,step)
y_p = np.arange(-2,2,step)
#也可以用x = np.linspace(-10,10,100)表示从-10到10，分100份
#将原始数据变成网格数据形式
X_p,Y_p = np.meshgrid(x_p,y_p)
#写入函数，z是大写
Z_p = f(X_p,Y_p)
'''
#设置打开画布大小,长10，宽10
plt.figure(figsize=(10,10))
#填充颜色，f即filled
plt.plot(point_0[:,0],point_0[:,1],'*',color="blue")
plt.triplot(point_0[:,0], point_0[:,1], tri_0.simplices.copy())
plt.contourf(X_p,Y_p,Z_p)
#画等高线
plt.contour(X_p,Y_p,Z_p)
plt.xlim((-2, 2))
plt.ylim((-2, 2))
plt.title('等高线图1')
plt.xlabel("x")
plt.ylabel("y")
plt.show()
'''
###############优化内部取点##################
x0=np.append(X2,Y2)  
point_m=[[ 0 for i in range(2) ] for j in range(n)]
for i in range(n1):
    point_m[i][0]=X1[i]
    point_m[i][1]=Y1[i]
def objective_function(x):
    x=x.reshape((len(x),1))
    for i in range(n2):
        point_m[i+n1][0]=x[i]
        point_m[i+n1][1]=x[i+n2]
    points_m=np.array(point_m,dtype=object)
    Gap=0
    nn=0
    for vertice in tri_0.vertices:
        h=0
        for i in range(3):
            A=points_m[vertice[i]]
            h+=1/3*(f(A[0],A[1]))
        cel_S=cel_area(points_m[vertice[0]],points_m[vertice[1]],points_m[vertice[2]])
        cel_v=cel_S*h
        def g(x,y):
            m=(points_m[vertice[0]][0]-points_m[vertice[2]][0])*x+(points_m[vertice[1]][0]
             -points_m[vertice[2]][0])*y+points_m[vertice[2]][0]
            n=(points_m[vertice[0]][1]-points_m[vertice[2]][1])*x+(points_m[vertice[1]][1]
             -points_m[vertice[2]][1])*y+points_m[vertice[2]][1]
            return f(m,n)
        v,err=integrate.dblquad(g,0,1,lambda x:0,hr)
        V=2*cel_S*v
        Gap+=(V-cel_v)**2
        norm1=((point_0[vertice[1]][0]-point_0[vertice[0]][0])*(point_0[vertice[2]][1]-point_0[vertice[0]][1])
               -(point_0[vertice[2]][0]-point_0[vertice[0]][0])*(point_0[vertice[1]][1]-point_0[vertice[0]][1]))
        norm2=((points_m[vertice[1]][0]-points_m[vertice[0]][0])*(points_m[vertice[2]][1]-points_m[vertice[0]][1])
        -(points_m[vertice[2]][0]-points_m[vertice[0]][0])*(points_m[vertice[1]][1]-points_m[vertice[0]][1]))
        if norm2 == 0 and norm1>0:
            norm2 = -1
        elif norm2 == 0 and norm1<0:
            norm2 = 1
        else:
            norm2 = norm2
        n=alpha_i*(norm1/abs(norm1)-norm2/abs(norm2))**2
        nn+=n
    ss=Gap+nn
    return ss   
bds_=[(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
      (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
      (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
      (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
      (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
      (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
      (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
      (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
      (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
      (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
      (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
      (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
      (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
      (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
      (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
      (-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),(-2,2),
      (-2,2),(-2,2)]
res = minimize(x0=x0.reshape(162,),fun=objective_function,method='SLSQP',bounds=bds_,options={'maxiter':800})
print('优化后的lossfunction值为：',res.fun)
print('优化算法最优解：',res.x)
print('迭代终止是否成功：', res.success)
print('迭代终止原因：', res.message)
#############优化解三角剖分图###############
answer=res.x
C=answer.reshape((2,n2))
point_2=[[C[j][i] for j in range(len(C))] for i in range(len(C[0]))]
point=point_1+point_2
points=np.array(point)
#tri=Delaunay(points)
'''
fig=plt.figure(figsize=(8, 8))
plt.triplot(points[:,0], points[:,1], tri_0.simplices.copy())
plt.plot(points[:,0], points[:,1], 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Delaunay triangulation_2')
plt.show()
'''
################优化点三维图###############
fig=plt.figure(figsize=(10, 10))
ax2=plt.axes(projection='3d')
Z=f(points[:,0],points[:,1])
ax2.scatter3D(points[:,0],points[:,1],Z,label='优化点', cmap='Blues')
ax2.set_xlabel('X Label')
ax2.set_ylabel('Y Label')
ax2.set_zlabel('Z Label')
plt.title('3D Random Scatter')
plt.legend()
plt.show()
plt.rcParams['font.sans-serif']=['FangSong'] 
plt.rcParams['axes.unicode_minus']=False 
#定义三维数据
xx = np.arange(-2,2,0.1)
yy = np.arange(-2,2,0.1)
X, Y = np.meshgrid(xx, yy)
Zz = f(X,Y)
ax2.plot_surface(X,Y,Zz,cmap='rainbow') 
plt.show()
#################等高线图2########################
#建立步长为0.01，即每隔0.01取一个点
step = 0.01
x_p = np.arange(-2,2,step)
y_p = np.arange(-2,2,step)
#也可以用x = np.linspace(-10,10,100)表示从-10到10，分100份
#将原始数据变成网格数据形式
X_p,Y_p = np.meshgrid(x_p,y_p)
#写入函数，z是大写
Z_p = f(X_p,Y_p)
#设置打开画布大小,长10，宽10
plt.figure(figsize=(10,10))
#填充颜色，f即filled
plt.plot(points[:,0],points[:,1],'*',color="blue")
plt.triplot(points[:,0], points[:,1], tri_0.simplices.copy())
plt.contourf(X_p,Y_p,Z_p)
#画等高线
plt.contour(X_p,Y_p,Z_p)
plt.xlim((-2, 2))
plt.ylim((-2, 2))
plt.title('等高线图2')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

end = time.time()
print(end-start)