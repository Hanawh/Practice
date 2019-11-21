import cv2
import glob
import numpy as np
from sklearn.model_selection import RepeatedKFold, GridSearchCV,train_test_split
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import pickle
import ipdb
from sklearn.metrics import classification_report

################### 读取数据集 ###################
def load_image(imgdir='yale/'):
    imgs = glob.glob(imgdir+'/s*.bmp')
    imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('s')[-1])) #因为读取后是无序的需要排序
    label = [(int(name.split('.')[-2].split('s')[-1])-1)//11 for name in imgs]
    image = [np.reshape(cv2.imread(x,0),(1,10000)) for x in imgs]
    image = np.concatenate(image, axis=0)
    label = np.reshape(label,(165,))
    return image,label 

imgs,label = load_image()

################## 划分数据集 ###################
test_ratio = 0.09
test_number = test_ratio * len(imgs)
x_num = test_number/5 #画图时行数
x_train, x_test, y_train, y_test = train_test_split(imgs, label, test_size=test_ratio, random_state=42) 

################# 模型定义 ####################
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)  #数据标准化

pca = PCA(n_components=90)
svc = svm.SVC(class_weight='balanced')
model = make_pipeline(pca, svc)

kf= RepeatedKFold(n_splits=10,n_repeats=2)
c_range = np.logspace(-5, 5, 11, base=10)
gamma_range = np.logspace(-5, 5, 11, base=10)
param_grid = [{'svc__kernel': ['rbf'], 'svc__C': c_range, 'svc__gamma': gamma_range}]
grid = GridSearchCV(model, param_grid, cv=kf)
grid.fit(x_train, y_train)

print('网格搜索-最佳度量值:',grid.best_score_)  # 获取最佳度量值
print('网格搜索-最佳参数：',grid.best_params_)  # 获取最佳度量值时的代定参数的值，存储为字典
print('网格搜索-最佳模型：',grid.best_estimator_)  # 获取最佳度量时的分类器模型

################# 模型存储 #####################
fw = open('para.pkl','wb')  
pickle.dump(grid.best_estimator_, fw)  
fw.close()  

################ 模型读取 #####################
fr = open('para.pkl','rb')  
model_test = pickle.load(fr)

############### 预测 #########################
x_test_cp =  scaler.fit_transform(x_test) 
y_fit = model_test.predict(x_test_cp)

############## 画图 ##########################
fig, ax = plt.subplots(x_num, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(x_test[i].reshape(100, 100), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(str(y_fit[i]),size=7,color='black' if y_fit[i] == y_test[i] else 'red')
fig.suptitle('Incorect Lables in Red', size=14)
plt.show()
plt.savefig("yale_{}.png".format(test_number))
print(classification_report(y_test,y_fit))

################### 画出特征脸 ######################
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()
    plt.savefig("feature_{}.png".format(50))

n_components = 50
pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(x_train)
eigenfaces = pca.components_.reshape((n_components, 100, 100))
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, 100, 100)





 
    




