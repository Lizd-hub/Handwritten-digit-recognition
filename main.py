from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
digits=load_digits()
images_and_labels=list(zip(digits.images,digits.target))
plt.figure(figsize=(8,6),dpi=200)
for index,(image,label)in enumerate(images_and_labels[:8]):
    plt.subplot(2,4,index+1)
    plt.axis('off') # 关闭坐标轴
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Digit:%i'%label,fontsize=20)

print("shape of raw image_data:{0}".format(digits.images.shape))
print("shape if data:{0}".format(digits.data.shape))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.20,random_state=2)

from sklearn.svm import SVC
#将逻辑回归改为SVC分类
clf=SVC(kernel='rbf',gamma='scale',random_state=42)
clf.fit(X_train,y_train)

accuracy=clf.score(X_test,y_test)
print('Accuracy score of the {} is {:.2f}'.format(clf.__class__.__name__,accuracy))

y_pred=clf.predict(X_test)
fig,axes=plt.subplots(4,4,figsize=(8,8))
fig.subplots_adjust(hspace=0.1,wspace=0.1)
for i,ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8,8),cmap=plt.cm.gray_r,interpolation='nearest')
    ax.text(0.05,0.05,str(y_pred[i]),fontsize=32,transform=ax.transAxes,
        color='green' if y_pred[i] == y_test[i] else 'red')
    ax.text(0.8,0.05,str(y_test[i]),fontsize=32,transform=ax.transAxes,color='black')
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
