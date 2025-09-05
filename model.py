#Import module that we need
import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump

# 1. Split dataset's label and data
ds_label = []
ds_data = []

# 2.Finding datasets location
for i, address in enumerate(glob.glob(r'fire_dataset\*\*')):
    # 3. Extracting dataset's data
    img = cv2.imread(address)
    # 4. Normalization data
    img = cv2.resize(img, (32, 32))
    img = img / 255.0 #scaling data
    img = img.flatten()
    ds_data.append(img)
    # 5. Extracting dataset's label
    # Note: Fire_project\fire_dataset\fire_images\fire.1.png
    lbl = address.split('\\')[-1].split('.')[0]
    ds_label.append(lbl)
    # *Extra
    if i % 100 == 0:
        print(f'{i}/1000 image processed!')
        
# 6. Convert "ds_data" to a numpy array
ds_data = np.array(ds_data)
print(ds_data.shape)
# 7. Separating train and test value
x_train, x_test, y_train, y_test = train_test_split(ds_data, ds_label, test_size= 0.1)
print(x_train.shape)
# 8. Determining the amount of nearest neighbor(K)
clf = KNeighborsClassifier()
# 9. Fitting train's data into "clf" to calculate Euclidean distance
clf.fit(x_train, y_train)
# 10. Saving model into a file with ".z" format
dump(clf, 'fire_model.z')
# 11. Calculating model accuracy 
acc = clf.score(x_test, y_test)
print(f'Model accuracy: {acc*100}%')
# 12. Surely close all opened windows in open cv
cv2.destroyAllWindows()