import cv2
import os
import glob
data_dir = './data'

from_data_dir = './down_sampled_AR'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

for i in range(1,101):
	full_data_dir = os.path.join(data_dir,str(i))
	if not os.path.exists(full_data_dir):
	    os.makedirs(full_data_dir)


for image in glob.glob(from_data_dir+"/*.bmp"):
    temp = image.split('-')
    temp2 = image.split('/')
    img = cv2.imread(image)
    img = cv2.resize(img, (160,160))
    path = os.path.join(data_dir,str(int(temp[1]))) if temp[0][-1] == 'M' else  os.path.join(data_dir,str(50+int(temp[1])))
    path = os.path.join(path,temp2[2][:-4])
    print(path)
    cv2.imwrite(path+'.png',img)
    # print(temp)
    # print(int(temp[1]))




