
#CODE of Explanatory Data Analysis for Understanding Cloud from Satellite Images
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as plt1
import warnings
import numpy as np
import seaborn as sns
import cv2
#Intializing the paths to import the files from
train_data = pd.read_csv('C:/Users/hp-pc/Desktop/Data Analytics Project/train_sample.csv')

path = 'C:/Users/hp-pc/Desktop/Data Analytics Project/'

test_images = sorted(glob(path + 'test_images/*.jpg'))

path1 = 'C:/Users/hp-pc/Desktop/Data Analytics Project/train_sample_images/'

train_images = sorted(glob(path + 'train_images/*.jpg'))
#print the count of train and test images
print('Count of test images data : {}'.format(len(test_images)))
print('Count of test images data : {}'.format(len(train_images)))

#Start Explanatory Data Analysis

#Plot Test_images and Train_images on a bar plot
names = ['Test_data', 'Train_data']
values = [len(test_images), len(train_data)]

plt.figure(figsize=(8, 6))

barlist=plt.bar(names, values, alpha=1)

barlist[0].set_color('#abd001')
barlist[1].set_color('#fff001')
plt.title('Distribution of Train Data and Test Data')

#Plot a pie chart of distribution of test_images and train_images

labels = ['Test_img', 'Train_img']
explode = (0, 0.1)
sizes = [len(test_images), len(train_images)]

fi, ax = plt.subplots()

ax.pie(sizes, explode=explode, labels=labels, startangle=90, autopct='%.2f%%', colors=('blue', 'yellow'))
ax.axis('equal')

plt.title('Percentage value of Test Image and Train Image')

plt.show()

#Plot count of different clouds in training data
split_df = train_data["Image_Label"].str.split('_', n=1, expand=True)

train_data['Images'] = split_df[0]
train_data['Label'] = split_df[1]

Fish_Count = (train_data[train_data['Label'] == 'Fish'].EncodedPixels.count())
Gravel_Count = (train_data[train_data['Label'] == 'Gravel'].EncodedPixels.count())
Flower_Count = (train_data[train_data['Label'] == 'Flower'].EncodedPixels.count())
Sugar_Count = (train_data[train_data['Label'] == 'Sugar'].EncodedPixels.count())

names = ['Fish_Count', 'Gravel_Count', 'Flower_Count', 'Sugar_Count']
values = [Fish_Count, Gravel_Count, Flower_Count, Sugar_Count]

plt.figure(figsize=(8, 8))
barlist1=plt.bar(names, values, width=0.4, alpha=1)
barlist1[0].set_color('#6d2fae')
barlist1[1].set_color('#7c3fbe')
barlist1[2].set_color('#8b4fce')
barlist1[3].set_color('#9a5fde')
plt.xlabel('Cloud Names')
plt.ylabel('Number of clouds')
plt.title('Cloud Classification')

plt.show()

#Plot the count of masks in the clouds in train images

fig1, ax1=plt.subplots(figsize=(8, 8))
count_image_label = train_data.groupby('Images')['EncodedPixels'].count()
ax1.hist(count_image_label)
ax1.set_title("Plot for average number of clouds in the train data")

plt.show()

#Create Correlation matrix of the data
corr_df = pd.get_dummies(train_data, columns=['Label'])
corr_df = corr_df.fillna('-1')


def get_dummy_value(row, cloud_type):
    if cloud_type == 'fish':
        return row['Label_Fish'] * (row['EncodedPixels'] != '-1')
    if cloud_type == 'flower':
        return row['Label_Flower'] * (row['EncodedPixels'] != '-1')
    if cloud_type == 'gravel':
        return row['Label_Gravel'] * (row['EncodedPixels'] != '-1')
    if cloud_type == 'sugar':
        return row['Label_Sugar'] * (row['EncodedPixels'] != '-1')


corr_df['Label_Fish'] = corr_df.apply(lambda row: get_dummy_value(row, 'fish'), axis=1)
corr_df['Label_Flower'] = corr_df.apply(lambda row: get_dummy_value(row, 'flower'), axis=1)
corr_df['Label_Gravel'] = corr_df.apply(lambda row: get_dummy_value(row, 'gravel'), axis=1)
corr_df['Label_Sugar'] = corr_df.apply(lambda row: get_dummy_value(row, 'sugar'), axis=1)

print(corr_df['Images'].head())

corr_df = corr_df.groupby('Images')['Label_Fish', 'Label_Flower', 'Label_Gravel', 'Label_Sugar'].max()
corr_df.head()

corrs = np.corrcoef(corr_df.values.T)
sns.set(rc={'figure.figsize':(12,12)})
hm = sns.heatmap(corrs, cbar=True, annot=True, square=True, fmt='.2f',
                 yticklabels=['Fish', 'Flower', 'Gravel', 'Sugar'],
                 xticklabels=['Fish', 'Flower', 'Gravel', 'Sugar']).set_title('Cloud type correlation heatmap')
plt.show()

#Define a function to create mask(Was produced for Project Progress Report main code does use this)

def rle_mask(rle,h,w):
  rows,cols = h,w
  rle_value = [int(rle) for rle in rle.split(' ')]
  rlePairs = np.array(rle_value).reshape(-1,2)
  img1 = np.zeros(rows*cols,dtype=np.uint8)
  for index,length in rlePairs:
    index -= 1
    img1[index:index+length] = 255
  img1 = img1.reshape(rows,cols)
  img1 = img1.T
  return img1

plt.figure(figsize=(15,20))

for i, data in train_data[:].iterrows():
    img1 = cv2.imread(path1 + data['Images'])
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    encoded_px = data['EncodedPixels']
    try:
        img_mask = rle_mask(encoded_px,2100,1400)
    except:
        img_mask = np.zeros((1400, 2100))

    plt.subplot(5, 4, i + 1)
    plt.imshow(img1)
    plt.imshow(img_mask, alpha=0.8, cmap='gray')
    cv2.imwrite("img1.jpg",img1)
    plt.title("%s" % data['Label'], fontsize=8)
    plt.axis('off')
plt.show()
