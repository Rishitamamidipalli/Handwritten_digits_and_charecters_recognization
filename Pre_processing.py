import cv2
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.datasets import mnist
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
data="A_Z Handwritten Data.csv"
dataset=pd.read_csv(data).astype("float32")
dataset.rename(columns={'0':"output"},inplace=True)
char_x=dataset.drop("output",axis=1)
char_y=dataset["output"]

(dig_train_x,dig_train_y),(dig_test_x,dig_test_y)=mnist.load_data()
char_x=char_x.values
print(char_x.shape,char_y.shape,'ppp1')
print(dig_train_x.shape,dig_train_y.shape,dig_test_x.shape,dig_test_y.shape,'ppp2')
digit_ip=np.concatenate((dig_train_x,dig_test_x))
digit_op=np.concatenate((dig_train_y,dig_test_y))
digit_op+=26
l=[]

for i in char_x:
    img=np.reshape(i,(28,28,1))
    l.append(img)

char_ip=np.array(l,dtype=np.float32)
char_op=char_y
digit_ip=np.reshape(digit_ip,(digit_ip.shape[0],digit_ip.shape[1],digit_ip.shape[2],1))

print(digit_ip.shape,digit_op.shape,'ppp3')
print(char_ip.shape,char_op.shape,'ppp4')


ip=np.concatenate((char_ip,digit_ip))
op=np.concatenate((char_op,digit_op))
print(ip.shape,op.shape,'ppp5')
train_x,test_x,train_y,test_y=train_test_split(ip,op,test_size=0.2)
train_x=train_x/255.0
test_x=test_x/255.0
train_x=np.reshape(train_x,(train_x.shape[0],train_x.shape[1],train_x.shape[2],1))
test_x=np.reshape(test_x,(test_x.shape[0],test_x.shape[1],test_x.shape[2],1))

print(test_x.shape,test_y.shape,'ppp7')
np.save("train_x",train_x)
np.save("train_y",train_y)
np.save("test_x",test_x)
np.save("test_y",test_y)



