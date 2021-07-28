from PIL import Image
from flask_ngrok import run_with_ngrok
from flask import Flask,render_template,request
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gc
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import math
from sklearn.neighbors import NearestNeighbors
from keras.models import load_model
from sklearn.decomposition import PCA

img_size =224
model = ResNet50(weights='imagenet', include_top=False,input_shape=(img_size, img_size, 3),pooling='max')
batch_size = 64
root_dir ='/content/drive/MyDrive/img/static/Train'

img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

datagen = img_gen.flow_from_directory(root_dir,
                                        target_size=(img_size, img_size),
                                        batch_size=batch_size,
                                        class_mode=None,
                                        shuffle=False)

num_images = len(datagen.filenames)
num_epochs = int(math.ceil(num_images / batch_size))
feature_list = model.predict(datagen,num_epochs)
print(feature_list)
pca = PCA(n_components=100)
pca.fit(feature_list)
compressed_features = pca.transform(feature_list)
neighbors_pca_features = NearestNeighbors(n_neighbors=200,metric='cosine')
neighbors_pca_features.fit(compressed_features)
filenames = [root_dir + '/' + s for s in datagen.filenames]
def fun(img_path):
  input_shape = (img_size, img_size, 3)
  img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
  img_array = image.img_to_array(img)
  expanded_img_array = np.expand_dims(img_array, axis=0)
  preprocessed_img = preprocess_input(expanded_img_array)
  test_img_features = model.predict(preprocessed_img, batch_size=1)
  test_img_compressed = pca.transform(test_img_features)
  distances, indices = neighbors_pca_features.kneighbors(test_img_compressed)
  list1=[]
  for i in range(len(distances[0])):
    if distances[0][i]<=0.38:
      list1.append(indices[0][i])
  #run random fun for 5 
  list2=[]
  if(len(list1)!=0):
    '''rand_idx = random.randint(0,int(len(list1)/4))
    random_num = list1[rand_idx]
    list2.append(random_num) 
    rand_idx = random.randint(int(len(list1)/4),int(len(list1)/2))
    random_num = list1[rand_idx]
    list2.append(random_num) 
    rand_idx = random.randint(int(len(list1)/4),int(len(list1)/2))
    random_num = list1[rand_idx]
    list2.append(random_num) 
    rand_idx = random.randint(int(len(list1)/2),len(list1)-1)
    random_num = list1[rand_idx]
    list2.append(random_num) '''
    for i in range(0,4):
      rand_idx = random.randint(0,len(list1)-1)
      rand_idx = random.randint(0,len(list1)-1)
      rand_idx = random.randint(0,len(list1)-1)
      random_num = list1[rand_idx]
      list2.append(random_num)
  return list2


app=Flask(__name__)
run_with_ngrok(app)
@app.route("/")
def index():
  return render_template('upload_image.html')
@app.route("/", methods=["POST"])
def upload_image():
  output=''
  val=0
  if request.method == "POST":
    file = request.files['img']
    if len(file.filename)==0:
      return render_template('upload_image.html',text1="NO IMG SELECTED")
    else:
      img = Image.open(file.stream)  # PIL image
      uploaded_img_path = "/content/drive/MyDrive/img/static/samples/"+ file.filename
      img.save(uploaded_img_path)
      if 'ban' in file.filename:
        val=1
      list1=fun(uploaded_img_path)
      if (len(list1))==0:
        nomatch='sad.png'
        return render_template('upload_image.html',nomatch=nomatch,text3="Product is Unavailable in the Store!!")
      up1=uploaded_img_path[34:]
      output=[]
      for i in range(len(list1)):
        output.append(filenames[list1[i]][34:])
      text="YOUR IMAGE"
      return render_template('upload_image.html',val=val,sc1=up1,output=output,text=text,text2="Product is Available in the Store!!")
  return render_template('upload_image.html')
app.run()