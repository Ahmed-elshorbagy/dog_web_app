# import the necessary packages        
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse,HttpResponse
import numpy as np
import urllib
import json
import cv2
import os
from .face import dog_ear
from glob import glob
from .forms import ImgForm,UrlForm
import base64
import requests
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image  
from keras.models import load_model

import io
import tensorflow as tf
from PIL import Image
graph = tf.get_default_graph()
# define ResNet50 model
dog_names = [item[9:-1] for item in sorted(glob("test/*/"))]
ResNet50_model = ResNet50(weights='imagenet')
InceptionV3_model=load_model('dog/saved_models/weights.best.InceptionV3.hdf5') 
# define the path to the face detector
FACE_DETECTOR_PATH = r"{base_path}/haarcascades/haarcascade_frontalface_alt.xml".format(
	base_path=os.path.abspath(os.path.dirname(__file__)))

def main(request):
    	con={'form1':ImgForm,'form2':UrlForm}
    	return render(request,'main.html',con)
@csrf_exempt
def detect(request):
	# initialize the data dictionary to be returned by the request
	global graph
	with graph.as_default():
		data = {"success": False}

		# check to see if this is a post request
		if request.method == "POST":
			# check to see if an image was uploaded
			if request.FILES.get("image", None) is not None:
				# grab the uploaded image
				image,dog = _grab_image(stream=request.FILES["image"])
				ad=request.POST.get("overlay", None)
			# otherwise, assume that a URL was passed in
			else:
				# grab the URL from the request
				url = request.POST.get("url", None)
				ad=request.POST.get("overlay", None)
				# if the URL is None, then return an error
				if url is None:
					data["error"] = "No URL provided."
					return JsonResponse(data)

				# load the image and convert
				image,dog  = _grab_image(url=url)

			# convert the image to grayscale, load the face cascade detector,
			# and detect faces in the image
			
			img = cv2.cvtColor(dog_ear(image,ad), cv2.COLOR_BGR2RGB)
			img2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
			rects = detector.detectMultiScale(image)

			# construct a list of bounding boxes from the detection
			rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
			
			response=imgenc(img,rects)
			# if len(rects)<2:
			# 	breed = InceptionV3_predict_breed(img2)
			
			# update the data dictionary with the faces detected
			data.update({"num_faces": len(rects), "faces": rects, "success": True,"dog":str(dog),"img":response,'breed':"breed"})
		
		return render(request,'main.html',data)	
		# return a JSON response
		# return JsonResponse(data)
		

def _grab_image(path=None, stream=None, url=None):
	# if the path is not None, then load the image from disk
	if path is not None:
		image = cv2.imread(path)

	# otherwise, the image does not reside on disk
	else:	
		# if the URL is not None, then download the image
		if url is not None:
			resp = urllib.request.urlopen(url)
			data = resp.read()
			
		# if the stream is not None, then the image has been uploaded
		elif stream is not None:
			data = stream.read()
			
		# convert the image to a NumPy array and then read it into
		# OpenCV format
		image = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
		
		img = preprocess_input(path_to_tensor(image))
		prediction = np.argmax(ResNet50_model.predict(img))
		#boolean variable of presence of dog in image or not
		dog=((prediction <= 268) & (prediction >= 151)) 
		
	# return the image,and bool dog
	return image,dog

def imgenc(image,rects):
	# for (startX, startY, endX, endY) in rects:
	# 	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

	# r = 300.0 / image.shape[1]
	# dim = (300, int(image.shape[0] * r))
	
	# # perform the actual resizing of the image and show it
	# resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	CDF=Image.fromarray(image)
	in_mem_file=io.BytesIO()			
	CDF.save(in_mem_file, format = "PNG")
	# reset file pointer to start
	in_mem_file.seek(0)
	img_bytes = in_mem_file.read()

	base64_encoded_result_bytes = base64.b64encode(img_bytes)
	base64_encoded_result_str = base64_encoded_result_bytes.decode('ascii')
	return "data:image/png;base64,{0} ".format(base64_encoded_result_str)

def path_to_tensor(image):
	# resize the shape of image
	image2 =cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)
	# change the data type to float to be accepted
	image2 = image2.astype(np.float32)
	# convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
	return np.expand_dims(image2, axis=0)	
def extract_InceptionV3(tensor):
    return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))
def InceptionV3_predict_breed(image):
    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(image))
    # obtain predicted vector
    predicted_vector = InceptionV3_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

