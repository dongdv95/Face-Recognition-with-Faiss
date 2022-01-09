from django.shortcuts import render
from face_recognition.api import face_encodings, face_locations
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from django.core.files.storage import FileSystemStorage
import faiss
import face_recognition
import numpy as np
import os
# Create your views here.

#faiss
face_index = faiss.read_index('face_index.bin')
labels = np.load('labels.npy')
def img2vec(paths):
    face_encodes = []
    for path in paths:
        img = face_recognition.load_image_file('api/upload/'+path)
        faceLocation = face_recognition.face_locations(img)
        if len(faceLocation) > 0:
            for (top,right,bottom,left) in faceLocation:
                face_img = img[top:bottom,left:right]
                face_encode = face_recognition.face_encodings(face_img)[0]
                face_encodes.append(face_encode)
    query = np.array(face_encodes,dtype=np.float32)
    query = query.reshape(-1,128)
    return query

@api_view(['POST'])
def search(request):
    
    if request.method == 'POST' and request.FILES:
        data = []
        fs = FileSystemStorage(location='api/upload/')
        files = request.FILES.getlist('file')
        for file in files:
            filename = fs.save(file.name, file)
            data.append(fs.url(filename).split('/')[-1])
        print(data)    
        query = img2vec(data)
        _,ids = face_index.search(query, k=1)
        label = [labels[i] for i in ids[0]]
    return Response(label,status=status.HTTP_200_OK)
