from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
import numpy as np
import face_recognition, faiss

#faiss
face_index = faiss.read_index('face_index.bin')
labels = np.load('labels.npy')
def img2vec(paths):
    face_encodes = []
    for path in paths:
        img = face_recognition.load_image_file('media/'+path)
        faceLocation = face_recognition.face_locations(img)
        if len(faceLocation) > 0:
            for (top,right,bottom,left) in faceLocation:
                face_img = img[top:bottom,left:right]
                face_encode = face_recognition.face_encodings(face_img)[0]
                face_encodes.append(face_encode)
    query = np.array(face_encodes,dtype=np.float32)
    query = query.reshape(-1,128)
    return query
# Create your views here.
def upload(request):
    if request.method == 'POST' and request.FILES:
        data = []
        fs = FileSystemStorage()
        file = request.FILES.get('file')
        filename = fs.save(file.name, file)
        url_img = fs.url(filename)
        data.append(url_img.split('/')[-1])
        query = img2vec(data)
        if len(query) == 0:
            label = 'not detect'
        else:
            _,ids = face_index.search(query,k=1)
            label = labels[ids[0]]

        return render(request,'upload.html',{'img':url_img,'name':label}) 

    return render(request,'upload.html')    
