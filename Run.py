import urllib.request

import cv2
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from scipy.spatial.distance import cosine


##Function to load an image from an image URL
def load_image_from_url(url):
    with urllib.request.urlopen(url) as url:
        s = url.read()
        arr = np.asarray(bytearray(s), dtype=np.uint8)
        return cv2.imdecode(arr, -1) # 'load it as it is'

ResNet50()  ##Extract facial features using the model
base_model = ResNet50(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv1_relu').output)

##Load images of idols from each agency and extract their facial characteristics

companies = ['sm', 'jyp', 'yg']
company_images = {
    'sm': ['https://ifh.cc/g/M3q0fL.jpg','https://ifh.cc/g/cvG7tD.jpg','https://ifh.cc/g/DadL0m.jpg','https://ifh.cc/g/waxBB8.jpg','https://ifh.cc/g/CJ0X6V.jpg','https://ifh.cc/g/0fMP5r.jpg','https://ifh.cc/g/kBlMcS.jpg','https://ifh.cc/g/1Mcjsw.jpg','https://ifh.cc/g/VryvtZ.jpg','https://ifh.cc/g/glsmzP.jpg','https://ifh.cc/g/NkBn3C.jpg','https://ifh.cc/g/Rv5qQo.jpg','https://ifh.cc/g/9ZXGjD.jpg','https://ifh.cc/g/fpXlxd.jpg','https://ifh.cc/g/M4CFYG.jpg','https://ifh.cc/g/RBQ1Sd.jpg','https://ifh.cc/g/MFmpVS.jpg'],
    'jyp': ['https://ifh.cc/g/3y1SOh.jpg', 'https://ifh.cc/g/1HQgNa.jpg','https://ifh.cc/g/YplXgt.jpg','https://ifh.cc/g/Y3W03z.jpg','https://ifh.cc/g/fA9bcQ.jpg','https://ifh.cc/g/XTDPcd.jpg','https://ifh.cc/g/Bb5Nb2.jpg','https://ifh.cc/g/2ajD9l.jpg','https://ifh.cc/g/nSndcs.jpg','https://ifh.cc/g/Zsq1HL.jpg','https://ifh.cc/g/yOQmgF.jpg','https://ifh.cc/g/okFFav.webp','https://ifh.cc/g/43tcAq.jpg','https://ifh.cc/g/ADYfox.jpg','https://ifh.cc/g/kwrg8m.jpg','https://ifh.cc/g/slLmqv.jpg','https://ifh.cc/g/PCRfJO.jpg'],
    'yg': ['https://ifh.cc/g/srgpkK.jpg', 'https://ifh.cc/g/qYHw1t.jpg','https://ifh.cc/g/kBTrdD.jpg','https://ifh.cc/g/wVlQr2.jpg','https://ifh.cc/g/fCXSV6.jpg','https://ifh.cc/g/Qcy1vH.jpg','https://ifh.cc/g/qBCY7a.jpg','https://ifh.cc/g/vjOMQV.jpg','https://ifh.cc/g/mjX9z6.webp'],
}

company_faces = {}
for company in companies:
    company_faces[company] = []
    for image_url in company_images[company]:
        image = load_image_from_url(image_url) # 이미지 로딩 부분 수정
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        features = model.predict(image)
        company_faces[company].append(features)

#Calculate the average facial characteristics of each agency
company_avg_features = {}
for company in companies:
    company_avg_features[company] = np.mean(company_faces[company], axis=0)


#Load the user's image and extract facial features
user_image_path = ''
user_image = cv2.imread(user_image_path)
user_image = cv2.resize(user_image, (224, 224))
user_image = np.expand_dims(user_image, axis=0)
user_image = preprocess_input(user_image)
user_features = model.predict(user_image)
user_features = user_features.flatten()
user_features = model.predict(user_image)
user_features = user_features.flatten()

##Calculate the cosine similarity of each agent's and user's facial attributes
similarity = {}
for company in companies:
    user_features_flattened = user_features.flatten() if len(user_features.shape) > 1 else user_features
    company_features_flattened = company_avg_features[company].flatten() if len(
        company_avg_features[company].shape) > 1 else company_avg_features[company]

    print(f'user_features shape: {user_features_flattened.shape}')
    print(f'company_avg_features shape: {company_features_flattened.shape}')

    similarity[company] = cosine(user_features_flattened, company_features_flattened)

most_similar_company = min(similarity, key=similarity.get)
print('Most Similar Agencies: ', most_similar_company)
# Normalize the cosine similarity to make it between 0 and 1
normalized_similarity = {k: (1-v)/2 for k, v in similarity.items()}

# Convert it to percentage
percentage_similarity = {k: v*100 for k, v in normalized_similarity.items()}

print('Similarity to each agency (in percent):')
for company, percent in percentage_similarity.items():
    print(f'{company}: {percent:.2f}%')

print('Similarity to most similar agencies (percent): {:.2f}%'.format(percentage_similarity[most_similar_company]))
