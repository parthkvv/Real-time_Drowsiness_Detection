# Uses Google Vision API for Emotion Detection 

import base64
from songs import joy, sad, neutral


def detect_faces(path):
    """Detects faces in an image."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.face_detection(image=image)
    faces = response.face_annotations

    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    # print('Faces:')

    values = {'UNKNOWN':1, 'VERY_UNLIKELY':2, 'UNLIKELY':3, 'POSSIBLE':4,
                        'LIKELY':5, 'VERY_LIKELY':6}

    for face in faces:
        a = likelihood_name[face.anger_likelihood]
        j = likelihood_name[face.joy_likelihood]
        s = likelihood_name[face.sorrow_likelihood]

        f = max(values.get(a),values.get(j),values.get(s))  
        if f==values.get(a) and f==values.get(j) and f==values.get(s):
          print('Neutral') 
          neutral()
        elif f==values.get(a):
          print('Angry')
          neutral()
        elif f==values.get(j):
          print('Happy')
          joy()
        else:
          print('Sad') 
          sad()
        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in face.bounding_poly.vertices])

        # print('face bounds: {}'.format(','.join(vertices)))

detect_faces('/home/yogesh/Downloads/image2.jpg')