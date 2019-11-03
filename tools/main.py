#Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#PDX-License-Identifier: MIT-0 (For details, see https://github.com/awsdocs/amazon-rekognition-developer-guide/blob/master/LICENSE-SAMPLECODE.)

import boto3
import cv2
import os

def preprocess_image(img):
    
    h,w,c = img.shape

    new_w = 420 # ~6
    new_h = 260 # ~6

    img_r = cv2.resize(img, (new_w, new_h))

    return img_r




def detect_text(photo, bucket):

    client=boto3.client('rekognition')

    response=client.detect_text(Image={'S3Object':{'Bucket':bucket,'Name':photo}})
                        
    textDetections=response['TextDetections']
    print ('Detected text\n----------')
    for text in textDetections:
            print ('Detected text:' + text['DetectedText'])
            print ('Confidence: ' + "{:.2f}".format(text['Confidence']) + "%")
            print ('Id: {}'.format(text['Id']))
            if 'ParentId' in text:
                print ('Parent Id: {}'.format(text['ParentId']))
            print ('Type:' + text['Type'])
            print()
    return len(textDetections)

def main():

    bucket='lucam-raw-assets'
    image_name='509401.jpg'
    text_count=detect_text(photo,bucket)
    print("Text detected: " + str(text_count))

    try: 
        
        img = cv2.imread(image_name)

        img_r = preprocess_image(img)


    img_r = preprocess_image(img)

if __name__ == "__main__":
    main()