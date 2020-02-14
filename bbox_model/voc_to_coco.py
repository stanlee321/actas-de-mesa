import os
import xml.etree.ElementTree as ET
import xmltodict
import json
from xml.dom import minidom
from collections import OrderedDict
import glob

from labels import Labels

#attrDict = {"images":[{"file_name":[],"height":[], "width":[],"id":[]}], "type":"instances", "annotations":[], "categories":[]}

#xmlfile = "000023.xml"

def to_dict(input_ordered_dict):
	return json.loads(json.dumps(input_ordered_dict))


class VOC2COCO:
	def __init__(self):
		
		self.labels = Labels()

	def generateVOC2Json(self, xmlFiles):
		attrDict = dict()

		attrDict["categories"] = [self.labels.create_attrDict()]
		

		images = list()
		annotations = list()

		image_id = 0

		for file_ in xmlFiles:
			image_id = image_id + 1
				
			annotation_path = file_
			image = dict()

			doc = xmltodict.parse(open(annotation_path).read())

			image['file_name'] = str(doc['annotation']['filename'])
			image['height'] = int(doc['annotation']['size']['height'])
			image['width'] = int(doc['annotation']['size']['width'])
			image['id'] = image_id
			#print ("File Name: {} and image_id {}".format(file_, image_id))
			images.append(image)

			id1 = 1

			if 'object' in doc['annotation']:
				for obj in doc['annotation']['object']: #48 boxes
					
					for value in attrDict["categories"]:
						#print(value)
						annotation = dict()
						obj = to_dict(obj)
						for val in value:
							if str(obj['name']) == val["name"]:
								#print str(obj['name'])
								#annotation["segmentation"] = []
								annotation["iscrowd"] = 0
								#annotation["image_id"] = str(doc['annotation']['filename']).split('.jpg')[0] #attrDict["images"]["id"]
								annotation["image_id"] = image_id
								x1 = float(obj["bndbox"]["xmin"]) # - 1
								y1 = float(obj["bndbox"]["ymin"]) # - 1
								x2 = float(obj["bndbox"]["xmax"]) # - x1
								y2 = float(obj["bndbox"]["ymax"]) # - y1
								annotation["bbox"] = [x1, y1, x2, y2]
								annotation["area"] = float(x2 * y2)
								annotation["category_id"] = val["id"]
								annotation["ignore"] = 0
								annotation["id"] = id1
								annotation["segmentation"] = [[x1,y1,x1,(y1 + y2), (x1 + x2), (y1 + y2), (x1 + x2), y1]]
								id1 +=1

								annotations.append(annotation)

			else:
				print ("{} doesn't have any object".format(file_))
			#image_id = image_id + 1
			#print(len(annotations))

				

		attrDict["images"] = images	
		attrDict["annotations"] = annotations
		attrDict["type"] = "instances"

		attrDict['categories'] = attrDict['categories'][0]
		classes = {}
		for o in attrDict['categories']:
			classes[o['id']] = o['name']
		#print attrDict
		jsonString = json.dumps(attrDict)
		with open("receipts_valid.json", "w") as f:
			f.write(jsonString)

	def main(self, trainXMLFiles = "path/to/annotation/xml/files"):

		self.generateVOC2Json(trainXMLFiles)
	
if __name__ == "__main__":

	voc2Coco = VOC2COCO()

	path = "/home/stanlee321/Desktop/DeepLearning/elecciones/actas/cuts"

	xml_files = glob.glob(f"{path}/*.xml")

	voc2Coco.main(trainXMLFiles=xml_files)