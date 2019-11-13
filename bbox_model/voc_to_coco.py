import os
import xml.etree.ElementTree as ET
import xmltodict
import json
from xml.dom import minidom
from collections import OrderedDict

from labels import Labels

#attrDict = {"images":[{"file_name":[],"height":[], "width":[],"id":[]}], "type":"instances", "annotations":[], "categories":[]}

#xmlfile = "000023.xml"


class VOC2COCO:
	def __init__(self):
		
		self.labels = Labels()

	def generateVOC2Json(self, rootDir, xmlFiles):
		attrDict = dict()

		attrDict["categories"] = [
			self.labels.create_attrDict()
		]

		images = list()
		annotations = list()

		for root, dirs, files in os.walk(rootDir):
			image_id = 0
			for file in xmlFiles:
				image_id = image_id + 1
				if file in files:
					
					#image_id = image_id + 1
					annotation_path = os.path.abspath(os.path.join(root, file))
					
					#tree = ET.parse(annotation_path)#.getroot()
					image = dict()
					#keyList = list()
					doc = xmltodict.parse(open(annotation_path).read())
					#print doc['annotation']['filename']
					image['file_name'] = str(doc['annotation']['filename'])
					#keyList.append("file_name")
					image['height'] = int(doc['annotation']['size']['height'])
					#keyList.append("height")
					image['width'] = int(doc['annotation']['size']['width'])
					#keyList.append("width")

					#image['id'] = str(doc['annotation']['filename']).split('.jpg')[0]
					image['id'] = image_id
					print ("File Name: {} and image_id {}".format(file, image_id))
					images.append(image)

					id1 = 1

					if 'object' in doc['annotation']:
						for obj in doc['annotation']['object']:
							for value in attrDict["categories"]:
								annotation = dict()
								#if str(obj['name']) in value["name"]:
								if str(obj['name']) == value["name"]:
									#print str(obj['name'])
									#annotation["segmentation"] = []
									annotation["iscrowd"] = 0
									#annotation["image_id"] = str(doc['annotation']['filename']).split('.jpg')[0] #attrDict["images"]["id"]
									annotation["image_id"] = image_id
									x1 = int(obj["bndbox"]["xmin"])  - 1
									y1 = int(obj["bndbox"]["ymin"]) - 1
									x2 = int(obj["bndbox"]["xmax"]) - x1
									y2 = int(obj["bndbox"]["ymax"]) - y1
									annotation["bbox"] = [x1, y1, x2, y2]
									annotation["area"] = float(x2 * y2)
									annotation["category_id"] = value["id"]
									annotation["ignore"] = 0
									annotation["id"] = id1
									annotation["segmentation"] = [[x1,y1,x1,(y1 + y2), (x1 + x2), (y1 + y2), (x1 + x2), y1]]
									id1 +=1

									annotations.append(annotation)
					
					else:
						print ("File: {} doesn't have any object".format(file))
					#image_id = image_id + 1
					
				else:
					print ("File: {} not found".format(file))
				

		attrDict["images"] = images	
		attrDict["annotations"] = annotations
		attrDict["type"] = "instances"

		#print attrDict
		jsonString = json.dumps(attrDict)
		with open("receipts_valid.json", "w") as f:
			f.write(jsonString)

	def main(self, rootDir = "path/to/annotation/xml/files"):
			
		trainFile = "./valid.txt"

		trainXMLFiles = list()

		with open(trainFile, "rb") as f:
			for line in f:
				fileName = line.strip()
				print (fileName)
				trainXMLFiles.append(str(fileName) + "_label.xml")

		self.generateVOC2Json(rootDir, trainXMLFiles)
	
if __name__ == "__main__":

	voc2Coco = VOC2COCO()

	voc2Coco.main(rootDir="../actas/cuts/")