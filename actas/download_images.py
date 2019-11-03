# Import requests, shutil python module.
import requests
import shutil
import re
import json
import ast


def find_numero_mesa():
    # read file
    with open('demo.json', 'r') as myfile:
        data=myfile.read()
        result = re.match(r'"NumeroMesa" : +"[0-9]+"', data)
        print(result)


def read_json():
    json_file = "demo_2.json"
    with open(json_file, 'r') as f:
        mesas_dict = json.load(f)


    return mesas_dict



def download_data(id):
        
    # This is the image url.
    image_url = f"https://computo.oep.org.bo/resul/imgActa/{id}.jpg"
    # Open the url image, set stream to True, this will return the stream content.
    resp = requests.get(image_url, stream=True)
    # Open a local file with wb ( write binary ) permission.
    local_file = open(f'{id}.jpg', 'wb')
    # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
    resp.raw.decode_content = True
    # Copy the response stream raw data to local image file.
    shutil.copyfileobj(resp.raw, local_file)
    # Remove the image url response object.
    del resp



def main():
        
    mesas = read_json()

    for m in mesas:
        mesa_string = m[0]["content"]
        mesa_list = mesa_string.split(":")
        mesa_id = mesa_list[1]
        mesa_id += "1"
        _id = int(mesa_id.replace('"', "")) 

        print(_id)
        download_data(_id)

        # for c in  m[0]:
        #     print(c)


main()