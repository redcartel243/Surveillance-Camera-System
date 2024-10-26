import json
import os

datasave = {"ON": False, "OFF": False, "ID": 0, "user1": "", "user2": "", "promptNotF": False, "LimitReached": False,
            "alarm": "1", "numberRep": 1, "voice": 0, "AlarmText": "intruder!"}


def load():  # this method is used to load the data from the txt file
    json_file = open(r"src\datasave.txt")
    name = json.load(json_file)
    json_file.close()
    return name


def save(data):  # this method is used to save the data in the text file
    file = open(r'src\datasave.txt', 'w')
    json.dump(data, file)
    file.close()


def setToDefault():  # this method is used to set everything back to default configuration
    save(datasave)
    directory = r'C:\Users\appli\PycharmProjects\IntruderFaceDetection\DataSet'
    for files in os.listdir(directory):
        os.remove("C:/Users/appli/PycharmProjects/IntruderFaceDetection/DataSet/" + str(files))
