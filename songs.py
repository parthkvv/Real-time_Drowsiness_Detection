# Song suggestion based on Emotion 

import json
import requests

def joy():
    file_joy = open("happy.txt", 'r')
    dict_j = {}

    for line in file_joy:
        fields = line.split(",")
        dict_j[0] = fields[0]
        dict_j[1] = fields[1]
        dict_j[2] = fields[2]

    url = 'https://jsonbase.com/songs_emo/data'
    headers = {'content-type':'application/json'}
    data = json.dumps(dict_j)
    r = requests.request("PUT",url,data = data,headers=headers)

def neutral():
    file_neutral = open("neutral.txt", 'r')
    dict_n = {}
    for line in file_neutral:
        fields = line.split(",")
        dict_n[0] = fields[0]
        dict_n[1] = fields[1]
        dict_n[2] = fields[2]
    url = 'https://jsonbase.com/songs_emo/data'
    headers = {'content-type':'application/json'}
    data = json.dumps(dict_n)
    r = requests.request("PUT",url,data = data,headers=headers)

def sad():
    file_sad = open("sad.txt", 'r')
    dict_s = {}
    for line in file_sad:
        fields = line.split(",")
        dict_s[0] = fields[0]
        dict_s[1] = fields[1]
        dict_s[2] = fields[2]

    url = 'https://jsonbase.com/song_emo/data'
    headers = {'content-type':'application/json'}
    data = json.dumps(dict_s)
    r = requests.request("PUT",url,data = data,headers=headers)

