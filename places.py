#  Uses Google Map API to povide neaby places based on the location provided. 

import requests
import googlemaps
import time
from put_html import co_or
from distance import distance
import json


def recommend_places():
    k="YOUR_KEY" 

    gmaps = googlemaps.Client(key=k)

    params = {
        'query': ['restaurants', 'cafe', 'hotel'],
        'location': co_or(), 
        'radius': 20000
    }
    x = gmaps.places(**params)
    # print (len(x['results'])) # outputs 20 (which is maximum per 1 page result) 

    dictd={}
    dictp={}
    
    for i in range(min(len(x['results']),3)):
        lat1, lon1 = float(co_or()[0]), float(co_or()[1])
        lat2 = x['results'][i]['geometry']['location']['lat']
        lon2 = x['results'][i]['geometry']['location']['lng']
        dist = distance(lat1, lon1, lat2, lon2)
        dictp[i]= x['results'][i]['name']
        dictd[i]= dist
        #print (x['results'][i]['name'],", ", "Distance: ", dist)

    url = 'https://jsonbase.com/maps_dist/data'
    headers = {'content-type':'application/json'}
    data = json.dumps(dictd)
    r = requests.request("PUT",url,data = data,headers=headers)
    # print(x['results'][0])

    url = 'https://jsonbase.com/maps_place/data'
    headers = {'content-type':'application/json'}
    data = json.dumps(dictp)
    r = requests.request("PUT",url,data = data,headers=headers)

