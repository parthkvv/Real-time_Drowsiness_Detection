# Extracting the vehicle's coordinates 

import urllib.request
import json

def co_or():
    contents = urllib.request.urlopen("https://jsonbase.com/gps_ford/data/").read().decode("utf8")
    jd = json.loads(contents)
    # j_file = open()
    dp = jd['Lat']
    dp1 = jd['Lon']
    return (dp, dp1)