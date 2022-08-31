
# Calculates distance of the landamark based on your current position on the map using lat/lng 

import math 

def distance(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    c = math.cos
    a = 0.5 - c((lat2 - lat1) * p) / 2 + c(lat1*p)*c(lat2*p)*(1 - c((lon2 - lon1)*p)) / 2
    return round(12742 * math.asin(math.sqrt(a)),1)
