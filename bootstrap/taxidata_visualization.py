# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 16:19:04 2018

@author: mtiw2
"""
#http://www.danielforsyth.me/mapping-nyc-taxi-data/

import pandas as pd
import google.cloud


df=pd.io.gbq.read_gbq("""  
SELECT ROUND(pickup_latitude, 4) as lat, ROUND(pickup_longitude, 4) as long, COUNT(*) as num_pickups  
FROM [nyc-tlc:yellow.trips_2015]  
WHERE (pickup_latitude BETWEEN 40.61 AND 40.91) AND (pickup_longitude BETWEEN -74.06 AND -73.77 )  
GROUP BY lat, long  
""", project_id='taxi-1029')

