from obspy.clients.fdsn.client import Client
import numpy as np
import timeit
import matplotlib.pyplot as plt
import obspy

startdate='1991-01-01'
enddate='2015-02-01'
starttime   = obspy.core.utcdatetime.UTCDateTime(startdate)
endtime     = obspy.core.utcdatetime.UTCDateTime(enddate)
        
client  = Client('IRIS')
cat = client.get_events(catalog='NEIC PDE', minlatitude=55, maxlatitude=65, minlongitude=-170, maxlongitude=-140, \
                        mindepth=70., starttime=starttime, endtime=endtime)
