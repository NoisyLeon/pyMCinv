import obspy
from obspy.clients.fdsn.client import Client

client      = Client('IRIS')
catISC      = client.get_events(minmagnitude=3.0, catalog='ISC', starttime=obspy.UTCDateTime('19910101'), 
                minlatitude=52., maxlatitude=73., minlongitude=-170., maxlongitude=-120.)
