gmt gmtset MAP_FRAME_TYPE fancy 
gmt nearneighbor ./inv_interp_surface/interp_paraval.lst -S1d -G./inv_interp_surface/interp_paraval.lst.grd -I0.5 -R188.0/238.0/52.0/72.0 
gmt grd2xyz ./inv_interp_surface/interp_paraval.lst.grd -R188.0/238.0/52.0/72.0 > ./inv_interp_surface/interp_paraval.lst.HD 
