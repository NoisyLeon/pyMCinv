import numpy as np

def read_vel(fname):
    latarr  = []
    lonarr  = []
    dndtarr = []
    dedtarr = []
    SNdarr  = []
    SEdarr  = []
    with open(fname, 'rb') as fid:
        isheader    = False
        for line in fid.readlines():
            if line[0] == '*':
                isheader    = True
                line_arr    = np.array(line.split())
                ilat        = np.where(line_arr == 'Ref_Nlat')[0][0]
                ilon        = np.where(line_arr == 'Ref_Elong')[0][0]
                idndt       = np.where(line_arr == 'dN/dt')[0][0]
                idedt       = np.where(line_arr == 'dE/dt')[0][0]
                iSNd        = np.where(line_arr == 'SNd')[0][0]
                iSEd        = np.where(line_arr == 'SEd')[0][0]
                continue
            if not isheader:
                continue
            line_arr    = line.split()
            lat         = float(line_arr[ilat])
            lon         = float(line_arr[ilon])
            dndt        = float(line_arr[idndt])
            dedt        = float(line_arr[idedt])
            SNd         = float(line_arr[iSNd])
            SEd         = float(line_arr[iSEd])
            latarr.append(lat)
            lonarr.append(lon)
            dndtarr.append(dndt)
            dedtarr.append(dedt)
            SNdarr.append(SNd)
            SEdarr.append(SEd)
    return np.array(latarr), np.array(lonarr), np.array(dndtarr), np.array(dedtarr),\
            np.array(SNdarr), np.array(SEdarr)
            # return lat, lon, dndt, dedt, SNd, SEd
            
            
# fid = open('pbo.final_nam08.vel', 'rb')
# alllines = fid.readlines()

# l = read_vel('pbo.final_nam08.vel')