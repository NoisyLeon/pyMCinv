import h5py
import numpy as np
outh5fname  = './CU_SDT1.0.mod.h5'
infname     = '../CU_SDT1.0.mod'

dset        = h5py.File(outh5fname)
firstgrd    = True
with open(infname, 'r') as fid:
    for line in fid.readlines():
        sline   = line.split()
        if len(sline) == 2:
            if firstgrd:
                lat     = float(sline[0])
                lon     = float(sline[1])
                firstgrd= False
                data    = np.array([])
                continue
            else:
                data    = data.reshape((data.size/10, 10))
                dset.create_dataset(name=str(lon)+'_'+str(lat), data=data, compression='gzip')
                lat     = float(sline[0])
                lon     = float(sline[1])
                data    = np.array([])
        elif len(sline) == 10:
            data    = np.append(data, float(sline[0]))
            data    = np.append(data, float(sline[1]))
            data    = np.append(data, float(sline[2]))
            data    = np.append(data, float(sline[3]))
            data    = np.append(data, float(sline[4]))
            data    = np.append(data, float(sline[5]))
            data    = np.append(data, float(sline[6]))
            data    = np.append(data, float(sline[7]))
            data    = np.append(data, float(sline[8]))
            data    = np.append(data, float(sline[9]))
        else:
            raise ValueError('Unexpected number of columns')

dset.close()
            
