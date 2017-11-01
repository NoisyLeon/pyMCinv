import math;
import string;
import sys;
import struct;

fsize = struct.calcsize('f');
ffsize = struct.calcsize('ffff');

binfile = open(sys.argv[1],"rb");
k = 1;
while k > 0:
    intsize = struct.calcsize('iiiiff');
    data = binfile.read(intsize);
    if data == '':
        break;
    num = struct.unpack('iiiiff', data);
#    print num;

    for i in range (num[3]):
        data = binfile.read(fsize);
        num = struct.unpack('f',data);
#        print num[0];
    intsize = struct.calcsize('ii');
    data = binfile.read(intsize);
    num = struct.unpack('ii',data);
#    print num;
    n0 = num[0];
    n1 = num[1];
    for i in range (n1):
        data = binfile.read(fsize);
        num = struct.unpack('f',data);
#        print num[0];
    for i in range (n0):
        data = binfile.read(ffsize);
        num = struct.unpack('ffff',data);
#        print num;
