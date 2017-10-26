#!/bin/bash
#                 p+g   in.p          in.g           in.rf   in.mod  in.para  pps
STARTTIME=$(date +%s)
#python ../do_MC_lf.py 1 Q22A.com.txt in.rf Q22A.mod1 in.para  p1 > log
python ../do_MC_lf.py 1 Q22A.com.txt in.rf Q22A.mod1 in.para  p1 
ENDTIME=$(date +%s)
echo "It takes $(($ENDTIME - $STARTTIME)) seconds to complete this task..."
