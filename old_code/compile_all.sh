#!/bin/bash
cd DISP
bash compile_disp.sh
cp fast_surf.so ..
cd ../RF
bash compile_RF.sh
cp theo.so ..

