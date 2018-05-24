cdir=`pwd`
echo $cdir

cd $cdir/fast_surf_src
./compile_fast_surf.sh
cd $cdir

cd $cdir/rftheo_src
./compile_rftheo.sh
cd $cdir

python setup.py build_ext --inplace
