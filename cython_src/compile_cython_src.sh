cdir=`pwd`
echo $cdir

cd $cdir/fast_surf_src
./compile_fast_surf_src.sh
cd $cdir
python setup.py build_ext --inplace
