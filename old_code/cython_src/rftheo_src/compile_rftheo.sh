echo "gfortran four1.f -c -o four1.o -O3 -fPIC -lgfortran"
gfortran four1.f -c -o four1.o -O3 -fPIC -lgfortran
echo "gfortran qlayer.f -c -o qlayer.o -O3 -fPIC -lgfortran"
gfortran qlayer.f -c -o qlayer.o -O3 -fPIC -lgfortran
echo "gfortran theo.f -c -o theo.o -O3 -fPIC -lgfortran"
gfortran theo.f -c -o theo.o -O3 -fPIC -lgfortran

#gfortran -shared -O3 -fPIC -lgfortran -ffixed-line-length-none calcul.o flat1.o init.o mchdepsun.o surfa.o fast_surf.o -o fast_surf.so
