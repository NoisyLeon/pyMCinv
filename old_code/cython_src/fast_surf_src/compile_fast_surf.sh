echo "gfortran calcul.f -c -o calcul.o -O3 -fPIC -lgfortran"
gfortran calcul.f -c -o calcul.o -O3 -fPIC -lgfortran 
echo "gfortran flat1.f -c -o flat1.o -O3 -fPIC -lgfortran"
gfortran flat1.f -c -o flat1.o -O3 -fPIC -lgfortran 
echo "gfortran init.f -c -o init.o -O3 -fPIC -lgfortran"
gfortran init.f -c -o init.o -O3 -fPIC -lgfortran 
echo "gfortran mchdepsun.f -c -o mchdepsun.o -O3 -fPIC -lgfortran"
gfortran mchdepsun.f -c -o mchdepsun.o -O3 -fPIC -lgfortran 
echo "gfortran surfa.f -c -o surfa.o -O3 -fPIC -lgfortran"
gfortran surfa.f -c -o surfa.o -O3 -fPIC -lgfortran 
echo "gfortran fast_surf.f -c -o fast_surf.o -O3 -fPIC -lgfortran -ffixed-line-length-none"
gfortran fast_surf.f -c -o fast_surf.o -O3 -fPIC -lgfortran -ffixed-line-length-none 


#gfortran -shared -O3 -fPIC -lgfortran -ffixed-line-length-none calcul.o flat1.o init.o mchdepsun.o surfa.o fast_surf.o -o fast_surf.so
