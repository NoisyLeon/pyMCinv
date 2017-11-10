ls ./*py > list
#cat list | sed -i 's/float32/float64/g' 
while read p; do
  sed -i 's/float64/float32/g' $p
done <list
rm list

