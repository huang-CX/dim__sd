file=$1
echo $file
for i in 1 2 3
do 
    echo $i
    python ${file}.py --seed $i
done