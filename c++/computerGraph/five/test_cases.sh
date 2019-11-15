#!/bin/sh
rm *.tga
rm *.bmp

./a5soln -input scene06_bunny_1k.txt  -size 300 300 -output 6.bmp\
   -shadows -bounces 4 -jitter -filter
./a5soln -input scene10_sphere.txt  -size 300 300 -output 10.bmp\
   -shadows -bounces 4 -jitter -filter
./a5soln -input scene11_cube.txt  -size 300 300 -output 11.bmp\
 -shadows -bounces 4 -jitter -filter
./a5soln -input scene12_vase.txt  -size 300 300 -output 12.bmp\
 -shadows -bounces 4 -jitter -filter

