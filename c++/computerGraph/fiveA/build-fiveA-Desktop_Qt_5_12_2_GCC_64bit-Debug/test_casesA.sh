#!/bin/sh
rm *.tga
rm *.bmp

./a5 -input scene01_plane.txt  -size 200 200 -output 1.bmp -bounces 0
./a5 -input scene02_cube.txt  -size 200 200 -output 2.bmp -bounces 0
./a5 -input scene03_sphere.txt  -size 200 200 -output 3.bmp -bounces 0
./a5 -input scene04_axes.txt  -size 200 200 -output 4.bmp -bounces 0
./a5 -input scene05_bunny_200.txt  -size 200 200 -output 5.bmp -bounces 0
./a5 -input scene06_bunny_1k.txt  -size 200 200 -output 6.bmp -bounces 0
./a5 -input scene07_shine.txt  -size 200 200 -output 7.bmp -bounces 0
./a5 -input scene08_c.txt -size 200 200 -output 8.bmp -bounces 0
./a5 -input scene09_s.txt -size 200 200 -output 9.bmp -bounces 0
