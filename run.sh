#!/bin/sh

mkdir matches;
image_directory="./ec1m_landmark_images";
cd $image_directory;
for first in *.jpg ;
do 
    first_cropped=`echo $first | sed -e 's/\.jpg$//'`
    mkdir ../matches/$first_cropped ;
    for second in *.jpg ; 
    do
        second_cropped=`echo $second | sed -e 's/\.jpg$//'`
        ../SURF_Affine_Worker $first $second ../matches/$first_cropped/$second_cropped;
    done ; 
done;
cd -;
