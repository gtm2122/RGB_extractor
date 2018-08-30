Pre-requisites - 

You shuold have scikit-learn package and numpy (you probably have that)
avconv (sudo apt-get install lib-avtools, but look for the exact command)

usage

1)first move all the videos of the anotated in this folder, they should have .mov extension ( this is probably the default format )
If the videos are not .mov simply change the .sh file accordingly

2) open terminal and run

./vid_to_frame.sh

## the frames for the video will be stored in <name of movie>_frame_folder
## the annotations will be stored in <name of movie>_anno_folder

## Do not delete model_.pkl and ROI.py and the bash script. 
## Feel free to delete this readme
