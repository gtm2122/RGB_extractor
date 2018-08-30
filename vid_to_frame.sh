for i in *.mov;
 do
 mkdir "$i"_frame_folder
 avconv -i $i -r 30 -f image2 "$i"_frame_folder/%04d.png
 mkdir "$i"_anno_folder
 echo hahahaha
 python ROI.py --frame_dir "$i"_frame_folder/ --dest_dir "$i"_anno_folder
done
