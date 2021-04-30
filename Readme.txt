==========================
ForgeryNet Benchmark
==========================

Directory Information

------------

- Train
  - images
    The images organized in subdirectories based on both video methods and image methods. See IMAGES AND CLASS LABELS section below for more info.
  - videos
    The images organized in subdirectories based on video methods. See VIDEO AND LOCATE LABELS section below for more info.
  - spatial_localize
    Forgery locations per image. 


=========================
IMAGES AND LOCATE LABELS:
=========================
Images are contained in the directory images*, with 15 "fake" subdirectories (one for each methods) and 4 "real" subdirectories.

------- List of image files ------
The list of image file names is contained in the file, with each line corresponding to one image:

<image_name> <binary_cls_label> <triple_cls_label> <16cls_label>
------------------------------------------

=========================
VIDEO AND CLASS LABELS:
=========================
Videos are contained in the directory video*/, with 8 "fake" subdirectories (one for each methods) and 4 "real" subdirectories.

------- List of image files ------
The list of image file names is contained in the file, with each line corresponding to one image:

<segmentation length> <video_name> [segmentation info]*segmentation length <binary_cls_label> <triple_cls_label> <16cls_label>

-------  segmentation info  ------
<start frame> <during frames count> <fake/real>

------------------------------------------

=========================
MD5 file:
=========================
750e34521e61f81486a678d5b76a5ef3  Training.tar             
c6fd18252b5e0ae04a5b163297b5cb28  Validation_with_real.tar 
7d2c0cf50bdc77890ace1ba4461e1893  Test_with_real.tar
8558b99366ccb93cd9da56f600f0901f  test_list.tar
8bda9b0a6dc7721e5f1d0b6b3793ce2a  train_list.tar
a280bb501166a01b8d376df2b5b9ae13  val_list.tar
 
