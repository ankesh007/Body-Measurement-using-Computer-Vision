# Body Measurements using CV
#### *Submitted by Ankesh Gupta(2015CS10435), Krunal Shah(2015EE10476), Saket Dingliwal(2015CS10254)*

The goal of assignment was to make ***real-world body part*** measurements using 2D images. The repository includes methods to measure ***shoulder distance, wrist-to-shoulder measurement, and waist approximation***. For implementation details and other nitty-gritty associated with the project, its recommended to lookup the attached presentation named: ***Presentation.pdf***.

To run code, change to *src/ directory* and type linux shell:
```
python code2.py -i1 <path to Image1> -i2 <path to Image2> -i3 <path to Image3> -a <Correction_mode>
```


## Notes on running the code

1. The code has been tested and developed in ***python2*** using ***Ubuntu 16.04***.
2. Images required for above code are specific. The details are given ***below***.
3. Correction_mode parameter is the flag, which tells code whether to perform ***affine + metric correction*** on the image.

### Image1 

This image is with the subject holding a ***checkered board*** in hands. This helps measure ***shoulder*** distance. Check the image below.

![alt text](https://github.com/ankesh007/Body-Measurement-using-Computer-Vision/blob/master/Images/final_saket1.jpg)

Checkered board is special. Its helps in ***calibration*** of camera image world for ***3D measurements***. If you use any other chess-type board, measure the side length of ***unit square and*** change ***global ref_ht*** parameter in ***code2.py***.

### Image2 

This image is with the subject spreading out his hands. This helps in ***wrist-to-shoulder*** measurement, and provide width of waist's projection.

![alt text](https://github.com/ankesh007/Body-Measurement-using-Computer-Vision/blob/master/Images/final_saket2.jpg)

### Image3 

This image is capturing side-view of subject. This provide thickness of waist and helps complete ***waist*** measurement.

![alt text](https://github.com/ankesh007/Body-Measurement-using-Computer-Vision/blob/master/Images/final_saket3.jpg)


Waist is modelled as an ***ellipse*** and measured analogous to finding ***perimeter of ellipse***. Hence ***appromation*** is mentioned.

When code runs, you will be shown points selected by our ***heuristic*** as to-be shoulder/wrist. If suspicious/incorrect, you can ***explicitly*** select those points on image and pressing `esc` key thereafter.
