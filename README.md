# Body Measurements using CV
#### *Submitted by Ankesh Gupta(2015CS10435), Krunal Shah(2015EE10476), Saket Dingliwal(2015CS10254)*

The goal of assignment was to make ***real-world body part*** measurements using 2D images. The repository includes methods to measure ***shoulder distance, wrist-to-shoulder measurement, and waist approximation***. For implementation details and other nitty-gritty associated with the project, its recommended to lookup the attached presentation named: ***Presentation.pdf***.

To run code, type linux shell:
```
python code2.py -i1 <Image1> -i2 <Image2> -i3 <Image3> -a <Correction_mode>
```


## Notes on running the code

1. The code has been tested and developed in python2 using Ubuntu 16.04.
2. Images required for above code are specific. The details are given below.
3. Correction_mode parameter is just the flag, which tells code whether to perform affine + metric correction on the image.

### Image1 

This image is with the subject holding a checkered board in hands. Check the image below.
![alt text](https://github.com/ankesh007/Body-Measurement-using-Computer-Vision/blob/master/Images/final1.jpg)

Checkered board is special. Its helps in calibration of camera image world for 3D measurements. If you use any other chess-type board, measure the side length of unit square and change global ref_ht parameter in code2.py.

