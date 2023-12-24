## Julia_Set
Python fractal generator based on the Julia set


>HOW DOES IT WORK 

The julia_set function works by creating a 2D array of size x_res*y_res, and then filling it with the iteration ratio of each pixel (the number of iterations it took to reach the max_iter value) and then plotting it using imshow from matplotlib (the iteration ratio is used to determine the color of each pixel).

The generator works with all types of resolutions although it is recommended to use even and similar numbers for x_res and y_res.


Here's some example with c = -0.8 + 0.16j and a resolution of 100x100 :

![100x100](https://github.com/Aatrick/Julia_Set/assets/113598245/3324f05a-7db3-4c25-b2de-9b15d1823499)

And here's the same c value but 2000x2000 as resolution instead :

![2000x2000](https://github.com/Aatrick/Julia_Set/assets/113598245/57d16ff2-9c50-411e-aad6-e1f07ddedc80)

The julia.py file is completely callable from a Powershell terminal and will ask, how you want it to behave.

![zoom cmd](https://github.com/Aatrick/Julia_Set/assets/113598245/af520068-b741-4ad2-905f-dab6503a3da5)

-------------------------------------------------------------------------------------------------------------------------------------------------------

Other than simply plotting images and saving them, the generator is also able to generate GIFs, there are two modes for now, one zooming progressively into one part of the fractal :

![zoom gif](https://github.com/Aatrick/Julia_Set/assets/113598245/79049b69-3945-4ab6-8410-a20657f8f650)

And there's the second mode, making the c value fluctuate, to see how these changes impact the output.

![gif b2b](https://github.com/Aatrick/Julia_Set/assets/113598245/9538663b-d8b7-49dd-89c2-0f312d83af40)

-------------------------------------------------------------------------------------------------------------------------------------------------------

#INSTALLATION 

Simply git clone this repository https://github.com/Aatrick/Julia_Set.git

then do in the cloned directory :
```sh
pip install -r requirements.txt
```

and you should be ready to go !
