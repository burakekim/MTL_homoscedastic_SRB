# A MULTI-TASK DEEP LEARNING FRAMEWORK FOR BUILDING FOOTPRINT SEGMENTATION
This repository contains the code for the paper [A MULTI-TASK DEEP LEARNING FRAMEWORK FOR BUILDING FOOTPRINT SEGMENTATION](https://arxiv.org/abs/2104.09375)


Framework
---------------------
![alt text](ims/motiv.png)


Outputs
---------------------
![alt text](ims/1.png)
![alt text](ims/2.png)
![alt text](ims/3.png)
![alt text](ims/4.png)
![alt text](ims/5.png)


How to use it?
---------------------

Simply download the repository and follow the *main_notebook.ipynb* after modifying the paths and the parameters in the *params.py* script.

The [Spacenet6](https://arxiv.org/abs/2004.06500) dataset needs to be downloaded prior to running the main notebook. 

The code was implemented in Python(3.8) and PyTroch(1.14.0) on Windows OS. The *segmentation models pytorh* library is used as a baseline for implementation. Apart from main data science libraries, RS-specific libraries such as GDAL, rasterio, and tifffile are also required.

  
Feel free to get in touch with me via the e-mail below for the pre-trained weights.


Citation
---------------------

Please kindly cite the paper below if this code is useful and helpful for your research.

@misc{ekim2021multitask,
      title={A Multi-Task Deep Learning Framework for Building Footprint Segmentation}, 
      author={Burak Ekim and Elif Sertel},
      year={2021},
      eprint={2104.09375},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Contact Information:
--------------------

If you encounter bugs while using this code, please do not hesitate to contact me.

Burak Ekim: ekim19@itu.edu.tr<br>
