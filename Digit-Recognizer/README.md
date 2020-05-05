# Digit-Recognizer
Using Google Datalab and Tensorflow

View Kaggle Kernel: https://www.kaggle.com/jander081/tf-using-datalab-google

The best score I achieved with this model was .98942. I'm sure I can get it up there a bit more with endless tweaking and training, but the score is decent and definitely served as a tensorflow refresher for me. Most of the notebook contents were borrowed from other tutorials and kernels (not trying to reinvent the wheel here). The main difference between this and other kernels is that a (cloud) Nvidia Tesla K80 GPU was used through the Datalab platform. This model is a bit rough on CPUs, so I'll run the code in the following notebook using truncated datasets.


For those of you who don't already know, Google Cloud Services offers a cloud tool built on Jupyter. You can learn more about it at https://cloud.google.com/datalab/. Using the service requires a credit card and some familiarity with google cloud. This notebook contains some useful code for importing and exporting data (csv) files. You can start learning about Datalab at https://cloud.google.com/datalab/docs/quickstart.
