# Repository for codes related to the course completion work.

### This work aims to build a convolutional network to detect, initially, the presence of deviation in patients. From the audios obtained from examinations made by phonoaudiologists, converts them to spectograms and then an analysis is made using a convolutional network.

### Configuring and installing the required dependencies to run the codes:

Install [Minconda3](https://conda.io/en/latest/miniconda.html) according to your system.

Clone the current repository, inside the folder will contain the file [dl-env.yml](dl-env.yml), which describes the environment settings that will be used.

On the terminal, give the following command to create the environment:

```sh
$ conda env create --file dl-env.yml
```

After creating the environment, to activate it, use the command in the Linux terminal:
```sh
$ conda activate tcc-env-cpu
```

If you have configured the video card and want to use the GPU, run the first command (to create the environment), but using the [dl-env-gpu.yml](dl-env-gpu.yml) file. After that, to activate the environment:
```sh
$ conda activate tcc-env-gpu
```

_Note that the environment name can be changed in the [dl-env.yml](dl-env.yml) or [dl-env-gpu.yml](dl-env-gpu.yml) file in the first line._

### The functionality of scripts and how to execute them

[pre_processing.py](pre_processing.py): This file preprocesses the data that will be used on the network. It separates the csv data for each audio and obtains the spectrograms from them.

To run the script:
```sh
$ python pre_processing.py --csv path/to/file.csv --audios path/to/audios_folders
```

[model.py](model.py): This file is a class that contains the implementation of the convolutional network VGG19 presented by *Simonyan and Zisserman* in 2014, as well as MobileNet. Which network will be used should be informed in the Model class constructor.

[main.py](main.py): This script has the function to perform all tasks so far presented. It works in two ways: by calling the functions of [model](model.py) or combined with [preprocessing](pre_processing.py) to preprocess data and train the network all at once.

To run the [main.py](main.py) script only as training (data must be pre-processed):
```sh
$ python main.py --spectrograms /path/to/spectrograms_folder --mode training
```

To run the [main.py](main.py) script by combining preprocessing with training:
```sh
$ python main.py --mode all_process --csv path/to/file.csv --audios path/to/audios_folders
```

