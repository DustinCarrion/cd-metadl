# Set up the environment with Docker

If you are new to docker, install the docker engine from [https://docs.docker.com/get-started/](https://docs.docker.com/get-started/). Make sure that your system is compatible with the version that you install. Once you have installed docker, you can pull the docker image that we provide for this challenge. This image contains all required packages and the Python environment for the competition. You can use this image to perform local tests on your machine.
The image can be obtained by running
```bash
docker pull dustrix397/cdmetadl:gpu
```

This image can then be used to create a running container using the following command:
```bash
docker run --name cdmetadl -it -v "[path_to_root_directory_of_this_repository]:/app/codalab" -p 8888:8888 dustrix397/cdmetadl:gpu
```
You have to substitute the `[path_to_root_directory_of_this_repository]` according to your machine, for example 
```bash
docker run --name cdmetadl -it -v "/home/user/cd-metadl:/app/codalab" -p 8888:8888 dustrix397/cdmetadl:gpu
```

The option `-v "[path_to_root_directory_of_this_repository]:/app/codalab"` mounts root directory (*i.e.*, `cd-metadl/`) as `/app/codalab`.
If you want to mount other directories on your disk, please replace `[path_to_root_directory_of_this_repository]` by your own directory. The option `-p 8888:8888` is useful for running a Jupyter notebook tutorial inside Docker.
This container environment can be exited at any time by pressing `Ctrl+D` (on Linux) or by typing exit.

The Docker image has python=3.8.10. If you want to run local test with Nvidia GPU support, please make sure you have installed `nvidia-docker` and run instead:
```bash
nvidia-docker run --name cdmetadl -it -v "[path_to_root_directory_of_this_repository]:/app/codalab" -p 8888:8888 dustrix397/cdmetadl:gpu
```
Make sure you use enough RAM (at least 4GB). If the port 8888 is occupied, you can use other ports, e.g. 8899, and use instead the option `-p 8899:8888`.

## Run the tutorial notebook
We provide a tutorial in the form of a Jupyter notebook. When you are in your docker container, enter:

```bash
jupyter-notebook --ip=0.0.0.0 --allow-root &
```
Then copy and paste the URL containing your token. It should look like something like that:
```bash
http://0.0.0.0:8888/?token=5cc8ad3dda2366b7b426bf84afe72d614fa79a2c0109fafc
```

Or
```bash
http://127.0.0.1:8888/?token=5cc8ad3dda2366b7b426bf84afe72d614fa79a2c0109fafc
```

You will access the Jupyter menu, click on `tutorial.ipynb` and you are all set.