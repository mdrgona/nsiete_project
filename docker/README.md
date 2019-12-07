Create docker image with following command (be in this directory, where Dockerfile is located):
```docker build -t ns-project .```

To start docker image in **windows** run:
```docker run -p 8888:8888 -p 6006:6006 -v C:\_data\skola\FIITSTU\ING\1_ISS\NSIETE\project\:/labs -it ns-project```

To start docker image in **linux based system** run:
```docker run -u $(id -u):$(id -g) -p 8888:8888 -p 6006:6006 -v $(pwd):/labs -it ns-project```
