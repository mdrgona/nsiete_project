Create docker image with following command (be in this directory, where Dockerfile is located):
```docker build -t ns-project .```

To start docker image run:
```docker run -p 8888:8888 -v C:\_data\skola\FIITSTU\ING\1_ISS\NSIETE\project\:/labs -it ns-project```