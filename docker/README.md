Save a local docker image to `.tar`

```docker
docker save -o <path for generated tar file> <image name>
```

Send the `.tar` file to the other host and load it by

```docker
docker load -i <path to image tar file>
```


```docker
docker run -v /mnt/ZOD:/mnt/ZOD <image name>
```