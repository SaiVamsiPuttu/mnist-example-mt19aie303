FROM ubuntu:20.04
COPY mnist-example-mt19aie303 /assignment10/mnist-example-mt19aie303
COPY requirements.txt /assignment10/requirements.txt 
COPY Models /assignment10/Models
COPY Api /assignment10/Api
COPY dockerShell.sh /assignment10/dockerShell.sh
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r /assignment10/requirements.txt

WORKDIR /assignment10
CMD ["python3", "/assignment10/Api/helloworld.py"]

RUN chmod u+r+x /assignment10/dockerShell.sh
RUN /assignment10/dockerShell.sh