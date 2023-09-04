FROM public.ecr.aws/docker/library/python:3.8

# copy assets over to image
COPY /src /defect_classify

# set the working directory
WORKDIR /defect_classify

# install dependancies
RUN pip3 install --user --no-cache-dir -r requirements.txt
#RUN conda install --yes --file environment.yml

# set PATH
ENV PATH=.local/bin:$PATH

# exposing endpoint port
EXPOSE 5001

ENTRYPOINT ["python", "src/serve.py"]