# We will use Ubuntu for our image
FROM ubuntu

# Updating Ubuntu packages
RUN apt-get update && yes|apt-get upgrade

# Adding wget and bzip2
RUN apt-get install -y wget bzip2

# Anaconda installing
RUN wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
RUN bash Anaconda3-5.0.1-Linux-x86_64.sh -b
RUN rm Anaconda3-5.0.1-Linux-x86_64.sh

# Set path to conda
ENV PATH /root/anaconda3/bin:$PATH

# Updating Anaconda packages
RUN conda update conda
RUN conda update anaconda
RUN conda update --all

# Bug fix
RUN pip uninstall notebook -y && pip install notebook==5.6.0

# Additional packages
RUN pip install tensorflow 
RUN apt-get install -y ffmpeg
RUN pip install librosa
RUN pip install jupyterthemes
RUN jt -t onedork  -cellw 88%

#next for text alignment
RUN apt-get install espeak -y 
RUN apt-get install ffmpeg -y
#RUN apt-get install ffprobe -y
#RUN apt-get install gcc -y
#RUN pip install aeneas

# Configuring access to Jupyter
RUN mkdir /opt/notebooks
# RUN jupyter notebook --generate-config --allow-root
# RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /root/.jupyter/jupyter_notebook_config.py
COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

# Jupyter listens port: 8888
EXPOSE 8888
# Run Jupytewr notebook as Docker main process
CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/opt/notebooks", "--ip='*'", "--port=8888", "--no-browser"]
