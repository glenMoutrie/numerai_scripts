FROM python3.8
WORKDIR /numerai_scripts
RUN pip install -e .
CMD python run.py