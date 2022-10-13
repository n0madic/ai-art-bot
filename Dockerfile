FROM python:3

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -qq && \
    apt-get install -yqq --no-install-recommends libgl1-mesa-glx && \
    apt-get -yqq clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app

RUN pip install --prefer-binary --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache

COPY gfpgan /app/gfpgan
COPY realesrgan /app/realesrgan
COPY *.py /app/
COPY ideas.txt ignores.txt /app/

VOLUME /root/.cache/huggingface

CMD ["python", "app.py"]
