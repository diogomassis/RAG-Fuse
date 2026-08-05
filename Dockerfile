FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

RUN python3 -m nltk.downloader -d /usr/local/nltk_data punkt punkt_tab
ENV NLTK_DATA=/usr/local/nltk_data

CMD ["bash"]
