FROM yoongunwo/ebpf:1.1

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-setuptools \
    net-tools \
    procps \
    linux-headers-5.4.0-200-generic \
    iptables-persistent \
    sudo

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./reverseProxyServer.py ./reverseProxyServer.py
COPY ./eBPF ./eBPF
COPY ./Detect/ ./Detect
COPY ./Proxy ./Proxy
COPY ./iptables.sh .
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# 새 사용자 생성 및 sudo 권한 부여 
RUN useradd -m -s /bin/bash -u 5555 proxyuser && \
    echo "proxyuser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# 앱 디렉토리 권한 설정
RUN chown -R proxyuser:proxyuser /app

# 실행 권한 부여
RUN chmod +x /app/iptables.sh

# model folder
RUN mkdir /app/model
RUN chmod -R 777 /app/model

# ENTRYPOINT ["/bin/sh", "-c", "/app/iptables.sh && while true; do sleep 1000; done"]
ENTRYPOINT ["/bin/sh", "-c", "cd /app && /app/iptables.sh && python3 /app/reverseProxyServer.py"]