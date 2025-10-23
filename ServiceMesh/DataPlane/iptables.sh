#!/bin/bash
PROXY_PORT=${PROXY_PORT:-8080}

# 기존 규칙 초기화
iptables -t nat -F OUTPUT
iptables -t nat -F PREROUTING
# iptables -t mangle -F PREROUTING
# iptables -t mangle -F OUTPUT

# 프록시 트래픽 마킹
# iptables -t mangle -A OUTPUT -m owner --uid-owner proxyuser -j MARK --set-mark 1
# iptables -t mangle -A PREROUTING -m mark --mark 1 -j MARK --set-mark 1

# conntrack 모듈을 사용하여 관련된 연결 추적
# iptables -t nat -A PREROUTING -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
# iptables -t nat -A PREROUTING -m mark --mark 1 -j RETURN
iptables -t nat -A PREROUTING -p tcp --dport 9011 -j REDIRECT --to-port 9011
iptables -t nat -A PREROUTING -p tcp ! --dport $PROXY_PORT -j REDIRECT --to-port $PROXY_PORT

# OUTPUT 체인 규칙
iptables -t nat -I OUTPUT -m owner --uid-owner proxyuser -j RETURN
iptables -t nat -I OUTPUT -s 127.0.0.1/32 -j RETURN
# iptables -t nat -I OUTPUT -p udp --dport 53 -j RETURN
iptables -t nat -A OUTPUT -p tcp ! --dport $PROXY_PORT -j REDIRECT --to-port $PROXY_PORT

# 연결 추적 설정 조정
# sysctl -w net.netfilter.nf_conntrack_tcp_loose=0