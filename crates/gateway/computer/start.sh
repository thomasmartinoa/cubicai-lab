#!/bin/sh
# The only way out: 127.0.0.1:3128 here → the gateway's proxy socket (public websites only).
socat TCP-LISTEN:3128,bind=127.0.0.1,fork,reuseaddr UNIX-CONNECT:/run/cubic/proxy.sock &
exec python3 /opt/cubic/browserd.py
