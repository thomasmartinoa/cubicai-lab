# The Bots' computer: a locked-down Linux box with a terminal and a real browser. The gateway runs it
# with no network of its own (every connection goes through the gateway's public-only proxy), a
# read-only system, no capabilities, the user's own uid, and only the Bots' workspace mounted.
FROM python:3.14-slim

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PLAYWRIGHT_BROWSERS_PATH=/opt/browsers
RUN apt-get update \
 && apt-get install -y --no-install-recommends socat git curl wget jq ripgrep procps unzip zip file less nano ca-certificates fonts-noto-color-emoji fonts-dejavu-core \
 && pip install playwright==1.64.1 requests beautifulsoup4 \
 && playwright install --with-deps chromium \
 && rm -rf /var/lib/apt/lists/* /root/.cache \
 && chmod -R a+rX /opt/browsers

COPY start.sh browserd.py browse.py kill.sh /opt/cubic/
RUN chmod 755 /opt/cubic/*.sh /opt/cubic/*.py

WORKDIR /workspace
CMD ["/opt/cubic/start.sh"]
