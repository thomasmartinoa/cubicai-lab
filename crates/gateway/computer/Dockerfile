# The Bots' computer: a locked-down Linux box with a terminal and a real browser. The gateway runs it
# with no network of its own (every connection goes through the gateway's public-only proxy), a
# read-only system, no capabilities, the user's own uid, and only the Bots' workspace mounted.
FROM python:3.14-slim

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PLAYWRIGHT_BROWSERS_PATH=/opt/browsers
RUN apt-get update \
 && apt-get install -y --no-install-recommends socat git curl wget jq ripgrep procps unzip zip xz-utils file less nano ca-certificates fonts-noto-color-emoji fonts-dejavu-core \
 && pip install playwright==1.64.1 requests beautifulsoup4 \
 && playwright install --with-deps chromium \
 && rm -rf /var/lib/apt/lists/* /root/.cache \
 && chmod -R a+rX /opt/browsers

# Node (for MCP servers started with npx; NODE_USE_ENV_PROXY sends its own requests through the proxy)
# and uv (uvx, for Python ones). Pinned and checked.
ARG NODE=v24.22.0 NODE_SHA=cc2f601cee82d9a5a233302fe307abc600727035fd7d2c3a3571755989e95008
RUN curl -fsSL -o /tmp/node.tar.xz https://nodejs.org/dist/$NODE/node-$NODE-linux-x64.tar.xz \
 && echo "$NODE_SHA  /tmp/node.tar.xz" | sha256sum -c - \
 && tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 --exclude='*/include' --exclude='*.md' --exclude='LICENSE' \
 && rm /tmp/node.tar.xz \
 && pip install uv==0.13.3 \
 && node --version && npx --version && uvx --version
ENV NODE_USE_ENV_PROXY=1 npm_config_update_notifier=false npm_config_fund=false UV_CACHE_DIR=/home/bot/.cache/uv

COPY start.sh browserd.py browse.py kill.sh /opt/cubic/
RUN chmod 755 /opt/cubic/*.sh /opt/cubic/*.py

WORKDIR /workspace
CMD ["/opt/cubic/start.sh"]
