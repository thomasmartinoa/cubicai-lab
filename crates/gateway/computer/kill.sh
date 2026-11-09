#!/bin/sh
# Ends a command the gateway started (tagged cubic-<tag>) with everything it started.
for p in $(pgrep -f "cubic-$1"); do pkill -KILL -s "$p"; done
exit 0
