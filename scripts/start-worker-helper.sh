#!/usr/bin/env bash
# Start MoLink Worker with proper log capture
cd /opt/molink/app
logfile="/opt/molink/app/logs/worker.log"
nohup bash /opt/molink/app/worker/start_worker.sh > "$logfile" 2>&1 &
echo "Worker PID: $!"
