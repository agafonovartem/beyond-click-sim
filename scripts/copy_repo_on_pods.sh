#!/bin/bash

zip -r ../beyond-click-sim.zip . -x ".venv/*" "outputs/*" "*.zip" ".git/*"

echo "Copying repo to pods..."

kubectl cp ../beyond-click-sim.zip agent-exp-0:/home/jovyan/beyond-click-sim.zip &
kubectl cp ../beyond-click-sim.zip agent-exp-1:/home/jovyan/beyond-click-sim.zip &
kubectl cp ../beyond-click-sim.zip agent-exp-2:/home/jovyan/beyond-click-sim.zip &
kubectl cp ../beyond-click-sim.zip agent-exp-3:/home/jovyan/beyond-click-sim.zip &
kubectl cp ../beyond-click-sim.zip agent-exp-4:/home/jovyan/beyond-click-sim.zip &
kubectl cp ../beyond-click-sim.zip agent-exp-5:/home/jovyan/beyond-click-sim.zip &
kubectl cp ../beyond-click-sim.zip agent-exp-6:/home/jovyan/beyond-click-sim.zip &
kubectl cp ../beyond-click-sim.zip agent-exp-7:/home/jovyan/beyond-click-sim.zip &
wait