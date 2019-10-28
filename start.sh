#!/bin/bash
conda activate homeassistant
sudo motion -m
sudo mosquitto  -c /etc/mosquitto/mosquitto.conf &
sudo hass -c . --open-ui
