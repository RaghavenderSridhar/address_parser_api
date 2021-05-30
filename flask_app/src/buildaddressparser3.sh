#!/usr/bin/env bash


cd libpostal
chmod -R 777 /address_parser_data
./bootstrap.sh
# mkdir -p /opt/address_parser_data/libpostal_data
./configure --datadir=/opt/address_parser_data/libpostal_data
sudo make
sudo make install

# On Linux it's probably a good idea to run
sudo ldconfig