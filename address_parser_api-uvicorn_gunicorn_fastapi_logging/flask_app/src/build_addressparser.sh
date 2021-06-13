#!/usr/bin/env bash

tar -xzvf addressParser_source.tar.gz

cd addressParser_source
./bootstrap.sh

mkdir -p /opt/address_parser_data

./configure --datadir=/opt/address_parser_data 
ldconfig

make
make install

