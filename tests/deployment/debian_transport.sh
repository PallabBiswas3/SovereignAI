#!/bin/sh
set -eu
sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources
apt-get -o Acquire::Retries=0 -o Acquire::https::Timeout=20 -o APT::Update::Error-Mode=any update
apt-get -o Acquire::Retries=0 -o Acquire::https::Timeout=20 download libgomp1
sha256sum libgomp1*.deb
