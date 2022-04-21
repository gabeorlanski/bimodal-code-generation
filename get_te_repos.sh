#!/bin/bash
cd data/
rm -rf repos
mkdir repos
cd repos

TO_GET=(
  "lxml,https://github.com/lxml/lxml/archive/refs/tags/lxml-4.8.0.zip"
  "scipy,https://github.com/scipy/scipy/archive/refs/tags/v1.8.0.zip"
  "numpy,https://github.com/numpy/numpy/archive/refs/tags/v1.22.3.zip"
  "sympy,https://github.com/sympy/sympy/archive/refs/tags/sympy-1.10.1.zip"
  "cerberus,https://github.com/pyeve/cerberus/archive/refs/tags/1.3.4.zip"
  "arrow,https://github.com/arrow-py/arrow/archive/refs/tags/1.2.2.zip"
  "delorean,https://github.com/myusuf3/delorean/archive/refs/tags/1.0.0.zip"
  "humanize,https://github.com/python-humanize/humanize/archive/refs/tags/4.0.0.zip"
  "bleach,https://github.com/mozilla/bleach/archive/refs/tags/v5.0.0.zip"
  "parsel,https://github.com/scrapy/parsel/archive/refs/tags/v1.6.0.zip"
  "natsort,https://github.com/SethMMorton/natsort/archive/refs/tags/8.1.0.zip"
  "networkx,https://github.com/networkx/networkx/archive/refs/tags/networkx-2.8.zip"
  "pint,https://github.com/hgrecco/pint/archive/refs/tags/0.19.1.zip"
  "pydash,https://github.com/dgilland/pydash/archive/refs/tags/v5.1.0.zip"
  "rapidjson,https://github.com/python-rapidjson/python-rapidjson/archive/refs/tags/v1.6.zip"
  "simplejson,https://github.com/simplejson/simplejson/archive/refs/tags/v3.17.6.zip"
  "simpy,https://gitlab.com/team-simpy/simpy/-/archive/4.0.1/simpy-4.0.1.zip"
  "theano,https://github.com/Theano/Theano/archive/refs/tags/rel-1.0.5.zip"
  "yarl,https://github.com/aio-libs/yarl/archive/refs/tags/v1.7.2.zip"
)

for i in ${TO_GET[@]};
do IFS=',';
  echo $i
  set -- $i
  URL=${2}
  NAME=${1}
  wget $URL
  ZIP_FILE=$(basename $URL)
  UNZIP_DIR=$(zipinfo -2 $ZIP_FILE | cut -d'/' -f 1 | head -n 1)
  unzip $ZIP_FILE
  mv $UNZIP_DIR $NAME
  rm -rf $ZIP_FILE
done


#git clone https://foss.heptapod.net/python-libs/passlib