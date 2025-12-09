1- Set proxy to download ml_dtypes and activate conda

```
export no_proxy=127.0.0.1,localhost,mirrors.tools.huawei.com,mirrors.myhuaweicloud.com,$no_proxy
function myproxy () {
    read -p "DE or HK or CN proxy [de/hk/cn]: " DE_or_CN
    if [ "$DE_or_CN" == "de" ]; then
        read -p "Huawei username (ex: a00xxxxxx): " USER
        read -s -p "Password: " PW
        PW_ENC=`python3 -c "import urllib.parse;print(urllib.parse.quote(input()))" <<< "$PW"`
        PROXY="$USER:$PW_ENC@proxyde.huawei.com:8080"
    elif [ "$DE_or_CN" == "hk" ]; then
        read -p "Huawei username (ex: a00xxxxxx): " USER
        read -s -p "Password: " PW
        PW_ENC=`python3 -c "import urllib.parse;print(urllib.parse.quote(input()))" <<< "$PW"`
        PROXY="$USER:$PW_ENC@proxyhk.huawei.com:8080"
    elif [ "$DE_or_CN" == "cn" ]; then
        PROXY="proxy.huawei.com:8080";
    else
        echo "invalid option."
    fi
    export HTTPS_PROXY=http://$PROXY;export HTTP_PROXY=http://$PROXY;export http_proxy=http://$PROXY;export Proxy=$http_proxy;export https_proxy=http://$PROXY;export ftp_proxy=ftp://$PROXY
}


myproxy

# Enter your username and pw

source $CONDA_HOME/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

clear
```

2- Install dependencies
```
cd /workdir/tree-mla-ascend/catlass_mla/catlass_standalone/mla
export CATLASS_DIR=/workdir/tree-mla-ascend/catlass_mla/ref_catlass/catlass
bash install.sh
```

3- Run code

```
python  paged_attention_test.py --bench
python  paged_attention_test.py --test

# args: batchSize, qSeqlen, kvSeqlen, qheadNum, numBlock, blockSize
python tests/gen_data.py 1 1 128 16 16 128 half  # default
python tests/test_attention_fromdata.py | tee run_test_attention_fromdata_default.log

python tests/gen_data.py 1 1 128 128 16 128 half  # qheadNum = 128
python tests/test_attention_fromdata.py | tee run_test_attention_fromdata_qheadNum=128.log

python tests/gen_data.py 4 1 128 16 16 128 half  # batchSize=4
python tests/test_attention_fromdata.py | tee run_test_attention_fromdata_batchSize=4.log

python tests/gen_data.py 1 1 1024 16 16 128 half  # kvSeqlen=1024
python tests/test_attention_fromdata.py | tee run_test_attention_fromdata_kvSeqlen=1024.log
```