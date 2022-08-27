TOTAL_TASKS=100
PREFIX=sample

# cd ..
EXP_LIST=./outputs/${PREFIX}*
IFS=$'\n' sorted=($(sort -r <<<"${EXP_LIST[*]}"))

if [ $# == 3 ]; then
    NODE_ALL=$1
    NODE_THIS=$2
    START_IDX=$3
fi
if [ $# == 2 ]; then
    NODE_ALL=$1
    NODE_THIS=$2
    START_IDX=0
fi

for ((i=$START_IDX;i<$TOTAL_TASKS;i++)); do
    NODE_TARGET=$(($i % $NODE_ALL))
    if [ $NODE_TARGET == $NODE_THIS ]; then
        exprdir="${sorted[i]}"
        echo $exprdir
        exprname=`basename ${exprdir}`
        if [ ! -f "${exprdir}/results.pt" ]; then
            python evaluation/evaluate.py --exp_name ${exprname}
        fi
    fi
done
