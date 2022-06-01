TOTAL_TASKS=100

if [ $# != 3 ]; then
    echo "Error: 2 arguments required."
    exit 1
fi

NODE_ALL=$1
NODE_THIS=$2
START_IDX=$3
# echo $NODE_ALL $NODE_THIS $START_IDX

for ((i=$START_IDX;i<$TOTAL_TASKS;i++)); do
    NODE_TARGET=$(($i % $NODE_ALL))
    if [ $NODE_TARGET == $NODE_THIS ]; then
        echo "Task ${i} assigned to this worker (${NODE_THIS})"
        python -m sample --data_id ${i}
    fi
done

echo "Finished node: $NODE_THIS"
