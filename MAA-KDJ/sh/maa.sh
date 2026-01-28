#!/bin/bash

# 并发运行 run_multi_gan.py 脚本，限制同时运行任务数量

DATA_DIR="../database/processed"
PYTHON_SCRIPT="../run_multi_gan.py"
MAX_JOBS=4

DEFAULT_START=31
DEFAULT_END=-1

job_count=0

for FILE in "$DATA_DIR"/*_processed.csv; do
    FILENAME=$(basename "$FILE")
    BASENAME="${FILENAME%.csv}"
    OUTPUT_DIR="../output/maa/${BASENAME}"

    START_TIMESTAMP=$DEFAULT_START
    END_TIMESTAMP=$DEFAULT_END

    # 🧷 可选：跳过已处理的数据集
    # if [ -d "$OUTPUT_DIR" ]; then
    #     echo "⚠️ 结果已存在，跳过：$BASENAME"
    #     continue
    # fi

    echo "🚀 启动任务：$FILENAME"

    python "$PYTHON_SCRIPT" \
        --data_path "$FILE" \
        --output_dir "$OUTPUT_DIR" \
        --feature_columns 1 21 1 21 1 21 \
        --start_timestamp "$START_TIMESTAMP" \
        --end_timestamp "$END_TIMESTAMP" \
        --N_pairs 3 \
        --distill_epochs 1 \
        --cross_finetune_epochs 5 \
        --backtrader True \
        --num_epochs 9999 \
        --patience 30 &

    ((job_count++))

    if (( job_count >= MAX_JOBS )); then
        wait -n
        ((job_count--))
    fi
done

wait
echo "✅ 所有任务完成！"
