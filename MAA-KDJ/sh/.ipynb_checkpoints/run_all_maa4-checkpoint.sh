#!/bin/bash

# 运行脚本：run_multi_gan.py
# 遍历指定的CSV数据文件并设置对应参数

DATA_DIR="../database/processed4"
PYTHON_SCRIPT="../run_multi_gan.py"


# 默认的 start_timestamp
DEFAULT_START=31
DEFAULT_END=-1



for FILE in "$DATA_DIR"/*_processed.csv; do
    FILENAME=$(basename "$FILE")
    BASENAME="${FILENAME%.csv}"

    # 设置输出目录（可按需更换）
    OUTPUT_DIR="../output/maa/${BASENAME}"

    START_TIMESTAMP=$DEFAULT_START
    END_TIMESTAMP=$DEFAULT_END


    echo "Running $FILENAME with start=$START_TIMESTAMP..."

    python "$PYTHON_SCRIPT" \
        --data_path "$FILE" \
        --output_dir "$OUTPUT_DIR" \
        --feature_columns 1 21 1 21 1 21\
        --start_timestamp "$START_TIMESTAMP"\
        --end_timestamp "$END_TIMESTAMP" \
        --N_pairs 3 \
        --distill_epochs 1 \
        --cross_finetune_epochs 5 \
        --backtrader True\
        --num_epochs 9999\
        --patience 30
        
done

