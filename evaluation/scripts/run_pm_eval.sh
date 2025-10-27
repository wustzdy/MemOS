#!/bin/bash

# Common parameters for all scripts
LIB="memos-api"
VERSION="072202"
WORKERS=10
TOPK=20

if [ "$LIB" = "mirix" ]; then
    echo "Running pm_mirix.py 100 times..."
    for i in {1..100}; do
        echo "Iteration $i/100"
        CUDA_VISIBLE_DEVICES=0 python scripts/personamem/pm_mirix.py --version $VERSION --workers 1
        if [ $? -ne 0 ]; then
            echo "Error running xx.py on iteration $i"
            exit 1
        fi
    done
elif ["$LIB" = "zep"]; then
    CUDA_VISIBLE_DEVICES=0 python scripts/personamem/pm_ingestion_zep.py --version $VERSION --workers $WORKERS
    CUDA_VISIBLE_DEVICES=0 python scripts/personamem/pm_search_zep.py --version $VERSION --top_k $TOPK --workers $WORKERS
    echo "Running pm_responses.py..."
    CUDA_VISIBLE_DEVICES=0 python scripts/personamem/pm_responses.py --lib $LIB --version $VERSION --workers $WORKERS
    if [ $? -ne 0 ]; then
        echo "Error running pm_responses.py"
        exit 1
    fi

    echo "Running pm_metric.py..."
    CUDA_VISIBLE_DEVICES=0 python scripts/personamem/pm_metric.py --lib $LIB --version $VERSION
    if [ $? -ne 0 ]; then
        echo "Error running pm_metric.py"
        exit 1
    fi
else
    echo "Running pm_ingestion.py..."
    CUDA_VISIBLE_DEVICES=0 python scripts/personamem/pm_ingestion.py --lib $LIB --version $VERSION --workers $WORKERS
    if [ $? -ne 0 ]; then
        echo "Error running pm_ingestion.py"
        exit 1
    fi

    echo "Running pm_search.py..."
    CUDA_VISIBLE_DEVICES=0 python scripts/personamem/pm_search.py --lib $LIB --version $VERSION --top_k $TOPK --workers $WORKERS
    if [ $? -ne 0 ]; then
        echo "Error running pm_search.py"
        exit 1
    fi

    echo "Running pm_responses.py..."
    CUDA_VISIBLE_DEVICES=0 python scripts/personamem/pm_responses.py --lib $LIB --version $VERSION --workers $WORKERS
    if [ $? -ne 0 ]; then
        echo "Error running pm_responses.py"
        exit 1
    fi

    echo "Running pm_metric.py..."
    CUDA_VISIBLE_DEVICES=0 python scripts/personamem/pm_metric.py --lib $LIB --version $VERSION
    if [ $? -ne 0 ]; then
        echo "Error running pm_metric.py"
        exit 1
    fi
fi

echo "All scripts completed successfully!"
