#!/usr/bin/env bash
set -euo pipefail

# 参数（可通过环境变量传入）
SF=${SF:-1}
LDBC_SNB_DATAGEN_MAX_MEM=${LDBC_SNB_DATAGEN_MAX_MEM:-16g}
BASE_DIR=${BASE_DIR:-/home/ylivm/ngdb}
SPARK_HOME=${SPARK_HOME:-"${HOME}/spark-3.2.2-bin-hadoop3.2"}
NEO4J_PAGECACHE=${NEO4J_PAGECACHE:-20G}
NEO4J_HEAP_MAX=${NEO4J_HEAP_MAX:-20G}
FORCE_REGENERATE=${FORCE_REGENERATE:-0}
SKIP_LOAD=${SKIP_LOAD:-0}
NEO4J_CONTAINER_NAME=${NEO4J_CONTAINER_NAME}

export SPARK_HOME
export PATH="${SPARK_HOME}/bin:${PATH}"

DATAGEN_DIR="${BASE_DIR}/ldbc_snb_datagen_spark"
BI_DIR="${BASE_DIR}/ldbc_snb_bi"

mkdir -p "${BASE_DIR}"

# 1) 确保仓库就位
if [ ! -d "${DATAGEN_DIR}" ]; then
  git clone https://github.com/ldbc/ldbc_snb_datagen_spark.git "${DATAGEN_DIR}"
fi
if [ ! -d "${BI_DIR}" ]; then
  git clone https://github.com/ldbc/ldbc_snb_bi.git "${BI_DIR}"
fi

# 2) 构建 datagen（若必要）
pushd "${DATAGEN_DIR}" >/dev/null
if ! sbt -batch -error 'print assembly / assemblyOutputPath' >/tmp/datagen_jar_path.txt 2>/dev/null; then
  sbt -batch assembly
  sbt -batch -error 'print assembly / assemblyOutputPath' >/tmp/datagen_jar_path.txt
fi

# 3) 生成数据（BI 模式，CSV + gzip），若已存在且未强制则跳过
OUT_DIR="out-sf${SF}"
if [ -d "${OUT_DIR}" ] && [ -n "$(ls -A "${OUT_DIR}" 2>/dev/null || true)" ] && [ "${FORCE_REGENERATE}" != "1" ]; then
  echo "检测到 ${OUT_DIR} 已存在且非空，跳过数据生成（设置 FORCE_REGENERATE=1 可强制重跑）。"
else
  rm -rf "${OUT_DIR}" || true
  # 安装 tools 依赖（如需要）
  if [ -f tools/requirements.txt ]; then
    python3 -m pip install -r tools/requirements.txt || true
  fi
  export LDBC_SNB_DATAGEN_MAX_MEM
  python3 tools/run.py \
    --cores "$(nproc)" \
    --memory "${LDBC_SNB_DATAGEN_MAX_MEM}" \
    -- \
    --format csv \
    --scale-factor "${SF}" \
    --explode-edges \
    --mode bi \
    --output-dir "${OUT_DIR}/" \
    --format-options header=false,quoteAll=true,compression=gzip
fi
popd >/dev/null

# 4) 使用 BI 脚本装载到 Neo4j（可通过 SKIP_LOAD=1 跳过）
if [ "${SKIP_LOAD}" = "1" ]; then
  echo "已设置 SKIP_LOAD=1，跳过 Neo4j 装载步骤。"
else
  export LDBC_SNB_DATAGEN_DIR="${DATAGEN_DIR}"
  export SF
  export NEO4J_ENV_VARS="${NEO4J_ENV_VARS-} --env NEO4J_dbms_memory_pagecache_size=${NEO4J_PAGECACHE} --env NEO4J_dbms_memory_heap_max__size=${NEO4J_HEAP_MAX}"
  
  pushd "${BI_DIR}/neo4j" >/dev/null
  bash "../scripts/use-datagen-data-set.sh"
  bash "../scripts/load-in-one-step.sh"
  popd >/dev/null
fi

echo "ldbc_snb_bi 数据生成与装载完成 (SF=${SF})."

