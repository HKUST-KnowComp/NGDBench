LDBC_SNB_DATA_DIR = ${LDBC_SNB_DATA_DIR}
NEO4J_CONTAINER_NAME = ${NEO4J_CONTAINER_NAME}
SF=${SF:-1}
export NEO4J_CSV_DIR = ${LDBC_SNB_DATA_DIR}/out-sf${SF}/graphs/csv/bi/composite-projected-fk/
export NEO4J_ENV_VARS = "${NEO4J_ENV_VARS-} --env NEO4J_dbms_memory_pagecache_size=${NEO4J_PAGECACHE} --env NEO4J_dbms_memory_heap_max__size=${NEO4J_HEAP_MAX}"

set -eu
set -o pipefail

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ..

. scripts/vars.sh

echo "==============================================================================="
echo "Loading the Neo4j database"
echo "-------------------------------------------------------------------------------"
echo "SF: ${SF}"
echo "NEO4J_CONTAINER_ROOT: ${NEO4J_CONTAINER_ROOT}"
echo "NEO4J_VERSION: ${NEO4J_VERSION}"
echo "NEO4J_CONTAINER_NAME: ${NEO4J_CONTAINER_NAME}"
echo "NEO4J_ENV_VARS: ${NEO4J_ENV_VARS}"
echo "NEO4J_DATA_DIR (on the host machine):"
echo "  ${NEO4J_DATA_DIR}"
echo "NEO4J_CSV_DIR (on the host machine):"
echo "  ${NEO4J_CSV_DIR}"
echo "==============================================================================="

if [ "$(uname)" == "Darwin" ]; then
    DATE_COMMAND=gdate
else
    DATE_COMMAND=date
fi

scripts/stop.sh
scripts/delete-database.sh

start_time=$(${DATE_COMMAND} +%s.%3N)

scripts/import.sh
scripts/start.sh
scripts/create-indices.sh

end_time=$(${DATE_COMMAND} +%s.%3N)

mkdir -p output/output-sf${SF}
elapsed=$(python3 -c "import argparse; parser = argparse.ArgumentParser(); parser.add_argument('--start_time', type=float); parser.add_argument('--end_time', type=float); args = parser.parse_args(); elapsed = args.end_time - args.start_time; print(f'{elapsed:.3f}')" --start_time $start_time --end_time $end_time)
echo -e "time\n${elapsed}" > output/output-sf${SF}/load.csv