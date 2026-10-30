#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-}"
mkdir -p /opt/hadoop/logs /opt/hadoop/pids /opt/hadoop/data/yarn/local /opt/hadoop/data/yarn/logs

wait_for_hdfs() {
  echo "Waiting for HDFS NameNode at master:9000 ..."
  for attempt in $(seq 1 90); do
    if hdfs dfs -ls / >/dev/null 2>&1; then
      echo "HDFS is responding."
      return 0
    fi
    sleep 2
  done
  echo "ERROR: HDFS did not become ready within 180 seconds." >&2
  echo "Check the master logs with: docker logs master" >&2
  exit 1
}

wait_for_datanodes() {
  echo "Waiting for both HDFS DataNodes..."
  for attempt in $(seq 1 90); do
    if hdfs dfsadmin -report 2>/dev/null | grep -q 'Live datanodes (2)'; then
      echo "Both HDFS DataNodes are live."
      return 0
    fi
    sleep 2
  done
  echo "ERROR: two HDFS DataNodes did not register within 180 seconds." >&2
  exit 1
}

case "$ROLE" in
  master)
    if [[ ! -f /opt/hadoop/data/namenode/current/VERSION ]]; then
      echo "Formatting HDFS NameNode for the first time (only if the data volume is empty)..."
      hdfs namenode -format -nonInteractive
    fi
    hdfs --daemon start namenode
    wait_for_hdfs
    hdfs dfs -mkdir -p /tmp /spark-jars /spark-logs /user/root/flights/raw
    hdfs dfs -chmod 1777 /tmp
    # spark.yarn.jars points at this HDFS directory. A brand-new named volume is
    # empty, so populate it once; otherwise YARN submissions cannot distribute
    # Spark itself to the worker containers reproducibly.
    wait_for_datanodes
    if ! hdfs dfs -test -e '/spark-jars/spark-core_*.jar'; then
      echo "Uploading Spark runtime jars to HDFS (first cluster start only)..."
      hdfs dfs -put -f "$SPARK_HOME"/jars/*.jar /spark-jars/
    fi
    yarn --daemon start resourcemanager
    "$SPARK_HOME/sbin/start-history-server.sh"
    echo "Master started: NameNode, ResourceManager, Spark History Server."
    echo "Open http://localhost:9870, http://localhost:8088, and http://localhost:18080"
    exec tail -f /dev/null
    ;;
  worker)
    wait_for_hdfs
    hdfs --daemon start datanode
    yarn --daemon start nodemanager
    echo "Worker started: DataNode and NodeManager."
    exec tail -f /dev/null
    ;;
  *)
    echo "Usage: bda-entrypoint.sh master|worker" >&2
    exit 2
    ;;
esac
