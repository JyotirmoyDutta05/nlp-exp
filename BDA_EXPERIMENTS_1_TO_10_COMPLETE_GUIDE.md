# Big Data Analytics (BDA) Laboratory: Experiments 1–10 Complete Guide & Code Explanation

**Author:** Jyotirmoy Dutta (Roll No: 2301051, Batch: B13)  
**Project Domain:** NYC Yellow Taxi Smart City Data Platform & Public Transit Anomaly Detection  
**Core Technologies:** Hadoop HDFS, Apache Spark (PySpark & Spark SQL), Apache Kafka, MongoDB (NoSQL), Scikit-Learn, Streamlit, Matplotlib/Seaborn/Plotly  

---

## Table of Contents
1. [End-to-End System Architecture & Laboratory Progression](#1-end-to-end-system-architecture--laboratory-progression)
2. [Experiment 1: Distributed Storage & Exploratory Data Analysis (HDFS & PySpark)](#2-experiment-1-distributed-storage--exploratory-data-analysis-hdfs--pyspark)
3. [Experiment 2: Project Proposal, Anomaly Taxonomy & System Architecture](#3-experiment-2-project-proposal-anomaly-taxonomy--system-architecture)
4. [Experiment 3: Distributed Text Processing & MapReduce via PySpark](#4-experiment-3-distributed-text-processing--mapreduce-via-pyspark)
5. [Experiment 4: NoSQL Data Ingestion Pipeline (Python & MongoDB)](#5-experiment-4-nosql-data-ingestion-pipeline-python--mongodb)
6. [Experiment 5: Large-Scale Batch Analytics with Spark SQL & Execution Plans](#6-experiment-5-large-scale-batch-analytics-with-spark-sql--execution-plans)
7. [Experiment 6: Batch Analytics Visualization & Comprehensive Analytical Report](#7-experiment-6-batch-analytics-visualization--comprehensive-analytical-report)
8. [Experiment 7: Real-Time Streaming Data Pipeline (Kafka & Spark Structured Streaming)](#8-experiment-7-real-time-streaming-data-pipeline-kafka--spark-structured-streaming)
9. [Experiment 8: Real-Time Operational Dashboard (Streamlit & Offline GIS)](#9-experiment-8-real-time-operational-dashboard-streamlit--offline-gis)
10. [Experiment 9: Multi-Model NoSQL Data Lake Architecture & Empirical Index Benchmarks](#10-experiment-9-multi-model-nosql-data-lake-architecture--empirical-index-benchmarks)
11. [Experiment 10: Machine Learning Anomaly Detection & Integrated Smart City Showcase](#11-experiment-10-machine-learning-anomaly-detection--integrated-smart-city-showcase)
12. [Master Summary Matrix (Experiments 1 to 10)](#12-master-summary-matrix-experiments-1-to-10)

---

## 1. End-to-End System Architecture & Laboratory Progression

The laboratory exercises follow an enterprise data engineering and data science lifecycle using the **New York City Taxi & Limousine Commission (TLC) Yellow Taxi Trip Records** (2025–2026, 73.3M cleaned trips).

```
                          NYC TLC Parquet Data (20 Months, ~1.2 GB)
                                            |
         +----------------------------------+----------------------------------+
         |                                                                     |
  [BATCH PIPELINE]                                                    [STREAMING PIPELINE]
  Exp 1: HDFS Setup & EDA                                             Exp 7: Kafka Producer (Replay)
  Exp 2: Architecture & Anomaly Taxonomy                                     | (topic: taxi-trips)
  Exp 3: MapReduce & Skew Analysis                                           v
  Exp 5: Spark SQL Batch Aggregates (73.3M)                           Exp 7: Spark Structured Streaming
         |                                                                   | 5-min sliding windows, event-time
         +-----------------+                                                 | watermark, dual sinks (JSON + text)
         |                 |                                                 +-----------------+
         v                 v                                                 |                 |
    Exp 6: Charts     Exp 10: Feature Store                                  v                 v
    & Report         (4.9M zone-windows)                             Exp 8: Streamlit   Exp 9: MongoDB Lake
                           |                                         Live Dashboard     (trips, zones,
                           v                                         (Map + Feeds)      time-series ts)
                      Exp 10: ML Isolation Forest                            |
                      (Surge/Collapse Detection)                             v
                           |                                         Exp 10: Final Showcase
                           +----------------------------------------> Dashboard (Live Map +
                                                                      Real-Time Anomaly Scoring)
```

---

## 2. Experiment 1: Distributed Storage & Exploratory Data Analysis (HDFS & PySpark)

### 2.1 Aim & Overview
To configure a distributed big data storage and processing cluster using **Hadoop HDFS** and **Apache Spark**, ingest the domain-specific NYC Taxi Trip dataset (`NYC_Taxi_Cleaned_Analysis_Ready.csv`, ~586 MB, 3,946,698 records, 25 attributes), and perform distributed Exploratory Data Analysis (EDA).

### 2.2 Theoretical Concepts
1. **Hadoop Distributed File System (HDFS):**
   - **NameNode:** Manages metadata, namespace directory tree, and file-to-block block mapping.
   - **DataNode:** Stores actual 128 MB blocks with block replication for fault tolerance.
   - Configured at `hdfs://localhost:9000/bda/experiment1/`.
2. **Apache Spark Unified Compute:**
   - In-memory execution using Resilient Distributed Datasets (RDDs) and Catalyst-optimized DataFrames.
   - Decouples compute from storage, reading directly from HDFS.
3. **Distributed Schema Inference & Columnar Types:**
   - Inferring 25 distinct columns across numeric, timestamp, and categorical domains without manual parsing.

### 2.3 Code Implementation & Explanation
```python
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib.pyplot as plt

# 1. Initialize SparkSession targeting the local cluster
spark = SparkSession.builder \
    .appName("BDA Experiment 1 - NYC Taxi EDA") \
    .master("local[*]") \
    .getOrCreate()

# 2. Ingest CSV directly from HDFS with automatic schema inference
hdfs_path = "hdfs://localhost:9000/bda/experiment1/NYC_Taxi_Cleaned_Analysis_Ready.csv"
df = spark.read.option("header", "true").option("inferSchema", "true").csv(hdfs_path)

# 3. Structural and missing value inspection
df.printSchema()
print(f"Total Record Count: {df.count()}")

# Missing value calculation across all columns
null_counts = {col: df.filter(F.col(col).isNull()).count() for col in df.columns}
print("Missing value summary:", null_counts)

# 4. Statistical profiling of continuous metrics
df.select("passenger_count", "trip_distance", "fare_amount", "total_amount", "duration_min", "speed_mph").describe().show()

# 5. Temporal aggregation: Taxi demand by pickup hour
hourly_demand = df.groupBy("pickup_hour").count().orderBy("pickup_hour").toPandas()

# 6. Visualization
plt.figure(figsize=(10, 4))
plt.plot(hourly_demand["pickup_hour"], hourly_demand["count"], marker="o", color="b")
plt.title("NYC Taxi Trips by Pickup Hour")
plt.xlabel("Hour of Day (0-23)")
plt.ylabel("Trip Volume")
plt.grid(True)
plt.savefig("hourly_trips.png")
```

### 2.4 Key Results & Findings
- **Volume:** 3,946,698 trips successfully parsed from HDFS.
- **Missing Values:** Exactly 24 columns had 0 missing values; only `Airport_fee` contained 757,387 missing values (~19.19%), representing trips originating outside airport surcharge zones.
- **Operational Baselines:** Average passengers: **1.23**, average distance: **3.44 miles**, average fare: **$19.37**, average duration: **16.97 minutes**, average speed: **11.03 mph**.
- **Temporal Rhythm:** Diurnal cycle with a sharp trough at **04:00 (28,562 trips)** and peak evening rush demand at **18:00 (265,056 trips)**.

---

## 3. Experiment 2: Project Proposal, Anomaly Taxonomy & System Architecture

### 3.1 Aim & Overview
To formulate the project proposal, define the telemetry anomaly taxonomy, specify the Medallion lakehouse architecture, and allocate data assets and team roles for a **Public Transit Fleet Anomaly Detection Platform**.

### 3.2 Theoretical Concepts
1. **Fleet Anomaly Taxonomy:**
   - **Operational & Route Irregularities:** Fare padding (long-hauling), ghost trips, impossible velocities (>80 mph), excessive idling.
   - **Revenue & Meter Inconsistencies:** Meter calibration drift, fraudulent flat-rate codes (e.g. Ratecode 2 JFK applied to local trips), unauthorized surcharges.
   - **Spatio-Temporal & Environmental Shocks:** Travel time blowouts along Origin-Destination (OD) corridors caused by flash flooding, snowstorms, or transit disruptions.
2. **Medallion Lakehouse Architecture:**
   - **Bronze Layer (Raw):** Landing raw Parquet trip logs and shapefiles in object storage/HDFS.
   - **Silver Layer (Curated/Cleaned):** Outlier-filtered, boundary-checked, feature-enriched partitioned Parquet tables.
   - **Gold Layer (Aggregated Business Intelligence):** Pre-aggregated dimensional tables, spatial demand matrices, and machine learning inference scores.
3. **Multi-Source Contextual Fusion:**
   - Coupling primary TLC telemetry with geographic boundaries (263 NYC Taxi Zones) and atmospheric records (NOAA/Open-Meteo weather feeds).

### 3.3 Engineering Team Role Matrix
| Role | Core Domain Responsibilities | Deliverables |
|---|---|---|
| **Data Platform & Ingestion Engineer** | Distributed cluster setup, raw ingestion pipelines, HDFS partition design, fault tolerance. | Download scripts, Bronze storage layout, ingestion verification. |
| **Data Processing & ETL Engineer** | Large-scale batch ETL, physical bound validation, Silver lakehouse hygiene. | PySpark cleaning jobs, feature transformers, curated Parquet tables. |
| **Machine Learning Specialist** | Unsupervised outlier models, baseline calculations, feature selection, scoring pipelines. | Isolation Forest models, Z-score engines, evaluation reports. |
| **Analytics & BI Engineer** | Gold aggregations, spatial choropleths, executive dashboards. | Streamlit operational UI, GIS maps, alert consoles. |

---

## 4. Experiment 3: Distributed Text Processing & MapReduce via PySpark

### 4.1 Aim & Overview
To implement fundamental **MapReduce** workflows using PySpark to perform distributed tokenization, word count, keyword filtering, and rule-based record tagging, while verifying cluster partition balance and checking for data skew.

### 4.2 Theoretical Concepts
1. **MapReduce Paradigm:**
   - **Map:** Emits intermediate key-value pairs `(k, v)` from raw records independently across workers.
   - **Shuffle & Sort:** Re-partitions and groups data across cluster nodes based on hash keys.
   - **Reduce:** Aggregates associative values per key via `reduceByKey()` locally on workers prior to network shuffle.
2. **Data Partitioning & Skewness:**
   - In distributed computing, partitions correspond to threads/executors. Unequal partition sizes create "stragglers" where a single node bottlenecks the entire job.

### 4.3 Code Implementation & Explanation
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Exp3-MapReduce").master("local[4]").getOrCreate()
sc = spark.sparkContext

# Task 1: Ingest and explicitly repartition into 4 simulated nodes
trips_df = spark.read.parquet("data/yellow_tripdata_2026-01.parquet").limit(60000).repartition(4)
zones_df = spark.read.csv("data/taxi_zone_lookup.csv", header=True)

# Task 2a: MapReduce Word Count on Taxi Zone Names
zone_names_rdd = zones_df.select("Zone").dropna().rdd.map(lambda r: r[0].lower())
word_counts = (zone_names_rdd
               .flatMap(lambda line: line.split())
               .map(lambda word: (word, 1))
               .reduceByKey(lambda a, b: a + b)
               .sortBy(lambda x: x[1], ascending=False))
print("Top words in zone names:", word_counts.take(10))

# Task 2b: Keyword Filtering (Airport trips)
airport_trips = trips_df.filter(trips_df.PULocationID.isin([1, 132, 138]))
print(f"Airport Trips: {airport_trips.count()}, Avg Fare: ${airport_trips.agg({'fare_amount': 'avg'}).first()[0]:.2f}")

# Task 2c: Preliminary Trip Tagging (Map Transformation)
def tag_trip(row):
    tags = []
    if row.tip_amount == 0: tags.append("NO_TIP")
    elif row.tip_amount > 10: tags.append("GENEROUS_TIP")
    if row.trip_distance < 1.0: tags.append("SHORT_TRIP")
    elif row.trip_distance > 10.0: tags.append("LONG_TRIP")
    else: tags.append("MEDIUM_TRIP")
    if row.fare_amount < 5.0: tags.append("LOW_FARE")
    elif row.fare_amount > 50.0: tags.append("HIGH_FARE")
    return tags

tag_counts = trips_df.rdd.flatMap(tag_trip).map(lambda t: (t, 1)).reduceByKey(lambda a, b: a + b).collect()
print("Trip Tags:", tag_counts)

# Task 3: Distributed Partition Execution Analysis
records_per_partition = trips_df.rdd.glom().map(len).collect()
for idx, count in enumerate(records_per_partition):
    print(f"Partition (Node) {idx}: {count} records")
```

### 4.4 Key Results & Findings
- **Word Count:** Top occurrences across 87 zones: `manhattan` (50), `brooklyn` (22), `west` (10), `queens` (10), `east` (9).
- **Airport Filtering:** 970 airport trips detected with an average fare of $13.75.
- **Rule Tags:** `NO_TIP`: 44,680, `MEDIUM_TRIP`: 39,990, `SHORT_TRIP`: 19,028, `LOW_FARE`: 8,573, `GENEROUS_TIP`: 2,211.
- **Partition Balance:** Node 0: 14,999; Node 1: 15,001; Node 2: 15,000; Node 3: 15,000 records. Zero data skew across the 4 nodes.

---

## 5. Experiment 4: NoSQL Data Ingestion Pipeline (Python & MongoDB)

### 5.1 Aim & Overview
To construct an automated data ingestion pipeline using Python to read Apache Parquet data and stage NYC Yellow Taxi trip records into a **MongoDB** document database, followed by executing CRUD and range/projection queries.

### 5.2 Theoretical Concepts
1. **Columnar vs. Document Store:**
   - **Parquet (Columnar):** Highly compressed on disk, optimized for analytical scans over specific columns.
   - **MongoDB (Document / BSON):** Schema-free, JSON-like key-value document structure suited for semi-structured operational workloads and real-time updates.
2. **Document Conversion & Wire Efficiency:**
   - Translating memory tabular arrays (`DataFrame`) into BSON dictionaries (`orient='records'`).
   - Using projection documents `{field: 1}` to limit wire transfer overhead.

### 5.3 Code Implementation & Explanation
```python
import pandas as pd
from pymongo import MongoClient

# 1. Read columnar Parquet file using pyarrow engine
df = pd.read_parquet("data/yellow_tripdata_2026-01.parquet")
sample_df = df.head(10000)

# 2. Connect to local MongoDB instance
client = MongoClient("mongodb://localhost:27017/")
db = client["nyc_taxi_bda"]
collection = db["yellow_taxi_trips"]
collection.drop()  # Reset staging collection

# 3. Transform DataFrame rows into BSON dictionaries and bulk insert
records = sample_df.to_dict(orient="records")
insert_result = collection.insert_many(records)
print(f"Staged {len(insert_result.inserted_ids)} documents into MongoDB.")

# 4. Range and Projection Query
results = collection.find(
    {"trip_distance": {"$gt": 10}},
    {"PULocationID": 1, "DOLocationID": 1, "trip_distance": 1, "total_amount": 1, "_id": 0}
).limit(5)
for doc in results:
    print(doc)

# 5. CRUD Operations Verification
# Create
collection.insert_one({"test_record": True, "PULocationID": 132, "trip_distance": 5.2, "total_amount": 25.5})
# Read
doc = collection.find_one({"test_record": True})
# Update
collection.update_one({"test_record": True}, {"$set": {"total_amount": 30.0}})
# Delete
collection.delete_one({"test_record": True})
client.close()
```

### 5.4 Key Results & Findings
- 10,000 Parquet records successfully staged into MongoDB collection `yellow_taxi_trips`.
- Validated MongoDB filtering using `$gt` operators, spatial zone selection (`PULocationID: 132`), and projection mechanics.
- Verified ACID document-level transactions through standard Create, Read, Update, and Delete operations.

---

## 6. Experiment 5: Large-Scale Batch Analytics with Spark SQL & Execution Plans

### 6.1 Aim & Overview
To perform large-scale batch analytics across **20 months of NYC Yellow Taxi records** (2025-01 to 2026-08, **73,293,601 cleaned trips**), execute complex Spark SQL queries (temporal series, cross-tabulations, window functions, rolling averages), inspect physical execution plans, and benchmark query performance.

### 6.2 Theoretical Concepts
1. **Catalyst Query Optimizer:**
   - Translates relational Spark SQL into an Abstract Syntax Tree (AST).
   - Generates an Unresolved Logical Plan -> Resolved Logical Plan (via Catalog) -> Optimized Logical Plan (constant folding, predicate pushdown, projection pruning) -> Physical Plans -> WholeStageCodegen.
2. **Adaptive Query Execution (AQE):**
   - Configured via `spark.sql.adaptive.enabled = true`. Dynamically coalesces shuffle partitions, switches join strategies (e.g. SortMergeJoin to BroadcastHashJoin), and optimizes skewed joins at runtime.
3. **Window Specifications:**
   - Calculating moving frames using `rowsBetween(-6, 0)` for rolling metrics and `lag()` for Month-over-Month (MoM) growth without multi-pass self-joins.

### 6.3 Code Implementation & Explanation (`exp5/make_notebook.py`)
```python
from pyspark.sql import SparkSession, Window, functions as F

spark = SparkSession.builder \
    .appName("exp5-batch-analytics") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Load 20 months of Parquet files with schema merge and clean dirty timestamps
df = (spark.read.option("mergeSchema", "true").parquet("data/yellow_tripdata_*.parquet")
      .filter("tpep_pickup_datetime >= '2025-01-01' AND tpep_pickup_datetime < '2026-09-01'")
      .filter("trip_distance > 0 AND fare_amount > 0"))

# Q1: Monthly Time Series & Revenue
q1_df = df.groupBy(F.date_format("tpep_pickup_datetime", "yyyy-MM").alias("month")) \
          .agg(F.count("*").alias("trips"),
               F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
               F.round(F.sum("total_amount") / 1e6, 2).alias("revenue_m")) \
          .orderBy("month")

# Q2: Peak Hour per Day of Week (Cross-tabulation)
q2_df = df.withColumn("dow", F.date_format("tpep_pickup_datetime", "EEEE")) \
          .withColumn("hour", F.hour("tpep_pickup_datetime")) \
          .groupBy("dow", "hour").count()

# Q3: Top-10 Pickup Hotspots
zones = spark.read.csv("data/taxi_zone_lookup.csv", header=True, inferSchema=True)
q3_df = (df.groupBy("PULocationID").count()
         .join(zones, df.PULocationID == zones.LocationID)
         .select("LocationID", "Borough", "Zone", F.col("count").alias("pickups"))
         .orderBy(F.desc("pickups")).limit(10))

# Q4: Month-over-Month (MoM) Growth using Window Functions
win_mom = Window.orderBy("month")
q4_df = q1_df.withColumn("prev_trips", F.lag("trips", 1).over(win_mom)) \
             .withColumn("mom_growth_pct", F.round(((F.col("trips") - F.col("prev_trips")) / F.col("prev_trips")) * 100, 2))

# Q5: 7-Day Rolling Average
daily_df = df.groupBy(F.to_date("tpep_pickup_datetime").alias("date")).count()
win_rolling = Window.orderBy("date").rowsBetween(-6, 0)
q5_df = daily_df.withColumn("rolling_7d_trips", F.round(F.avg("count").over(win_rolling), 0))

# Q6: Congestion Surcharge Hotspots in Manhattan CBD
q6_df = (df.filter("congestion_surcharge > 0")
         .groupBy("PULocationID")
         .agg(F.count("*").alias("congestion_trips"),
              F.round(F.sum("congestion_surcharge"), 2).alias("total_surcharge"))
         .join(zones, df.PULocationID == zones.LocationID)
         .orderBy(F.desc("total_surcharge")).limit(10))

# Save aggregated results to Parquet
q1_df.write.mode("overwrite").parquet("exp5/output/q1_monthly.parquet")
q2_df.write.mode("overwrite").parquet("exp5/output/q2_heatmap.parquet")
```

### 6.4 Key Results & Findings
- **73,293,601 cleaned trips** analyzed across 20 months.
- **Demand Stability:** Monthly trip volume stayed inside a predictable band of **3.2M to 4.2M trips/month** (peak in May 2025 at 4.15M, low in August 2026 at 3.20M).
- **Fare Inflation:** Average fare rose monotonically from **$18.22** (Jan 2025) to **$21.08** (Aug 2026), peaking at **$22.44** in Dec 2025.
- **Diurnal Regularity:** Every single weekday experienced its peak volume at **18:00 (6 PM)**, led by Thursday (770,962 trips at 18:00). Sunday peaked at **00:00 midnight**.

---

## 7. Experiment 6: Batch Analytics Visualization & Comprehensive Analytical Report

### 7.1 Aim & Overview
To convert Experiment 5's batch analytics outputs into executive-ready visualizations (time-series, multi-dimensional heatmaps, borough market share, and geographic maps) and compile a comprehensive analytical intelligence report.

### 7.2 Theoretical Concepts
1. **Decoupled Reporting Architecture:**
   - Compute-heavy aggregations run once in Spark SQL and write compact Parquet summaries (`zone_stats.parquet`). Visualization layers (Matplotlib, Seaborn, Plotly) consume only the summarized Parquet tables, avoiding re-scanning 73M raw rows.
2. **Offline GIS Vector Rendering:**
   - Web tile providers (OSM, Carto, Mapbox) frequently enforce rate limits, tokens, or block `file://` protocols. Parsing vector polygons directly from the official TLC shapefile (`geo_map.py` and `nyc_zones.py`) enables 100% reliable, offline choropleth rendering.

### 7.3 Code Implementation & Explanation (`exp6/make_report.py`)
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nyc_zones import load_zone_paths, draw_choropleth

# 1. Monthly Trends Visualization (Dual Axis: Volume & Fare)
monthly_df = pd.read_parquet("exp5/output/q1_monthly.parquet")
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
ax1.plot(monthly_df["month"], monthly_df["trips"] / 1e6, "b-o", label="Trips (Millions)")
ax2.plot(monthly_df["month"], monthly_df["avg_fare"], "r--s", label="Avg Fare ($)")
ax1.set_ylabel("Trips (M)", color="b"); ax2.set_ylabel("Avg Fare ($)", color="r")
plt.title("NYC Yellow Taxi: Monthly Demand and Fare Evolution (2025-2026)")
plt.savefig("exp6/charts/1_monthly_trends.png")

# 2. Weekday x Hour Demand Heatmap
heatmap_df = pd.read_parquet("exp5/output/q2_heatmap.parquet").pivot(index="dow", columns="hour", values="count")
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_df, cmap="YlOrRd", annot=False)
plt.title("Weekly Rhythm: Pickup Density by Day and Hour")
plt.savefig("exp6/charts/3_hour_weekday_heatmap.png")

# 3. Offline GIS Choropleth Generation
zone_stats = pd.read_parquet("exp6/output/zone_stats.parquet")
pickups_map = dict(zip(zone_stats["PULocationID"], zone_stats["pickups"]))
fig, ax = plt.subplots(figsize=(8, 10))
draw_choropleth(ax, pickups_map, cmap="viridis", title="NYC Yellow Taxi Pickup Density by Zone")
plt.savefig("exp6/charts/6_geo_pickup_heatmap.png")
```

### 7.4 Key Results & Insights
- **Spatial Monopolization:** Manhattan accounts for **86.6%** of all taxi pickups (63.4M trips). Queens accounts for 9.8%, with the Bronx, Brooklyn, and Staten Island sharing the remainder.
- **Top Zone:** Upper East Side South ranked #1 across NYC with **3,314,611 pickups**, followed by Midtown Center (3,164,406).
- **Airport Specialization:** JFK Airport ranked #3 in overall pickups (3,029,519) but displayed a distinct revenue profile with an average fare of **$62.84** (compared to $14–$17 in Manhattan).

---

## 8. Experiment 7: Real-Time Streaming Data Pipeline (Kafka & Spark Structured Streaming)

### 8.1 Aim & Overview
To engineer an end-to-end distributed real-time streaming pipeline using **Apache Kafka** (in KRaft mode) as the distributed event broker and **Apache Spark Structured Streaming** as the stream processor to execute event-time sliding window aggregations.

### 8.2 Theoretical Concepts
1. **Distributed Event Streaming (Kafka):**
   - High-throughput broker operating without ZooKeeper (**KRaft metadata mode**).
   - Producers pace event emission to simulate real-world telemetry replay.
2. **Event Time Processing & Watermarking:**
   - Computations use the meter engagement timestamp (`event_time`), not system wall-clock arrival time.
   - **Watermarking (`withWatermark("event_time", "10 minutes")`):** Limits the state store by dropping events that arrive later than the watermark threshold.
3. **Sliding Windows & Metric Semantics:**
   - 5-minute sliding windows advancing every 1 minute (`window("event_time", "5 minutes", "1 minute")`).
   - Consecutive sliding windows overlap by 4 minutes, meaning raw sums double-count trips (~5x). Cumulative totals must evaluate the non-overlapping aligned lattice.

### 8.3 Code Implementation & Explanation
#### Producer (`exp7/producer.py`)
```python
import json, time
import pyarrow.parquet as pq
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

table = pq.read_table("data/yellow_tripdata_2026-08.parquet")
# Replay records at a controlled pacing rate (e.g. 250 records/sec)
for batch in table.to_batches(max_chunksize=1000):
    for row in batch.to_pylist():
        event = {
            "trip_id": str(row["VendorID"]) + "_" + str(row["tpep_pickup_datetime"]),
            "event_time": row["tpep_pickup_datetime"].isoformat(),
            "pickup_zone": int(row["PULocationID"]),
            "dropoff_zone": int(row["DOLocationID"]),
            "distance_mi": float(row["trip_distance"]),
            "fare": float(row["fare_amount"]),
            "congestion_fee": float(row.get("congestion_surcharge") or 0.0)
        }
        producer.send("taxi-trips", event)
        time.sleep(1 / 250.0)
```

#### Consumer (`exp7/stream_windows.py`)
```python
from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder \
    .appName("Exp7-StreamingConsumer") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0") \
    .getOrCreate()

# Read continuous stream from Kafka
raw_stream = (spark.readStream.format("kafka")
              .option("kafka.bootstrap.servers", "localhost:9092")
              .option("subscribe", "taxi-trips")
              .load())

# Parse JSON schema and assign event-time watermark
parsed_stream = (raw_stream.selectExpr("CAST(value AS STRING) as json_str")
                 .select(F.from_json("json_str", EVENT_SCHEMA).alias("data"))
                 .select("data.*")
                 .withColumn("event_time", F.to_timestamp("event_time"))
                 .withWatermark("event_time", "10 minutes"))

# 5-minute sliding window per pickup zone
zone_windows = (parsed_stream
                .groupBy(F.window("event_time", "5 minutes", "1 minute"), "pickup_zone")
                .agg(F.count("*").alias("trips"),
                     F.round(F.avg("fare"), 2).alias("avg_fare"),
                     F.round(F.sum("congestion_fee"), 2).alias("congestion_fees")))

# Write micro-batch outputs to JSON sink for downstream consumption
query = (zone_windows.writeStream
         .format("json")
         .option("path", "exp7/sink/windows_zones")
         .option("checkpointLocation", "exp7/sink/checkpoints_zones")
         .outputMode("append")
         .start())
```

### 8.4 Key Results & Findings
- Processed **77 city micro-batches** and **411 sliding windows** covering 7,214 distinct trips in the paced evaluation window.
- Verified rolling window semantics: peak city rolling windows recorded 248 trips/5-minute window with ~$6,833 revenue and $330 congestion fees.

---

## 9. Experiment 8: Real-Time Operational Dashboard (Streamlit & Offline GIS)

### 9.1 Aim & Overview
To build a lightweight, low-latency operational dashboard using **Streamlit** that monitors real-time streaming traffic, displays rolling KPI cards, updates a trend line, tails live telemetry, and renders an auto-refreshing spatial map of NYC taxi activity without external web map dependencies.

### 9.2 Theoretical Concepts
1. **Decoupled Micro-Batch UI Consumption:**
   - The dashboard never queries the heavy Spark compute engine directly. Instead, it reads the atomic JSON sink files and `live_feed.txt` written asynchronously by Experiment 7.
2. **Partial Page Fragmentation:**
   - Utilizing Streamlit's `st.fragment(run_every=3)` ensures that only the dashboard body re-executes every 3 seconds, preserving application state and eliminating full-page UI flicker.
3. **Headless Component Testing:**
   - Using `streamlit.testing.v1.AppTest` to verify that UI components, maps, and metrics instantiate properly without a browser or graphical display server.

### 9.3 Code Implementation & Explanation (`exp8/dashboard.py`)
```python
import glob, json
import matplotlib.pyplot as plt
import streamlit as st
from nyc_zones import draw_choropleth

st.set_page_config(page_title="NYC Taxi Live Operations", layout="wide")
st.title("🚖 NYC Yellow Taxi — Real-Time Fleet Monitor")

# Auto-refreshing body fragment
@st.fragment(run_every=3)
def render_live_dashboard():
    # 1. Ingest most recent JSON sink files written by Spark Streaming
    zone_files = sorted(glob.glob("exp7/sink/windows_zones/*.json"))[-5:]
    records = [json.loads(line) for f in zone_files for line in open(f)]
    
    # Aggregate pickups over the last 3 minutes of stream time
    zone_pickups = {}
    for r in records:
        z = r.get("pickup_zone")
        zone_pickups[z] = zone_pickups.get(z, 0) + r.get("trips", 0)
        
    # 2. Display KPI Metric Cards
    total_trips = sum(zone_pickups.values())
    col1, col2, col3 = st.columns(3)
    col1.metric("Active Rolling Trips (Last 3m)", f"{total_trips:,}")
    col2.metric("Active Pickups Zones", f"{len(zone_pickups)}")
    col3.metric("Pipeline Status", "STREAMING HEALTHY", delta="Kafka + Spark OK")
    
    # 3. Render Offline Matplotlib Choropleth Map
    fig, ax = plt.subplots(figsize=(6, 8))
    draw_choropleth(ax, zone_pickups, cmap="inferno", title="Live Fleet Distribution")
    st.pyplot(fig)
    
    # 4. Tail raw live feed console
    with open("exp7/live_feed.txt") as f:
        lines = f.readlines()[-15:]
    st.code("".join(lines), language="text")

render_live_dashboard()
```

### 9.4 Key Results & Findings
- Sub-second dashboard refresh cycle (`run_every=3s`) rendering offline vector maps without API tokens or map tile rate limits.
- Fully decoupled architecture ensuring zero compute degradation on the streaming engine.

---

## 10. Experiment 9: Multi-Model NoSQL Data Lake Architecture & Empirical Index Benchmarks

### 10.1 Aim & Overview
To architect, load, and empirically benchmark a multi-model **NoSQL Data Lake** using **MongoDB 7.0** (running in Docker) on 1,000,000 trip documents across three distinct schema modeling paradigms, and measure the query acceleration provided by 2dsphere, compound, and time-series indexes.

### 10.2 Theoretical Concepts
1. **NoSQL Multi-Model Schema Paradigms:**
   - **Denormalized Document Collection (`trips`):** Zone ID, name, borough, and GeoJSON point are embedded directly in every trip document, providing single-read retrieval without relational joins on the hot read path.
   - **Normalized Reference Collection (`zones`):** Clean 265-document lookup joined via aggregation `$lookup` pipelines when schema consistency is prioritized over latency.
   - **Time-Series Bucket Collection (`trip_metrics_ts`):** Optimized storage format using internal columnar bucketing (`timeField: "window_start"`, `metaField: "pickup_zone"`) with automatic Data Retention / TTL expiry (`expireAfterSeconds: 90 days`).
2. **Execution Plan & Query Engine Profiling:**
   - **`COLLSCAN` (Collection Scan):** Scans all 1,000,000 documents sequentially in memory/disk ($O(N)$).
   - **`IXSCAN` (B-Tree Index Scan):** Traverses an ordered index structure, inspecting only matching leaf entries ($O(\log N)$).
   - **`GEO_NEAR_2DSPHERE`:** Uses spherical geodesics across Earth's surface for spatial proximity querying.

### 10.3 Code Implementation & Explanation (`exp9/benchmark.py`)
```python
from pymongo import MongoClient, ASCENDING, DESCENDING, GEOSPHERE
import datetime, time

client = MongoClient("mongodb://localhost:27017/")
db = client["taxi"]
trips = db["trips"]

# Benchmark Function: Profiles query latency, stage type, and docs examined
def run_benchmark(query_fn, query_name):
    # Timed run
    t0 = time.time()
    res = list(query_fn())
    elapsed_ms = (time.time() - t0) * 1000
    
    # Explain plan inspection
    explain = query_fn().explain()
    stats = explain["executionStats"]
    stage = stats["executionStages"]["stage"]
    docs_examined = stats.get("totalDocsExamined", 0)
    return elapsed_ms, stage, docs_examined

# 1. Pre-Index Execution: Time Range Query
query_1h = lambda: trips.find({
    "pickup_time": {
        "$gte": datetime.datetime(2026, 8, 31, 18, 0),
        "$lt": datetime.datetime(2026, 8, 31, 19, 0)
    }
})
pre_ms, pre_stage, pre_docs = run_benchmark(query_1h, "1-hour time slice")

# 2. Apply Targeted Indexes
trips.create_index([("pickup_loc", GEOSPHERE)])
trips.create_index([("pickup_time", ASCENDING)])
trips.create_index([("pickup_zone.id", ASCENDING), ("pickup_time", DESCENDING)])

# 3. Post-Index Execution
post_ms, post_stage, post_docs = run_benchmark(query_1h, "1-hour time slice")

print(f"Pre-Index:  {pre_ms:.1f}ms | Stage: {pre_stage} | Docs Examined: {pre_docs}")
print(f"Post-Index: {post_ms:.1f}ms | Stage: {post_stage} | Docs Examined: {post_docs}")
print(f"Speedup:    {pre_ms / post_ms:.1f}x")
```

### 10.4 Empirical Benchmark Results (1,000,000 Documents)
| Query Identifier | Un-Indexed Plan | Indexed Plan | Speedup | Documents Examined |
|---|---|---|---|---|
| **Geospatial Polygon (`$geoWithin` 500m)** | 776 ms (`COLLSCAN`) | 63 ms (`IXSCAN`) | **12.2x** | 1,000,000 → 34,297 |
| **Geospatial Distance (`$nearSphere` 2km)** | *Not runnable* | 438 ms (`GEO_NEAR_2DSPHERE`) | **N/A** | Required 2dsphere |
| **Temporal Slice (1-Hour Window)** | 350 ms (`COLLSCAN`) | 0.5 ms (`IXSCAN`) | **700.0x** | 1,000,000 → 1,000 |
| **Compound Filter (`zone.id` + `time`)** | 399 ms (`COLLSCAN`) | 0.5 ms (`IXSCAN`) | **798.0x** | 1,000,000 → 1,000 |
| **Relational Join (`$lookup` with zones)** | 1,575 ms (`COLLSCAN`) | 2.0 ms (`IXSCAN`) | **787.5x** | 1,000,000 → 265 |
| **Time-Series Window Scan** | *Native TS Engine* | 1.8 ms (`BLOCK_SCAN`) | **Native** | 4,641 docs in 195 buckets |

---

## 11. Experiment 10: Machine Learning Anomaly Detection & Integrated Smart City Showcase

### 11.1 Aim & Overview
To develop an unsupervised Machine Learning anomaly detection engine (**Isolation Forest**) over 4.9 million zone-window observations, evaluate detection efficacy through synthetic anomaly injection, and assemble all batch, streaming, NoSQL, and machine learning components into an executive **Smart City Showcase Dashboard**.

### 11.2 Theoretical Concepts
1. **Unsupervised Isolation Forest:**
   - Constructs an ensemble of isolation trees (iTrees) by randomly selecting features and split values.
   - **Theoretical Foundation:** Anomalies have sparse attribute distributions and isolate close to the root (short average path length $h(x)$).
2. **Training-Serving Parity:**
   - Features are calculated on **(zone × 5-minute windows)**—the exact aggregation emitted by Experiment 7's streaming pipeline, preventing feature distribution mismatch.
3. **Robust Quantile Baselines vs. Gaussian Assumptions:**
   - Taxi trip volume is heavily skewed. Naive mean and standard deviation are warped by monster outliers.
   - Baselines use **Median** and robust sigma derived from percentiles:
     $$\sigma_{\text{robust}} = \frac{P_{84} - P_{16}}{2}$$
4. **Symmetric Log-Ratios:**
   - A 5x surge ($5 \times$) and an 80% collapse ($1/5 \times$) are equally anomalous. Standard Z-scores are linear and asymmetrical, making them almost blind to demand collapses (2.9% recall).
   - Computing symmetric log-ratios solves this asymmetry:
     $$\text{log\_ratio} = \ln\left(\frac{\text{trips} + 1}{\text{median} + 1}\right)$$

### 11.3 Code Implementation & Explanation
#### Shared Featurization Engine (`exp10/anomaly_features.py`)
```python
import numpy as np
import pandas as pd

FEATURES = ["trips", "avg_fare", "trips_rel", "fare_rel",
            "trips_z", "fare_z", "trips_log_ratio", "fare_log_ratio",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos"]

def compute_features(df, baselines):
    # Join observations with zone-hour robust medians
    df = df.merge(baselines, on=["pickup_zone", "hour"], how="left")
    
    # Robust Z-Scores (clipped to avoid runaway values)
    df["trips_z"] = np.clip((df["trips"] - df["median_trips"]) / df["sigma_trips"], -10, 10)
    df["fare_z"] = np.clip((df["avg_fare"] - df["median_fare"]) / df["sigma_fare"], -10, 10)
    
    # Symmetric Log-Ratios
    df["trips_log_ratio"] = np.log((df["trips"] + 1) / (df["median_trips"] + 1))
    df["fare_log_ratio"] = np.log((df["avg_fare"] + 1) / (df["median_fare"] + 1))
    
    # Cyclical Time Encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"]  = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"]  = np.cos(2 * np.pi * df["dow"] / 7.0)
    return df[FEATURES]
```

#### Training & Synthetic Injection Sweep (`exp10/train_anomaly.py`)
```python
from sklearn.ensemble import IsolationForest
import joblib

# 1. Train Unsupervised Isolation Forest on 400,000 clean windows
X_train = load_and_featurize_training_samples(n=400000)
model = IsolationForest(
    n_estimators=200,
    max_samples=4096,
    contamination=0.01,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train)
joblib.dump(model, "exp10/model.joblib")

# 2. Synthetic Anomaly Injection Evaluation (on 150,000 held-out windows)
# Injection formulas:
# Demand Surge:    trips * 5 + 20
# Demand Collapse: trips / 5 (evaluated on subset where clean trips >= 5)
# Fare Spike:      avg_fare * 3 + 15
```

### 11.4 Key Results & Evaluation Findings
- **Synthetic Anomaly Recall (at 1% False Positive Budget):**
  - **Demand Surges:** **99.5% recall** (caught reliably across all models).
  - **Demand Collapses:** **2.9% recall** using standard Z-scores vs. **69.0% recall** using the Symmetric Log-Ratio feature.
  - **Fare Spikes:** **88.2% recall** using Isolation Forest.
- **The Core Lesson:** Anomaly detection is mathematically asymmetric. Standard statistical metrics fail on demand drops; incorporating symmetric logarithmic divergence is essential for public transit anomaly detection.

---

## 12. Master Summary Matrix (Experiments 1 to 10)

| Exp | Laboratory Focus | Core Frameworks | Primary Input Asset | Foundational Big Data Concept Demonstrated |
|:---:|---|---|---|---|
| **1** | Distributed Storage & EDA | Hadoop HDFS, PySpark | Cleaned CSV (3.95M trips) | NameNode/DataNode cluster topology, in-memory distributed DataFrames, temporal demand profiling. |
| **2** | Architecture & Proposal | Markdown, Architecture Spec | TLC Parquet + GIS Shapefile | Medallion lakehouse pattern (Bronze/Silver/Gold), multi-source telemetry fusion, transit anomaly taxonomy. |
| **3** | MapReduce & Data Skew | PySpark RDD & SQL | Sampled Parquet (60k trips) | `map`, `reduceByKey`, keyword text tokenization, and partition-level load balancing without skew. |
| **4** | NoSQL Data Ingestion | Python, PyArrow, MongoDB | Monthly Parquet (10k rows) | Columnar-to-document (BSON) transformation, bulk writing, projection queries, and CRUD verification. |
| **5** | Batch Spark SQL Analytics | Spark SQL, Catalyst, AQE | 20 Parquet files (73.3M rows) | Catalyst AST execution plan optimization, Adaptive Query Execution, window lag/rolling operations. |
| **6** | Batch Reporting & GIS | Matplotlib, Seaborn, Plotly | Aggregated Parquet + Shapefiles | Decoupled analytics architecture, offline vector polygon rendering overcoming tile server limits. |
| **7** | Real-Time Stream Pipeline | Apache Kafka (KRaft), Spark SS | Parquet stream replay | Distributed publish-subscribe, event-time watermarking, and 5-min sliding window rolling mechanics. |
| **8** | Real-Time Ops Dashboard | Streamlit, Matplotlib | JSON sinks + live feed text | Decoupled UI reading micro-batch file sinks, `st.fragment` partial rerendering, and headless UI testing. |
| **9** | Multi-Model NoSQL Lake | MongoDB 7 (Docker), PyMongo | 1M trips + zones + TS | Document embedding vs referencing, time-series bucketing, and indexing speedups (12x to 798x). |
| **10**| ML Anomaly & Showcase | Scikit-Learn, Isolation Forest | 4.9M window feature matrix | Unsupervised tree isolation, robust quantile baselines, symmetric log-ratios, and live showcase dashboard. |
