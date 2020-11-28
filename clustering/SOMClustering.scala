import org.apache.spark.sql.SparkSession
import xyz.florentforest.spark.ml.som.SOM
import org.apache.spark.sql.functions._
import org.apache.spark.ml._

val spark = SparkSession.builder
  .appName("Python Spark Bisecting Kmeans")
  .config("spark.some.config.option", "some-value")
  .getOrCreate()

import spark.implicits._
spark.sparkContext.setLogLevel("WARN")

val file_path = "dbfs:/FileStore/tables/data_preprocess/part1.csv"
val data_rdd = spark.sparkContext.textFile(file_path).map(line => (1, linalg.Vectors.dense(line.split (",").map(_.toDouble)))).collect().toList
val data = data_rdd.toDF("id", "features")

for( val num <- 2 until 6) {
  val this_file_path = "dbfs:/FileStore/tables/data_preprocess/part" + num.toString + ".csv"
  val file_data_rdd = spark.sparkContext.textFile(this_file_path).map(line => (1, linalg.Vectors.dense(line.split (",").map(_.toDouble)))).collect().toList
  val file_data = data_rdd.toDF("id", "features")
  data = data.union(file_data)
}
// data.show()

val som = new SOM()
  .setHeight(10)
  .setWidth(10)
val model = som.fit(data)

val summary = model.summary // training summary

val res = summary.predictions
val result_file = "dbfs:/FileStore/tables/SOM_result/10x10_total.csv"
res.select("prediction").repartition(1)
  .write.format("com.databricks.spark.csv")
  .option("header", "true")
  .save(result_file)
// res.show()

val cost: Array[Double] = summary.objectiveHistory
println(cost.mkString("[", ",", "]"))