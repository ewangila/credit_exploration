import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# PART 1: Initial Data Exploration

spark = SparkSession.builder.master('local').appName('CreditCardDefault').getOrCreate()
spark_df = spark.read.csv('./credit_card_default.csv', header=True, inferSchema=True)

# Standardize the target column name if needed
if 'default payment next month' in spark_df.columns:
    spark_df = spark_df.withColumnRenamed('default payment next month', 'default')
elif 'default.payment.next.month' in spark_df.columns:
    spark_df = spark_df.withColumnRenamed('default.payment.next.month', 'default')

print("--- First 5 Records ---")
spark_df.show(5)
print("\n--- Schema ---")
spark_df.printSchema()

print("\n--- Distinct Categories ---")
spark_df.select('SEX').distinct().show()
spark_df.select('EDUCATION').distinct().show()
spark_df.select('MARRIAGE').distinct().show()

# Plot Raw Categories
edu_pd = spark_df.groupBy('EDUCATION').count().toPandas()
edu_pd.plot(kind='bar', x='EDUCATION', y='count', title='Education (Raw)')
plt.savefig('education_raw.png', bbox_inches='tight')
plt.show()

mar_pd = spark_df.groupBy('MARRIAGE').count().toPandas()
mar_pd.plot(kind='bar', x='MARRIAGE', y='count', title='Marriage (Raw)')
plt.savefig('marriage_raw.png', bbox_inches='tight')
plt.show()


# PART 2: Binning

spark_df = spark_df.withColumn(
    'EDUCATION',
    F.when(F.col('EDUCATION').isin(0, 5, 6, '0', '5', '6'), 'Other')
     .otherwise(F.col('EDUCATION').cast('string'))
)

spark_df = spark_df.withColumn(
    'MARRIAGE',
    F.when(F.col('MARRIAGE').isin(0, '0'), 'Other')
     .otherwise(F.col('MARRIAGE').cast('string'))
)

print("\n--- Distinct Binned Categories ---")
spark_df.select('EDUCATION').distinct().show()
spark_df.select('MARRIAGE').distinct().show()

# Plot Binned Categories
edu_binned_pd = spark_df.groupBy('EDUCATION').count().toPandas()
edu_binned_pd.plot(kind='bar', x='EDUCATION', y='count', title='Education (Binned)', legend=False)
plt.xticks(rotation=0)
plt.savefig('education_binned.png', bbox_inches='tight')
plt.show()

mar_binned_pd = spark_df.groupBy('MARRIAGE').count().toPandas()
mar_binned_pd.plot(kind='bar', x='MARRIAGE', y='count', title='Marriage (Binned)', legend=False)
plt.xticks(rotation=0)
plt.savefig('marriage_binned.png', bbox_inches='tight')
plt.show()


# PART 3: Class Balance Exploration

# Step 7: Target Data
target_pd = spark_df.groupBy('default').count().toPandas()

# Map labels to be more readable
target_pd['default_label'] = target_pd['default'].map({0: 'Default (0)', 1: 'Non-Default (1)'})

target_pd.plot(kind='bar', x='default_label', y='count', title='Class Balance', color=['skyblue', 'salmon'], legend=False)
plt.xlabel('Target Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.savefig('class_balance.png', bbox_inches='tight')
plt.show()


# Step 8: Target Data Grouped by SEX
target_sex_pd = spark_df.groupBy('default', 'SEX').count().toPandas()

# Pivot the data to create grouped side-by-side bars
target_sex_pivot = target_sex_pd.pivot(index='default', columns='SEX', values='count')
target_sex_pivot.index = target_sex_pivot.index.map({0: 'Default (0)', 1: 'Non-Default (1)'})

target_sex_pivot.plot(kind='bar', title='Default Rate by Sex')
plt.xlabel('Target Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Sex')
plt.savefig('default_by_sex.png', bbox_inches='tight')
plt.show()

# Clean up
spark.stop()