"""
Data Pipeline for NBA Recommendation System
Handles multi-channel data integration, preprocessing, and feature engineering
"""

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from datetime import datetime, timedelta
import logging

class DataPipeline:
    def __init__(self, spark_config=None):
        """Initialize Spark session and configure data pipeline"""
        self.spark = SparkSession.builder \
            .appName("NBA_DataPipeline") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        self.channels = ['email', 'mobile', 'web', 'instore', 'directmail']
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def collect_channel_data(self, channel_configs):
        """
        Collect data from all 5 channels: Email, Mobile, Web, In-store, Direct Mail
        """
        channel_data = {}
        
        for channel, config in channel_configs.items():
            try:
                if channel == 'email':
                    df = self._collect_email_data(config)
                elif channel == 'mobile':
                    df = self._collect_mobile_data(config)
                elif channel == 'web':
                    df = self._collect_web_data(config)
                elif channel == 'instore':
                    df = self._collect_instore_data(config)
                elif channel == 'directmail':
                    df = self._collect_directmail_data(config)
                
                channel_data[channel] = df
                self.logger.info(f"Successfully collected {channel} data: {df.count()} records")
                
            except Exception as e:
                self.logger.error(f"Error collecting {channel} data: {str(e)}")
                
        return channel_data
    
    def _collect_email_data(self, config):
        """Collect email campaign data"""
        schema = StructType([
            StructField("customer_id", StringType(), True),
            StructField("email_id", StringType(), True),
            StructField("campaign_id", StringType(), True),
            StructField("sent_date", TimestampType(), True),
            StructField("opened", BooleanType(), True),
            StructField("clicked", BooleanType(), True),
            StructField("converted", BooleanType(), True),
            StructField("channel", StringType(), True)
        ])
        
        df = self.spark.read \
            .option("header", "true") \
            .schema(schema) \
            .csv(config['path'])
        
        return df.withColumn("channel", lit("email"))
    
    def _collect_mobile_data(self, config):
        """Collect mobile app interaction data"""
        schema = StructType([
            StructField("customer_id", StringType(), True),
            StructField("session_id", StringType(), True),
            StructField("app_version", StringType(), True),
            StructField("interaction_date", TimestampType(), True),
            StructField("screen_views", IntegerType(), True),
            StructField("time_spent", DoubleType(), True),
            StructField("push_notification_clicked", BooleanType(), True),
            StructField("in_app_purchase", BooleanType(), True)
        ])
        
        df = self.spark.read \
            .option("header", "true") \
            .schema(schema) \
            .csv(config['path'])
        
        return df.withColumn("channel", lit("mobile"))
    
    def _collect_web_data(self, config):
        """Collect web analytics data"""
        schema = StructType([
            StructField("customer_id", StringType(), True),
            StructField("session_id", StringType(), True),
            StructField("page_views", IntegerType(), True),
            StructField("session_duration", DoubleType(), True),
            StructField("bounce_rate", DoubleType(), True),
            StructField("conversion_event", BooleanType(), True),
            StructField("referrer_source", StringType(), True),
            StructField("visit_date", TimestampType(), True)
        ])
        
        df = self.spark.read \
            .option("header", "true") \
            .schema(schema) \
            .csv(config['path'])
        
        return df.withColumn("channel", lit("web"))
    
    def _collect_instore_data(self, config):
        """Collect in-store transaction data"""
        schema = StructType([
            StructField("customer_id", StringType(), True),
            StructField("store_id", StringType(), True),
            StructField("transaction_id", StringType(), True),
            StructField("transaction_date", TimestampType(), True),
            StructField("purchase_amount", DoubleType(), True),
            StructField("product_category", StringType(), True),
            StructField("loyalty_points_used", IntegerType(), True),
            StructField("staff_interaction", BooleanType(), True)
        ])
        
        df = self.spark.read \
            .option("header", "true") \
            .schema(schema) \
            .csv(config['path'])
        
        return df.withColumn("channel", lit("instore"))
    
    def _collect_directmail_data(self, config):
        """Collect direct mail campaign data"""
        schema = StructType([
            StructField("customer_id", StringType(), True),
            StructField("campaign_id", StringType(), True),
            StructField("mail_type", StringType(), True),
            StructField("sent_date", TimestampType(), True),
            StructField("response_date", TimestampType(), True),
            StructField("responded", BooleanType(), True),
            StructField("offer_redeemed", BooleanType(), True)
        ])
        
        df = self.spark.read \
            .option("header", "true") \
            .schema(schema) \
            .csv(config['path'])
        
        return df.withColumn("channel", lit("directmail"))
    
    def collect_historical_customer_data(self, customer_data_path):
        """Collect historical customer demographic and behavioral data"""
        customer_schema = StructType([
            StructField("customer_id", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("gender", StringType(), True),
            StructField("region", StringType(), True),
            StructField("brand_preference", StringType(), True),
            StructField("customer_segment", StringType(), True),
            StructField("lifetime_value", DoubleType(), True),
            StructField("acquisition_channel", StringType(), True),
            StructField("registration_date", TimestampType(), True),
            StructField("last_purchase_date", TimestampType(), True),
            StructField("total_purchases", IntegerType(), True),
            StructField("average_order_value", DoubleType(), True)
        ])
        
        customer_df = self.spark.read \
            .option("header", "true") \
            .schema(customer_schema) \
            .csv(customer_data_path)
        
        return customer_df
    
    def apply_brand_region_segmentation(self, df):
        """Apply brand-region specific segmentation"""
        segmentation_df = df.withColumn(
            "brand_region_segment",
            concat(col("brand_preference"), lit("_"), col("region"))
        ).withColumn(
            "segment_priority",
            when(col("lifetime_value") > 1000, "High")
            .when(col("lifetime_value") > 500, "Medium")
            .otherwise("Low")
        )
        
        return segmentation_df
    
    def create_customer_journey_features(self, channel_data, customer_data):
        """Create customer journey-based features for ML models"""
        
        # Combine all channel interactions
        combined_interactions = None
        for channel, df in channel_data.items():
            if combined_interactions is None:
                combined_interactions = df
            else:
                combined_interactions = combined_interactions.union(df)
        
        # Create journey features
        journey_features = combined_interactions.groupBy("customer_id") \
            .agg(
                count("*").alias("total_interactions"),
                countDistinct("channel").alias("channel_diversity"),
                max("interaction_date").alias("last_interaction_date"),
                min("interaction_date").alias("first_interaction_date"),
                avg("conversion_rate").alias("avg_conversion_rate")
            )
        
        # Add recency, frequency, monetary features
        journey_features = journey_features.withColumn(
            "recency_days",
            datediff(current_date(), col("last_interaction_date"))
        ).withColumn(
            "interaction_frequency",
            col("total_interactions") / 
            (datediff(col("last_interaction_date"), col("first_interaction_date")) + 1)
        )
        
        # Join with customer data
        final_features = customer_data.join(journey_features, "customer_id", "left") \
            .fillna(0)
        
        return final_features
    
    def prepare_training_data(self, features_df, target_column="next_best_action"):
        """Prepare data for ML model training"""
        
        # Create sequential features for RNN/LSTM
        sequential_features = features_df.select(
            "customer_id",
            "total_interactions",
            "channel_diversity",
            "recency_days",
            "interaction_frequency",
            "lifetime_value",
            "average_order_value",
            target_column
        )
        
        # Create time-based windows for sequential modeling
        windowed_features = sequential_features.withColumn(
            "interaction_window",
            floor(col("recency_days") / 7)  # Weekly windows
        )
        
        return windowed_features
    
    def save_processed_data(self, df, output_path, format="parquet"):
        """Save processed data to specified format"""
        df.write \
            .mode("overwrite") \
            .option("path", output_path) \
            .save(format)
        
        self.logger.info(f"Data saved to {output_path} in {format} format")
    
    def get_real_time_stream(self, stream_config):
        """Setup real-time data streaming for production"""
        stream_df = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", stream_config['servers']) \
            .option("subscribe", stream_config['topic']) \
            .load()
        
        return stream_df
    
    def close(self):
        """Close Spark session"""
        self.spark.stop()

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Channel data configuration
    channel_configs = {
        'email': {'path': '/data/email_campaigns.csv'},
        'mobile': {'path': '/data/mobile_interactions.csv'},
        'web': {'path': '/data/web_analytics.csv'},
        'instore': {'path': '/data/instore_transactions.csv'},
        'directmail': {'path': '/data/directmail_campaigns.csv'}
    }
    
    # Process data
    channel_data = pipeline.collect_channel_data(channel_configs)
    customer_data = pipeline.collect_historical_customer_data('/data/customer_profiles.csv')
    
    # Apply segmentation
    segmented_customers = pipeline.apply_brand_region_segmentation(customer_data)
    
    # Create features
    features = pipeline.create_customer_journey_features(channel_data, segmented_customers)
    
    # Prepare training data
    training_data = pipeline.prepare_training_data(features)
    
    # Save processed data
    pipeline.save_processed_data(training_data, '/output/processed_training_data')
    
    pipeline.close()
