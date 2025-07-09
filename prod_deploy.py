"""
Production Deployment System for NBA Recommendation Engine
Real-time data pipeline, monitoring, and performance tracking
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from confluent_kafka import Producer, Consumer
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import mlflow
import mlflow.sklearn
import redis
from flask import Flask, request, jsonify
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio

@dataclass
class RecommendationRequest:
    """Data class for recommendation requests"""
    customer_id: str
    channel_context: Dict
    timestamp: datetime
    region: str
    brand: str
    session_id: str

@dataclass
class RecommendationResponse:
    """Data class for recommendation responses"""
    customer_id: str
    recommended_action: str
    channel: str
    confidence_score: float
    timestamp: datetime
    model_version: str
    ab_test_group: str

class RealTimeDataPipeline:
    """Real-time data pipeline using Kafka and Spark Streaming"""
    
    def __init__(self, kafka_config: Dict, spark_config: Dict):
        self.kafka_config = kafka_config
        self.spark_config = spark_config
        self.spark = self._initialize_spark()
        self.logger = logging.getLogger(__name__)
        
    def _initialize_spark(self):
        """Initialize Spark session for streaming"""
        return SparkSession.builder \
            .appName("NBA_RealTimePipeline") \
            .config("spark.sql.streaming.checkpointLocation", "/tmp/checkpoints") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
    
    def setup_kafka_stream(self, topic: str):
        """Setup Kafka streaming source"""
        df = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_config['bootstrap_servers']) \
            .option("subscribe", topic) \
            .option("startingOffsets", "latest") \
            .load()
        
        # Parse JSON messages
        parsed_df = df.select(
            from_json(col("value").cast("string"), self._get_message_schema()).alias("data")
        ).select("data.*")
        
        return parsed_df
    
    def _get_message_schema(self):
        """Define schema for incoming messages"""
        return StructType([
            StructField("customer_id", StringType(), True),
            StructField("event_type", StringType(), True),
            StructField("channel", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("properties", MapType(StringType(), StringType()), True)
        ])
    
    def process_stream(self, stream_df):
        """Process real-time stream data"""
        # Add derived features
        processed_df = stream_df \
            .withColumn("hour_of_day", hour("timestamp")) \
            .withColumn("day_of_week", dayofweek("timestamp")) \
            .withColumn("is_weekend", 
                       when(dayofweek("timestamp").isin([1, 7]), 1).otherwise(0))
        
        # Windowed aggregations
        windowed_df = processed_df \
            .groupBy(
                window("timestamp", "5 minutes", "1 minute"),
                "customer_id"
            ) \
            .agg(
                count("*").alias("event_count"),
                collect_list("channel").alias("channels_used"),
                max("timestamp").alias("last_activity")
            )
        
        return windowed_df
    
    def write_to_output(self, df, output_path: str, output_format: str = "parquet"):
        """Write processed data to output sink"""
        query = df.writeStream \
            .outputMode("append") \
            .format(output_format) \
            .option("path", output_path) \
            .option("checkpointLocation", f"/tmp/checkpoints/{output_format}") \
            .start()
        
        return query

class ModelServing:
    """Model serving infrastructure for real-time predictions"""
    
    def __init__(self, model_registry_uri: str, redis_host: str = "localhost"):
        self.model_registry_uri = model_registry_uri
        self.redis_client = redis.Redis(host=redis_host, port=6379, decode_responses=True)
        self.models = {}
        self.model_versions = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(model_registry_uri)
        
    def load_models(self, model_names: List[str]):
        """Load models from MLflow registry"""
        for model_name in model_names:
            try:
                # Get latest version
                latest_version = mlflow.MlflowClient().get_latest_versions(
                    model_name, stages=["Production"]
                )[0]
                
                # Load model
                model_uri = f"models:/{model_name}/Production"
                model = mlflow.sklearn.load_model(model_uri)
                
                self.models[model_name] = model
                self.model_versions[model_name] = latest_version.version
                
                self.logger.info(f"Loaded model {model_name} version {latest_version.version}")
                
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {str(e)}")
    
    def predict(self, model_name: str, features: np.ndarray) -> np.ndarray:
        """Make prediction using specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        predictions = model.predict(features)
        
        # Cache recent predictions
        cache_key = f"prediction:{model_name}:{hash(features.tobytes())}"
        self.redis_client.setex(cache_key, 3600, json.dumps(predictions.tolist()))
        
        return predictions
    
    def predict_proba(self, model_name: str, features: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(features)
        else:
            # For models without predict_proba, return binary predictions
            predictions = model.predict(features)
            proba = np.zeros((len(predictions), 2))
            proba[np.arange(len(predictions)), predictions] = 1.0
            return proba
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get model information"""
        if model_name not in self.models:
            return {"error": "Model not loaded"}
        
        return {
            "model_name": model_name,
            "version": self.model_versions.get(model_name, "unknown"),
            "loaded_at": datetime.now().isoformat(),
            "features": getattr(self.models[model_name], 'feature_names_in_', None)
        }
    
    def reload_model(self, model_name: str):
        """Reload specific model from registry"""
        if model_name in self.models:
            self.logger.info(f"Reloading model {model_name}")
            self.load_models([model_name])

class FeatureStore:
    """Feature store for managing ML features"""
    
    def __init__(self, redis_host: str = "localhost"):
        self.redis_client = redis.Redis(host=redis_host, port=6379, decode_responses=True)
        self.logger = logging.getLogger(__name__)
        
    def get_features(self, customer_id: str, feature_names: List[str]) -> Dict:
        """Get features for a customer"""
        features = {}
        
        for feature_name in feature_names:
            cache_key = f"feature:{customer_id}:{feature_name}"
            value = self.redis_client.get(cache_key)
            
            if value:
                features[feature_name] = json.loads(value)
            else:
                features[feature_name] = None
                
        return features
    
    def set_features(self, customer_id: str, features: Dict, ttl: int = 3600):
        """Set features for a customer"""
        for feature_name, value in features.items():
            cache_key = f"feature:{customer_id}:{feature_name}"
            self.redis_client.setex(cache_key, ttl, json.dumps(value))
    
    def compute_realtime_features(self, customer_id: str, context: Dict) -> Dict:
        """Compute real-time features from context"""
        features = {}
        
        # Time-based features
        now = datetime.now()
        features['hour_of_day'] = now.hour
        features['day_of_week'] = now.weekday()
        features['is_weekend'] = 1 if now.weekday() >= 5 else 0
        
        # Context-based features
        features['channel'] = context.get('channel', 'unknown')
        features['region'] = context.get('region', 'unknown')
        features['brand'] = context.get('brand', 'unknown')
        
        # Session features
        session_id = context.get('session_id')
        if session_id:
            session_key = f"session:{session_id}"
            session_data = self.redis_client.get(session_key)
            if session_data:
                session_info = json.loads(session_data)
                features['session_duration'] = (now - datetime.fromisoformat(session_info['start_time'])).seconds
                features['page_views'] = session_info.get('page_views', 0)
            else:
                features['session_duration'] = 0
                features['page_views'] = 0
        
        return features

class ABTestManager:
    """A/B testing framework for model experiments"""
    
    def __init__(self, redis_host: str = "localhost"):
        self.redis_client = redis.Redis(host=redis_host, port=6379, decode_responses=True)
        self.logger = logging.getLogger(__name__)
        
    def get_test_group(self, customer_id: str, test_name: str) -> str:
        """Get A/B test group for a customer"""
        cache_key = f"ab_test:{test_name}:{customer_id}"
        group = self.redis_client.get(cache_key)
        
        if not group:
            # Assign customer to test group based on hash
            hash_value = hash(customer_id + test_name) % 100
            group = "A" if hash_value < 50 else "B"
            
            # Cache assignment
            self.redis_client.setex(cache_key, 86400, group)  # 24 hours
            
        return group
    
    def log_test_result(self, customer_id: str, test_name: str, group: str, 
                       outcome: str, metrics: Dict):
        """Log A/B test results"""
        result = {
            "customer_id": customer_id,
            "test_name": test_name,
            "group": group,
            "outcome": outcome,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in Redis list for later analysis
        self.redis_client.lpush(f"ab_results:{test_name}", json.dumps(result))
        
    def get_test_results(self, test_name: str, limit: int = 1000) -> List[Dict]:
        """Get A/B test results for analysis"""
        results = self.redis_client.lrange(f"ab_results:{test_name}", 0, limit - 1)
        return [json.loads(result) for result in results]

class RecommendationEngine:
    """Main recommendation engine orchestrating all components"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_serving = ModelServing(config['model_registry_uri'])
        self.feature_store = FeatureStore(config['redis_host'])
        self.ab_test_manager = ABTestManager(config['redis_host'])
        self.logger = logging.getLogger(__name__)
        
        # Load models
        self.model_serving.load_models(config['model_names'])
        
    def generate_recommendation(self, request: RecommendationRequest) -> RecommendationResponse:
        """Generate recommendation for a customer"""
        try:
            # Get A/B test group
            test_group = self.ab_test_manager.get_test_group(
                request.customer_id, "model_version_test"
            )
            
            # Select model based on A/B test
            model_name = "recommendation_model_v1" if test_group == "A" else "recommendation_model_v2"
            
            # Get features
            feature_names = self.config['feature_names']
            stored_features = self.feature_store.get_features(request.customer_id, feature_names)
            realtime_features = self.feature_store.compute_realtime_features(
                request.customer_id, request.channel_context
            )
            
            # Combine features
            all_features = {**stored_features, **realtime_features}
            
            # Convert to numpy array for model
            feature_vector = np.array([
                all_features.get(name, 0) for name in feature_names
            ]).reshape(1, -1)
            
            # Make prediction
            prediction = self.model_serving.predict(model_name, feature_vector)[0]
            confidence = self.model_serving.predict_proba(model_name, feature_vector)[0].max()
            
            # Map prediction to action
            action_mapping = {
                0: "no_action",
                1: "email_campaign",
                2: "push_notification",
                3: "discount_offer",
                4: "product_recommendation"
            }
            
            recommended_action = action_mapping.get(prediction, "no_action")
            
            # Create response
            response = RecommendationResponse(
                customer_id=request.customer_id,
                recommended_action=recommended_action,
                channel=request.channel_context.get('channel', 'unknown'),
                confidence_score=float(confidence),
                timestamp=datetime.now(),
                model_version=self.model_serving.model_versions.get(model_name, "unknown"),
                ab_test_group=test_group
            )
            
            # Log for A/B testing
            self.ab_test_manager.log_test_result(
                request.customer_id,
                "model_version_test",
                test_group,
                recommended_action,
                {"confidence": confidence}
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {str(e)}")
            # Return default response
            return RecommendationResponse(
                customer_id=request.customer_id,
                recommended_action="no_action",
                channel="unknown",
                confidence_score=0.0,
                timestamp=datetime.now(),
                model_version="error",
                ab_test_group="error"
            )

class MonitoringService:
    """Monitoring and alerting service"""
    
    def __init__(self, redis_host: str = "localhost"):
        self.redis_client = redis.Redis(host=redis_host, port=6379, decode_responses=True)
        self.logger = logging.getLogger(__name__)
        self.metrics = {}
        
    def log_request(self, request: RecommendationRequest, response: RecommendationResponse, 
                   latency: float):
        """Log recommendation request and response"""
        log_entry = {
            "customer_id": request.customer_id,
            "recommended_action": response.recommended_action,
            "confidence_score": response.confidence_score,
            "latency_ms": latency,
            "timestamp": datetime.now().isoformat(),
            "model_version": response.model_version,
            "ab_test_group": response.ab_test_group
        }
        
        # Store in Redis
        self.redis_client.lpush("recommendation_logs", json.dumps(log_entry))
        
        # Update metrics
        self.update_metrics(log_entry)
        
    def update_metrics(self, log_entry: Dict):
        """Update real-time metrics"""
        now = datetime.now()
        minute_key = now.strftime("%Y-%m-%d_%H:%M")
        
        # Request count
        self.redis_client.incr(f"requests_per_minute:{minute_key}")
        self.redis_client.expire(f"requests_per_minute:{minute_key}", 3600)
        
        # Average latency
        latency_key = f"latency:{minute_key}"
        self.redis_client.lpush(latency_key, log_entry['latency_ms'])
        self.redis_client.expire(latency_key, 3600)
        
        # Action distribution
        action_key = f"action_count:{log_entry['recommended_action']}:{minute_key}"
        self.redis_client.incr(action_key)
        self.redis_client.expire(action_key, 3600)
        
    def get_metrics(self, time_range: int = 60) -> Dict:
        """Get system metrics for the last N minutes"""
        now = datetime.now()
        metrics = {
            "request_count": 0,
            "average_latency": 0,
            "action_distribution": {},
            "timestamp": now.isoformat()
        }
        
        total_requests = 0
        total_latency = 0
        
        for i in range(time_range):
            minute = (now - timedelta(minutes=i)).strftime("%Y-%m-%d_%H:%M")
            
            # Request count
            count = self.redis_client.get(f"requests_per_minute:{minute}")
            if count:
                total_requests += int(count)
                
            # Latency
            latencies = self.redis_client.lrange(f"latency:{minute}", 0, -1)
            if latencies:
                minute_latency = sum(float(l) for l in latencies)
                total_latency += minute_latency
        
        metrics["request_count"] = total_requests
        metrics["average_latency"] = total_latency / total_requests if total_requests > 0 else 0
        
        return metrics
    
    def check_alerts(self):
        """Check for alert conditions"""
        metrics = self.get_metrics(5)  # Last 5 minutes
        
        alerts = []
        
        # High latency alert
        if metrics["average_latency"] > 1000:  # 1 second
            alerts.append({
                "type": "high_latency",
                "message": f"High latency detected: {metrics['average_latency']:.2f}ms",
                "severity": "warning"
            })
        
        # Low request volume alert
        if metrics["request_count"] < 10:  # Less than 10 requests in 5 minutes
            alerts.append({
                "type": "low_volume",
                "message": f"Low request volume: {metrics['request_count']} requests",
                "severity": "info"
            })
        
        return alerts

class APIServer:
    """Flask API server for recommendation service"""
    
    def __init__(self, recommendation_engine: RecommendationEngine, 
                 monitoring_service: MonitoringService):
        self.app = Flask(__name__)
        self.recommendation_engine = recommendation_engine
        self.monitoring_service = monitoring_service
        self.logger = logging.getLogger(__name__)
        
        self.setup_routes()
        
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})
        
        @self.app.route('/recommend', methods=['POST'])
        def recommend():
            start_time = time.time()
            
            try:
                data = request.get_json()
                
                # Create request object
                req = RecommendationRequest(
                    customer_id=data['customer_id'],
                    channel_context=data.get('channel_context', {}),
                    timestamp=datetime.now(),
                    region=data.get('region', 'unknown'),
                    brand=data.get('brand', 'unknown'),
                    session_id=data.get('session_id', '')
                )
                
                # Generate recommendation
                response = self.recommendation_engine.generate_recommendation(req)
                
                # Calculate latency
                latency = (time.time() - start_time) * 1000
                
                # Log request
                self.monitoring_service.log_request(req, response, latency)
                
                return jsonify({
                    "customer_id": response.customer_id,
                    "recommended_action": response.recommended_action,
                    "channel": response.channel,
                    "confidence_score": response.confidence_score,
                    "timestamp": response.timestamp.isoformat(),
                    "model_version": response.model_version,
                    "ab_test_group": response.ab_test_group
                })
                
            except Exception as e:
                self.logger.error(f"Error in recommendation endpoint: {str(e)}")
                return jsonify({"error": "Internal server error"}), 500
        
        @self.app.route('/metrics', methods=['GET'])
        def get_metrics():
            time_range = request.args.get('time_range', 60, type=int)
            metrics = self.monitoring_service.get_metrics(time_range)
            return jsonify(metrics)
        
        @self.app.route('/alerts', methods=['GET'])
        def get_alerts():
            alerts = self.monitoring_service.check_alerts()
            return jsonify({"alerts": alerts})
        
        @self.app.route('/models', methods=['GET'])
        def get_models():
            models_info = {}
            for model_name in self.recommendation_engine.model_serving.models.keys():
                models_info[model_name] = self.recommendation_engine.model_serving.get_model_info(model_name)
            return jsonify(models_info)
    
    def run(self, host: str = "0.0.0.0", port: int = 5000):
        """Run the API server"""
        self.app.run(host=host, port=port, debug=False)

class StreamingRecommendationProcessor:
    """Process streaming data for real-time recommendations"""
    
    def __init__(self, kafka_config: Dict, recommendation_engine: RecommendationEngine):
        self.kafka_config = kafka_config
        self.recommendation_engine = recommendation_engine
        self.consumer = Consumer(kafka_config)
        self.producer = Producer(kafka_config)
        self.logger = logging.getLogger(__name__)
        
    def process_stream(self, input_topic: str, output_topic: str):
        """Process streaming recommendation requests"""
        self.consumer.subscribe([input_topic])
        
        try:
            while True:
                msg = self.consumer.poll(1.0)
                
                if msg is None:
                    continue
                    
                if msg.error():
                    self.logger.error(f"Consumer error: {msg.error()}")
                    continue
                
                try:
                    # Parse message
                    data = json.loads(msg.value().decode('utf-8'))
                    
                    # Create request
                    request = RecommendationRequest(
                        customer_id=data['customer_id'],
                        channel_context=data.get('channel_context', {}),
                        timestamp=datetime.now(),
                        region=data.get('region', 'unknown'),
                        brand=data.get('brand', 'unknown'),
                        session_id=data.get('session_id', '')
                    )
                    
                    # Generate recommendation
                    response = self.recommendation_engine.generate_recommendation(request)
                    
                    # Send to output topic
                    response_data = {
                        "customer_id": response.customer_id,
                        "recommended_action": response.recommended_action,
                        "channel": response.channel,
                        "confidence_score": response.confidence_score,
                        "timestamp": response.timestamp.isoformat(),
                        "model_version": response.model_version,
                        "ab_test_group": response.ab_test_group
                    }
                    
                    self.producer.produce(
                        output_topic,
                        key=response.customer_id,
                        value=json.dumps(response_data)
                    )
                    self.producer.flush()
                    
                except Exception as e:
                    self.logger.error(f"Error processing message: {str(e)}")
                    
        except KeyboardInterrupt:
            self.logger.info("Stopping stream processor")
        finally:
            self.consumer.close()

def main():
    """Main function to run the recommendation system"""
    
    # Configuration
    config = {
        "model_registry_uri": "http://localhost:5000",
        "redis_host": "localhost",
        "model_names": ["recommendation_model_v1", "recommendation_model_v2"],
        "feature_names": [
            "hour_of_day", "day_of_week", "is_weekend", "channel",
            "region", "brand", "session_duration", "page_views"
        ],
        "kafka_config": {
            "bootstrap.servers": "localhost:9092",
            "group.id": "recommendation_service",
            "auto.offset.reset": "earliest"
        },
        "spark_config": {
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true"
        }
    }
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize components
    recommendation_engine = RecommendationEngine(config)
    monitoring_service = MonitoringService(config['redis_host'])
    api_server = APIServer(recommendation_engine, monitoring_service)
    
    # Setup streaming processor
    streaming_processor = StreamingRecommendationProcessor(
        config['kafka_config'], recommendation_engine
    )
    
    # Run API server in a separate thread
    server_thread = threading.Thread(
        target=api_server.run,
        kwargs={"host": "0.0.0.0", "port": 5000}
    )
    server_thread.daemon = True
    server_thread.start()
    
    # Run streaming processor
    streaming_processor.process_stream(
        "recommendation_requests", 
        "recommendation_responses"
    )

if __name__ == "__main__":
    main()
