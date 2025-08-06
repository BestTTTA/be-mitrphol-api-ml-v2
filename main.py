from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import pandas as pd
import xgboost as xgb
import joblib
import os
import requests
import numpy as np
from pydantic import BaseModel
from collections import defaultdict
import asyncio
import aiohttp
import time
import logging
import json
import hashlib
import redis.asyncio as redis
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Prediction Service with Redis Cache", version="1.1.0")

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODELS_DIR = "models-ml"
BASE_URL = "https://mitrphol.api.rust.thetigerteamacademy.net"
REDIS_URL = "redis://119.59.102.60:6380"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
CACHE_TTL = 31536000  
CACHE_PREFIX = "ml_predict"

# Prediction Level Constants
PREDICTION_THRESHOLDS = {
    "HIGH_MIN": 12.0,  # > 12
    "MEDIUM_MIN": 10.0,  # >= 10 and <= 12
    "MEDIUM_MAX": 12.0,
    "LOW_MAX": 10.0  # < 10
}

# Redis connection
redis_client = None

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    year: int
    start_month: int
    end_month: int
    models: List[str]
    zones: Optional[str] = "MAC,MKB,MKS,MPDC,MPK,MPL,MPV,SB"
    limit: Optional[int] = 50000

class GroupedPredictionRequest(BaseModel):
    year: int
    start_month: int
    end_month: int
    models: List[str]
    zones: Optional[str] = "MAC,MKB,MKS,MPDC,MPK,MPL,MPV,SB"
    limit: Optional[int] = 50000
    group_by_level: Optional[bool] = True

class PlantPrediction(BaseModel):
    lat: float
    lon: float
    plant_id: str
    prediction: float  # ใช้แค่ prediction อย่างเดียว ไม่ใช้ upper/lower bound
    ndvi: float
    ndwi: float
    gli: float
    precipitation: float
    zone: str
    cane_type: str

class GroupedPlantPrediction(BaseModel):
    lat: float
    lon: float
    plant_id: str
    prediction: float
    prediction_level: str  # "HIGH", "MEDIUM", "LOW"
    ndvi: float
    ndwi: float
    gli: float
    precipitation: float
    zone: str
    cane_type: str

class PredictionGroup(BaseModel):
    level: str  # "HIGH", "MEDIUM", "LOW"
    count: int
    percentage: float
    average_prediction: float
    predictions: List[GroupedPlantPrediction]

class ZoneStatistics(BaseModel):
    zone: str
    high_prediction_count: int
    high_prediction_percentage: float
    medium_prediction_count: int
    medium_prediction_percentage: float
    low_prediction_count: int
    low_prediction_percentage: float
    total_plantations: int
    average_prediction: float

class ModelPredictionResult(BaseModel):
    model_name: str
    predictions: List[PlantPrediction]
    zone_statistics: List[ZoneStatistics]
    overall_average: float

class GroupedModelPredictionResult(BaseModel):
    model_name: str
    prediction_groups: List[PredictionGroup]
    zone_statistics: List[ZoneStatistics]
    overall_average: float
    total_predictions: int

class PredictionResponse(BaseModel):
    success: bool
    message: str
    results: List[ModelPredictionResult]
    cached: Optional[bool] = False
    cache_key: Optional[str] = None

class GroupedPredictionResponse(BaseModel):
    success: bool
    message: str
    results: List[GroupedModelPredictionResult]
    cached: Optional[bool] = False
    cache_key: Optional[str] = None

# Global variables for loaded models
loaded_models = {}

# Redis connection management
async def get_redis_client():
    """สร้าง Redis connection"""
    global redis_client
    if redis_client is None:
        try:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            # Test connection
            await redis_client.ping()
            logger.info("Redis connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            redis_client = None
    return redis_client

async def close_redis_client():
    """ปิด Redis connection"""
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None

# Cache key generation
def generate_cache_key(request_type: str, **params) -> str:
    """สร้าง cache key จาก parameters"""
    # เรียงลำดับ parameters เพื่อให้ได้ key เดียวกัน
    sorted_params = dict(sorted(params.items()))
    
    # แปลง list เป็น string และเรียงลำดับ
    for key, value in sorted_params.items():
        if isinstance(value, list):
            sorted_params[key] = sorted(value)
    
    # สร้าง string จาก parameters
    param_string = json.dumps(sorted_params, sort_keys=True)
    
    # สร้าง hash
    param_hash = hashlib.md5(param_string.encode()).hexdigest()
    
    return f"{CACHE_PREFIX}:{request_type}:{param_hash}"

async def get_from_cache(cache_key: str):
    """ดึงข้อมูลจาก Redis cache"""
    try:
        redis_conn = await get_redis_client()
        if redis_conn is None:
            return None
        
        cached_data = await redis_conn.get(cache_key)
        if cached_data:
            logger.info(f"Cache HIT for key: {cache_key}")
            return json.loads(cached_data)
        
        logger.info(f"Cache MISS for key: {cache_key}")
        return None
    except Exception as e:
        logger.error(f"Error getting from cache: {e}")
        return None

async def set_to_cache(cache_key: str, data: dict, ttl: int = CACHE_TTL):
    """บันทึกข้อมูลลง Redis cache"""
    try:
        redis_conn = await get_redis_client()
        if redis_conn is None:
            return False
        
        # เพิ่ม metadata
        cache_data = {
            "data": data,
            "cached_at": datetime.now().isoformat(),
            "ttl": ttl
        }
        
        await redis_conn.setex(
            cache_key, 
            ttl, 
            json.dumps(cache_data, default=str)
        )
        
        logger.info(f"Data cached with key: {cache_key}, TTL: {ttl}s")
        return True
    except Exception as e:
        logger.error(f"Error setting cache: {e}")
        return False

async def delete_cache_pattern(pattern: str):
    """ลบ cache ตาม pattern"""
    try:
        redis_conn = await get_redis_client()
        if redis_conn is None:
            return 0
        
        keys = await redis_conn.keys(pattern)
        if keys:
            deleted_count = await redis_conn.delete(*keys)
            logger.info(f"Deleted {deleted_count} cache keys matching pattern: {pattern}")
            return deleted_count
        return 0
    except Exception as e:
        logger.error(f"Error deleting cache pattern: {e}")
        return 0

def get_prediction_level(prediction_value: float) -> str:
    """กำหนดระดับ prediction ตามเกณฑ์ที่กำหนด"""
    if prediction_value > PREDICTION_THRESHOLDS["HIGH_MIN"]:
        return "HIGH"
    elif PREDICTION_THRESHOLDS["MEDIUM_MIN"] <= prediction_value <= PREDICTION_THRESHOLDS["MEDIUM_MAX"]:
        return "MEDIUM"
    else:  # prediction_value < PREDICTION_THRESHOLDS["LOW_MAX"]
        return "LOW"

def load_model(model_name: str):
    """โหลด model และ metadata"""
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    try:
        model_path = os.path.join(MODELS_DIR, f"{model_name}.json")
        metadata_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model JSON file not found: {model_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model PKL file not found: {metadata_path}")
        
        booster = xgb.Booster()
        booster.load_model(model_path)
        metadata = joblib.load(metadata_path)
        
        required_features = metadata.get('required_features', [])
        if not required_features:
            logger.warning(f"No required_features found in metadata for {model_name}")
            months = ['February', 'March', 'April', 'May', 'June', 'July', 'August']
            required_features = []
            for feature in ['NDVI', 'GLI', 'NDWI', 'Precipitation']:
                for month in months:
                    required_features.append(f"{feature}_{month}")
            
            logger.info(f"Generated required_features: {len(required_features)} features")
        
        loaded_models[model_name] = {
            'booster': booster,
            'metadata': metadata,
            'required_features': required_features
        }
        
        logger.info(f"Model {model_name} loaded successfully with {len(required_features)} required features")
        return loaded_models[model_name]
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")

async def fetch_raw_data_with_retry(year: int, start_month: int, end_month: int, zones: str = "MAC,MKB,MKS,MPDC,MPK,MPL,MPV,SB", limit: int = 50000):
    """ดึงข้อมูลดิบจาก API พร้อม retry mechanism และ cache"""
    
    # สร้าง cache key สำหรับ raw data
    raw_data_cache_key = generate_cache_key(
        "raw_data",
        year=year,
        start_month=start_month,
        end_month=end_month,
        zones=zones,
        limit=limit
    )
    
    # ลองดึงจาก cache ก่อน
    cached_data = await get_from_cache(raw_data_cache_key)
    if cached_data and "data" in cached_data:
        logger.info("Using cached raw data")
        return cached_data["data"]
    
    url = f"{BASE_URL}/raw-data"
    params = {
        'year': year,
        'start_month': start_month,
        'end_month': end_month,
        'zones': zones,
        'limit': limit
    }
    
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Fetching data from API (attempt {attempt + 1}/{MAX_RETRIES})")
            
            timeout = aiohttp.ClientTimeout(total=60)  # 60 second timeout
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=10)
            
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                async with session.get(url, params=params) as response:
                    logger.info(f"API Response Status: {response.status}")
                    
                    if response.status == 200:
                        response_data = await response.json()
                        
                        # ตรวจสอบโครงสร้างข้อมูล
                        if isinstance(response_data, dict) and 'data' in response_data:
                            data = response_data['data']
                        elif isinstance(response_data, list):
                            data = response_data
                        else:
                            raise ValueError(f"Unexpected data structure: {type(response_data)}")
                        
                        if not data:
                            raise ValueError("No data returned from API")
                        
                        logger.info(f"Successfully fetched {len(data)} records")
                        
                        # Cache the raw data
                        await set_to_cache(raw_data_cache_key, data, ttl=1800)  # 30 minutes for raw data
                        
                        return data
                    
                    elif response.status in [502, 503, 504]:  # Server errors that we can retry
                        error_msg = f"Server error {response.status} from API"
                        logger.warning(f"{error_msg} - attempt {attempt + 1}")
                        last_error = HTTPException(status_code=response.status, detail=error_msg)
                        
                        if attempt < MAX_RETRIES - 1:
                            logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                            await asyncio.sleep(RETRY_DELAY)
                            continue
                    
                    else:
                        # Client errors (400, 401, 403, etc.) - don't retry
                        error_msg = f"Client error {response.status} from API"
                        logger.error(error_msg)
                        raise HTTPException(status_code=response.status, detail=error_msg)
                        
        except asyncio.TimeoutError:
            error_msg = f"Timeout error when fetching data (attempt {attempt + 1})"
            logger.warning(error_msg)
            last_error = HTTPException(status_code=504, detail="API request timeout")
            
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                await asyncio.sleep(RETRY_DELAY)
                continue
                
        except aiohttp.ClientError as e:
            error_msg = f"Network error when fetching data: {str(e)} (attempt {attempt + 1})"
            logger.warning(error_msg)
            last_error = HTTPException(status_code=502, detail=f"Network error: {str(e)}")
            
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                await asyncio.sleep(RETRY_DELAY)
                continue
                
        except Exception as e:
            error_msg = f"Unexpected error when fetching data: {str(e)}"
            logger.error(error_msg)
            last_error = HTTPException(status_code=500, detail=error_msg)
            break
    
    # If we've exhausted all retries, raise the last error
    if last_error:
        raise last_error
    else:
        raise HTTPException(status_code=500, detail="Failed to fetch data after all retry attempts")

def prepare_data_for_model(raw_data: List[Dict], required_features: List[str]):
    """เตรียมข้อมูลสำหรับ model"""
    try:
        df = pd.DataFrame(raw_data)
        
        logger.info(f"Available columns: {list(df.columns)}")
        logger.info(f"Total records: {len(df)}")
        logger.info(f"Required features count: {len(required_features)}")
        
        if 'plant_id' not in df.columns:
            df['plant_id'] = df.apply(lambda row: f"{row['lat']:.6f}_{row['lng']:.6f}", axis=1)
            logger.info("Created plant_id from lat, lng coordinates")
        
        required_cols = ['lat', 'lng', 'month', 'year', 'plant_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df['zone'] = df['zone'].fillna('Unknown').replace('', 'Unknown')
        df['cane_type'] = df['cane_type'].fillna('Unknown').replace('', 'Unknown')
        
        logger.info(f"Found {df['plant_id'].nunique()} unique plants")
        
        processed_data = []
        
        for plant_id, group in df.groupby('plant_id'):
            plant_row = {
                'PlantID': str(plant_id),
                'Lat': float(group['lat'].iloc[0]),
                'Lon': float(group['lng'].iloc[0]),
                'zone': str(group['zone'].iloc[0]),
                'cane_type': str(group['cane_type'].iloc[0])
            }
            
            # Initialize features
            for feat in required_features:
                plant_row[feat] = 0.0
            
            month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                          'July', 'August', 'September', 'October', 'November', 'December']
            
            # Fill actual data
            for _, row in group.iterrows():
                if 'month' in row and pd.notna(row['month']) and 1 <= row['month'] <= 12:
                    month_name = month_names[int(row['month']) - 1]
                    
                    feature_mappings = {
                        'NDVI': 'ndvi',
                        'NDWI': 'ndwi', 
                        'GLI': 'gli',
                        'Precipitation': 'precipitation'
                    }
                    
                    for feature_type, column_name in feature_mappings.items():
                        feature_name = f"{feature_type}_{month_name}"
                        if feature_name in required_features:
                            plant_row[feature_name] = float(row.get(column_name, 0) or 0)
            
            processed_data.append(plant_row)
        
        result_df = pd.DataFrame(processed_data)
        
        logger.info(f"Processed {len(result_df)} plant records")
        
        # Add missing features
        missing_features = [feat for feat in required_features if feat not in result_df.columns]
        if missing_features:
            logger.info(f"Adding missing features: {len(missing_features)}")
            for feat in missing_features:
                result_df[feat] = 0.0
        
        if result_df.empty:
            raise ValueError("No data after processing")
        
        logger.info(f"Final data shape: {result_df.shape}")
        return result_df
        
    except Exception as e:
        logger.error(f"Error in prepare_data_for_model: {str(e)}")
        raise

def predict_with_model(model_info: Dict, data: pd.DataFrame) -> pd.DataFrame:
    """ทำนายผลด้วย model - ส่งคืนเฉพาะค่า prediction อย่างเดียว"""
    try:
        required_features = model_info['required_features']
        booster = model_info['booster']
        
        logger.info(f"Model requires {len(required_features)} features")
        
        missing_features = [feat for feat in required_features if feat not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        feature_data = data[required_features].fillna(0)
        logger.info(f"Feature data shape: {feature_data.shape}")
        
        dmatrix = xgb.DMatrix(feature_data)
        scores = booster.predict(dmatrix)
        
        logger.info(f"Prediction successful! Scores shape: {scores.shape if hasattr(scores, 'shape') else len(scores)}")
        
        # คำนวณ average features
        avg_features = {}
        months_in_model = set()
        for feat in required_features:
            if '_' in feat:
                month = feat.split('_', 1)[1]
                months_in_model.add(month)
        
        months_list = sorted(list(months_in_model))
        
        for feature in ["NDVI", "NDWI", "GLI", "Precipitation"]:
            feature_cols = [f"{feature}_{month}" for month in months_list 
                           if f"{feature}_{month}" in required_features]
            if feature_cols:
                avg_features[feature] = data[feature_cols].mean(axis=1)
            else:
                avg_features[feature] = pd.Series([0.0] * len(data))
        
        # สร้างผลลัพธ์
        predictions = pd.DataFrame({
            "lat": data["Lat"].astype(float),
            "lon": data["Lon"].astype(float),
            "plant_id": data["PlantID"].astype(str),
            "zone": data["zone"],
            "cane_type": data["cane_type"]
        })
        
        # ใช้เฉพาะค่า prediction อย่างเดียว ไม่ใช้ upper/lower bound
        if hasattr(scores, 'shape') and len(scores.shape) > 1:
            # ถ้า model ส่งคืนหลายค่า ใช้ค่ากลาง
            if scores.shape[1] >= 3:
                predictions["prediction"] = scores[:, 1]  # ค่ากลาง
            else:
                predictions["prediction"] = scores[:, 0]
        else:
            predictions["prediction"] = scores
        
        # เพิ่ม average features
        for feature, values in avg_features.items():
            predictions[feature.lower()] = values
        
        logger.info(f"Generated predictions for {len(predictions)} records")
        return predictions
        
    except Exception as e:
        logger.error(f"Error in predict_with_model: {str(e)}")
        raise

def calculate_zone_statistics(predictions: pd.DataFrame) -> List[ZoneStatistics]:
    """คำนวณสถิติแต่ละ zone ด้วยเกณฑ์: >12, 10-12, <10"""
    zone_stats = []
    
    for zone, zone_data in predictions.groupby('zone'):
        total = len(zone_data)
        
        # ใช้เกณฑ์ที่กำหนด
        high_pred = zone_data[zone_data['prediction'] > PREDICTION_THRESHOLDS["HIGH_MIN"]]
        medium_pred = zone_data[
            (zone_data['prediction'] >= PREDICTION_THRESHOLDS["MEDIUM_MIN"]) & 
            (zone_data['prediction'] <= PREDICTION_THRESHOLDS["MEDIUM_MAX"])
        ]
        low_pred = zone_data[zone_data['prediction'] < PREDICTION_THRESHOLDS["LOW_MAX"]]
        
        high_count = len(high_pred)
        medium_count = len(medium_pred)
        low_count = len(low_pred)
        
        zone_stat = ZoneStatistics(
            zone=zone,
            high_prediction_count=high_count,
            high_prediction_percentage=round((high_count / total) * 100, 2) if total > 0 else 0,
            medium_prediction_count=medium_count,
            medium_prediction_percentage=round((medium_count / total) * 100, 2) if total > 0 else 0,
            low_prediction_count=low_count,
            low_prediction_percentage=round((low_count / total) * 100, 2) if total > 0 else 0,
            total_plantations=total,
            average_prediction=round(zone_data['prediction'].mean(), 2)
        )
        zone_stats.append(zone_stat)
    
    return zone_stats

def group_predictions_by_level(predictions_df: pd.DataFrame) -> List[PredictionGroup]:
    """แบ่งกลุ่ม predictions ตามระดับ HIGH, MEDIUM, LOW"""
    groups = []
    total_count = len(predictions_df)
    
    # แบ่งกลุ่มตามระดับ
    level_groups = {
        'HIGH': predictions_df[predictions_df['prediction'] > PREDICTION_THRESHOLDS["HIGH_MIN"]],
        'MEDIUM': predictions_df[
            (predictions_df['prediction'] >= PREDICTION_THRESHOLDS["MEDIUM_MIN"]) & 
            (predictions_df['prediction'] <= PREDICTION_THRESHOLDS["MEDIUM_MAX"])
        ],
        'LOW': predictions_df[predictions_df['prediction'] < PREDICTION_THRESHOLDS["LOW_MAX"]]
    }
    
    for level, level_data in level_groups.items():
        if len(level_data) > 0:
            # แปลงเป็น list ของ GroupedPlantPrediction
            grouped_predictions = []
            for _, row in level_data.iterrows():
                grouped_pred = GroupedPlantPrediction(
                    lat=row['lat'],
                    lon=row['lon'],
                    plant_id=row['plant_id'],
                    prediction=row['prediction'],
                    prediction_level=level,
                    ndvi=row['ndvi'],
                    ndwi=row['ndwi'],
                    gli=row['gli'],
                    precipitation=row['precipitation'],
                    zone=row['zone'],
                    cane_type=row['cane_type']
                )
                grouped_predictions.append(grouped_pred)
            
            group = PredictionGroup(
                level=level,
                count=len(level_data),
                percentage=round((len(level_data) / total_count) * 100, 2) if total_count > 0 else 0,
                average_prediction=round(level_data['prediction'].mean(), 2),
                predictions=grouped_predictions
            )
            groups.append(group)
    
    # เรียงตาม level
    level_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    groups.sort(key=lambda x: level_order.get(x.level, 3))
    
    return groups

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize Redis connection on startup"""
    await get_redis_client()

@app.on_event("shutdown")
async def shutdown_event():
    """Close Redis connection on shutdown"""
    await close_redis_client()

@app.get("/")
async def root():
    return {
        "message": "ML Prediction Service API with Redis Cache",
        "version": "1.1.0",
        "features": ["CORS enabled", "Retry mechanism", "Grouped predictions", "Redis caching"],
        "prediction_thresholds": PREDICTION_THRESHOLDS,
        "cache_config": {
            "redis_url": REDIS_URL,
            "cache_ttl": CACHE_TTL,
            "cache_prefix": CACHE_PREFIX
        }
    }

@app.get("/models")
async def get_available_models():
    """แสดงรายการ models ที่พร้อมใช้งาน"""
    try:
        models = []
        if os.path.exists(MODELS_DIR):
            for file in os.listdir(MODELS_DIR):
                if file.endswith('.json'):
                    model_name = file.replace('.json', '')
                    pkl_file = f"{model_name}.pkl"
                    pkl_path = os.path.join(MODELS_DIR, pkl_file)
                    
                    if os.path.exists(pkl_path):
                        models.append(model_name)
                    else:
                        logger.warning(f"Found {file} but missing {pkl_file}")
                        
        return {
            "available_models": models,
            "count": len(models),
            "models_directory": MODELS_DIR,
            "prediction_thresholds": PREDICTION_THRESHOLDS
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """ทำนายผลด้วย ML models (Original endpoint) with Redis cache"""
    try:
        logger.info(f"Received prediction request: {request.dict()}")
        
        # สร้าง cache key
        cache_key = generate_cache_key(
            "predict",
            year=request.year,
            start_month=request.start_month,
            end_month=request.end_month,
            models=request.models,
            zones=request.zones,
            limit=request.limit
        )
        
        # ตรวจสอบ cache ก่อน
        cached_result = await get_from_cache(cache_key)
        if cached_result and "data" in cached_result:
            logger.info("Returning cached prediction result")
            result = cached_result["data"]
            result["cached"] = True
            result["cache_key"] = cache_key
            return result
        
        # ตรวจสอบ parameters
        if request.start_month < 1 or request.start_month > 12:
            raise HTTPException(status_code=400, detail="start_month must be between 1 and 12")
        if request.end_month < 1 or request.end_month > 12:
            raise HTTPException(status_code=400, detail="end_month must be between 1 and 12")
        if request.start_month > request.end_month:
            raise HTTPException(status_code=400, detail="start_month must be less than or equal to end_month")
        
        # ดึงข้อมูลดิบด้วย retry mechanism
        try:
            raw_data = await fetch_raw_data_with_retry(
                request.year, 
                request.start_month, 
                request.end_month, 
                request.zones, 
                request.limit
            )
        except HTTPException as e:
            logger.error(f"Failed to fetch raw data: {e.detail}")
            raise HTTPException(
                status_code=503, 
                detail=f"Unable to fetch data from external API: {e.detail}. Please try again later."
            )
        
        if not raw_data:
            raise HTTPException(status_code=404, detail="No data found for the specified parameters")
        
        results = []
        
        # ประมวลผลแต่ละ model
        for model_name in request.models:
            try:
                logger.info(f"Processing model: {model_name}")
                
                # โหลด model
                model_info = load_model(model_name)
                
                # เตรียมข้อมูล
                prepared_data = prepare_data_for_model(raw_data, model_info['required_features'])
                
                # ทำนายผล
                predictions_df = predict_with_model(model_info, prepared_data)
                
                # แปลงเป็น list of objects (ไม่มี upper/lower bound)
                predictions_list = []
                for _, row in predictions_df.iterrows():
                    pred = PlantPrediction(
                        lat=row['lat'],
                        lon=row['lon'],
                        plant_id=row['plant_id'],
                        prediction=row['prediction'],
                        ndvi=row['ndvi'],
                        ndwi=row['ndwi'],
                        gli=row['gli'],
                        precipitation=row['precipitation'],
                        zone=row['zone'],
                        cane_type=row['cane_type']
                    )
                    predictions_list.append(pred)
                
                # คำนวณสถิติแต่ละ zone
                zone_stats = calculate_zone_statistics(predictions_df)
                
                # คำนวณค่าเฉลี่ยรวม
                overall_avg = predictions_df['prediction'].mean()
                
                model_result = ModelPredictionResult(
                    model_name=model_name,
                    predictions=predictions_list,
                    zone_statistics=zone_stats,
                    overall_average=round(overall_avg, 2)
                )
                
                results.append(model_result)
                logger.info(f"Successfully processed model {model_name}")
                
            except Exception as e:
                logger.error(f"Error processing model {model_name}: {str(e)}")
                continue
        
        if not results:
            raise HTTPException(status_code=500, detail="Failed to process any models")
        
        logger.info(f"Successfully processed {len(results)} models")
        
        response = PredictionResponse(
            success=True,
            message=f"Successfully processed {len(results)} models",
            results=results,
            cached=False,
            cache_key=cache_key
        )
        
        # Cache the result
        await set_to_cache(cache_key, response.dict())
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict/grouped", response_model=GroupedPredictionResponse)
async def predict_grouped(request: GroupedPredictionRequest):
    """ทำนายผลแบบจัดกลุ่มตาม prediction level (NEW endpoint) with Redis cache"""
    try:
        logger.info(f"Received grouped prediction request: {request.dict()}")
        
        # สร้าง cache key
        cache_key = generate_cache_key(
            "predict_grouped",
            year=request.year,
            start_month=request.start_month,
            end_month=request.end_month,
            models=request.models,
            zones=request.zones,
            limit=request.limit,
            group_by_level=request.group_by_level
        )
        
        # ตรวจสอบ cache ก่อน
        cached_result = await get_from_cache(cache_key)
        if cached_result and "data" in cached_result:
            logger.info("Returning cached grouped prediction result")
            result = cached_result["data"]
            result["cached"] = True
            result["cache_key"] = cache_key
            return result
        
        # ตรวจสอบ parameters
        if request.start_month < 1 or request.start_month > 12:
            raise HTTPException(status_code=400, detail="start_month must be between 1 and 12")
        if request.end_month < 1 or request.end_month > 12:
            raise HTTPException(status_code=400, detail="end_month must be between 1 and 12")
        if request.start_month > request.end_month:
            raise HTTPException(status_code=400, detail="start_month must be less than or equal to end_month")
        
        # ดึงข้อมูลดิบด้วย retry mechanism
        try:
            raw_data = await fetch_raw_data_with_retry(
                request.year, 
                request.start_month, 
                request.end_month, 
                request.zones, 
                request.limit
            )
        except HTTPException as e:
            logger.error(f"Failed to fetch raw data: {e.detail}")
            raise HTTPException(
                status_code=503, 
                detail=f"Unable to fetch data from external API: {e.detail}. Please try again later."
            )
        
        if not raw_data:
            raise HTTPException(status_code=404, detail="No data found for the specified parameters")
        
        results = []
        
        # ประมวลผลแต่ละ model
        for model_name in request.models:
            try:
                logger.info(f"Processing model: {model_name}")
                
                # โหลด model
                model_info = load_model(model_name)
                
                # เตรียมข้อมูล
                prepared_data = prepare_data_for_model(raw_data, model_info['required_features'])
                
                # ทำนายผล
                predictions_df = predict_with_model(model_info, prepared_data)
                
                # จัดกลุ่มตาม prediction level
                prediction_groups = group_predictions_by_level(predictions_df)
                
                # คำนวณสถิติแต่ละ zone
                zone_stats = calculate_zone_statistics(predictions_df)
                
                # คำนวณค่าเฉลี่ยรวม
                overall_avg = predictions_df['prediction'].mean()
                
                model_result = GroupedModelPredictionResult(
                    model_name=model_name,
                    prediction_groups=prediction_groups,
                    zone_statistics=zone_stats,
                    overall_average=round(overall_avg, 2),
                    total_predictions=len(predictions_df)
                )
                
                results.append(model_result)
                logger.info(f"Successfully processed model {model_name} - {len(predictions_df)} predictions grouped into {len(prediction_groups)} levels")
                
            except Exception as e:
                logger.error(f"Error processing model {model_name}: {str(e)}")
                continue
        
        if not results:
            raise HTTPException(status_code=500, detail="Failed to process any models")
        
        logger.info(f"Successfully processed {len(results)} models with grouping")
        
        response = GroupedPredictionResponse(
            success=True,
            message=f"Successfully processed {len(results)} models with prediction level grouping",
            results=results,
            cached=False,
            cache_key=cache_key
        )
        
        # Cache the result
        await set_to_cache(cache_key, response.dict())
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict_grouped endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/prediction-levels")
async def get_prediction_levels():
    """แสดงเกณฑ์การแบ่งระดับ prediction"""
    return {
        "prediction_thresholds": PREDICTION_THRESHOLDS,
        "levels": {
            "HIGH": f"prediction > {PREDICTION_THRESHOLDS['HIGH_MIN']}",
            "MEDIUM": f"{PREDICTION_THRESHOLDS['MEDIUM_MIN']} <= prediction <= {PREDICTION_THRESHOLDS['MEDIUM_MAX']}",
            "LOW": f"prediction < {PREDICTION_THRESHOLDS['LOW_MAX']}"
        },
        "description": "Prediction levels based on fixed thresholds"
    }

@app.get("/cache/status")
async def get_cache_status():
    """ตรวจสอบสถานะ Redis cache"""
    try:
        redis_conn = await get_redis_client()
        if redis_conn is None:
            return {
                "status": "disconnected",
                "redis_url": REDIS_URL,
                "error": "Cannot connect to Redis"
            }
        
        # Test connection
        await redis_conn.ping()
        
        # Get cache info
        info = await redis_conn.info()
        keys_count = await redis_conn.dbsize()
        
        # Get cache keys with our prefix
        our_keys = await redis_conn.keys(f"{CACHE_PREFIX}:*")
        
        return {
            "status": "connected",
            "redis_url": REDIS_URL,
            "total_keys": keys_count,
            "our_cache_keys": len(our_keys),
            "cache_prefix": CACHE_PREFIX,
            "cache_ttl": CACHE_TTL,
            "redis_info": {
                "version": info.get("redis_version"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients")
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "redis_url": REDIS_URL,
            "error": str(e)
        }

@app.delete("/cache/clear")
async def clear_cache():
    """ลบ cache ทั้งหมดที่เกี่ยวข้องกับ ML predictions"""
    try:
        deleted_count = await delete_cache_pattern(f"{CACHE_PREFIX}:*")
        return {
            "success": True,
            "message": f"Cleared {deleted_count} cache entries",
            "deleted_count": deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

@app.delete("/cache/clear/{cache_type}")
async def clear_cache_by_type(cache_type: str):
    """ลบ cache ตามประเภท (predict, predict_grouped, raw_data)"""
    try:
        if cache_type not in ["predict", "predict_grouped", "raw_data"]:
            raise HTTPException(status_code=400, detail="Invalid cache type. Must be: predict, predict_grouped, or raw_data")
        
        deleted_count = await delete_cache_pattern(f"{CACHE_PREFIX}:{cache_type}:*")
        return {
            "success": True,
            "message": f"Cleared {deleted_count} {cache_type} cache entries",
            "deleted_count": deleted_count,
            "cache_type": cache_type
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing {cache_type} cache: {str(e)}")

@app.get("/cache/keys")
async def get_cache_keys():
    """แสดงรายการ cache keys ทั้งหมด"""
    try:
        redis_conn = await get_redis_client()
        if redis_conn is None:
            raise HTTPException(status_code=503, detail="Redis not connected")
        
        keys = await redis_conn.keys(f"{CACHE_PREFIX}:*")
        
        key_info = []
        for key in keys:
            ttl = await redis_conn.ttl(key)
            key_info.append({
                "key": key,
                "ttl": ttl,
                "type": key.split(':')[1] if ':' in key else "unknown"
            })
        
        return {
            "total_keys": len(keys),
            "cache_prefix": CACHE_PREFIX,
            "keys": key_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cache keys: {str(e)}")

@app.get("/debug/raw-data")
async def debug_raw_data(year: int = 2024, start_month: int = 1, end_month: int = 3):
    """ตรวจสอบโครงสร้างข้อมูลดิบจาก API"""
    try:
        raw_data = await fetch_raw_data_with_retry(year, start_month, end_month)
        
        if not raw_data:
            return {"message": "No data found"}
        
        sample_data = raw_data[:3] if len(raw_data) > 3 else raw_data
        df = pd.DataFrame(raw_data)
        
        return {
            "total_records": len(raw_data),
            "columns": list(df.columns),
            "sample_data": sample_data,
            "data_types": df.dtypes.to_dict(),
            "unique_plants": df['plant_id'].nunique() if 'plant_id' in df.columns else "plant_id column not found",
            "unique_zones": df['zone'].unique().tolist() if 'zone' in df.columns else "zone column not found",
            "month_range": {
                "min": df['month'].min() if 'month' in df.columns else None,
                "max": df['month'].max() if 'month' in df.columns else None
            },
            "zones_count": df['zone'].value_counts().to_dict() if 'zone' in df.columns else {}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/model/{model_name}")
async def debug_model(model_name: str):
    """ตรวจสอบข้อมูล metadata ของ model"""
    try:
        model_info = load_model(model_name)
        metadata = model_info['metadata']
        required_features = model_info['required_features']
        
        months_in_features = set()
        feature_types = set()
        
        for feat in required_features:
            if '_' in feat:
                feature_type, month = feat.split('_', 1)
                months_in_features.add(month)
                feature_types.add(feature_type)
        
        return {
            "model_name": model_name,
            "metadata_keys": list(metadata.keys()),
            "required_features_count": len(required_features),
            "required_features": required_features,
            "months_in_features": sorted(list(months_in_features)),
            "feature_types": sorted(list(feature_types)),
            "sample_features": required_features[:10] if required_features else [],
            "metadata_sample": {k: str(v)[:200] + "..." if len(str(v)) > 200 else v 
                              for k, v in metadata.items() if k != 'required_features'}
        }
    except Exception as e:
        return {"error": str(e), "model_name": model_name}

@app.get("/debug/prediction-grouping")
async def debug_prediction_grouping(
    year: int = Query(2024),
    start_month: int = Query(2),
    end_month: int = Query(8),
    model: str = Query("m12"),
    limit: int = Query(100)
):
    """ตรวจสอบการจัดกลุ่ม prediction"""
    try:
        # ดึงข้อมูลตัวอย่าง
        raw_data = await fetch_raw_data_with_retry(year, start_month, end_month, limit=limit)
        
        if not raw_data:
            return {"message": "No data found"}
        
        # โหลด model
        model_info = load_model(model)
        
        # เตรียมข้อมูล
        prepared_data = prepare_data_for_model(raw_data, model_info['required_features'])
        
        # ทำนายผล
        predictions_df = predict_with_model(model_info, prepared_data)
        
        # วิเคราะห์การแจกแจงของ prediction values
        prediction_stats = {
            "total_predictions": len(predictions_df),
            "prediction_range": {
                "min": float(predictions_df['prediction'].min()),
                "max": float(predictions_df['prediction'].max()),
                "mean": float(predictions_df['prediction'].mean()),
                "median": float(predictions_df['prediction'].median()),
                "std": float(predictions_df['prediction'].std())
            },
            "level_counts": {
                "HIGH": len(predictions_df[predictions_df['prediction'] > PREDICTION_THRESHOLDS['HIGH_MIN']]),
                "MEDIUM": len(predictions_df[
                    (predictions_df['prediction'] >= PREDICTION_THRESHOLDS['MEDIUM_MIN']) & 
                    (predictions_df['prediction'] <= PREDICTION_THRESHOLDS['MEDIUM_MAX'])
                ]),
                "LOW": len(predictions_df[predictions_df['prediction'] < PREDICTION_THRESHOLDS['LOW_MAX']])
            },
            "thresholds_used": PREDICTION_THRESHOLDS,
            "sample_predictions": predictions_df[['plant_id', 'prediction', 'zone']].head(10).to_dict('records')
        }
        
        # เพิ่มเปอร์เซ็นต์
        total = prediction_stats["total_predictions"]
        for level in prediction_stats["level_counts"]:
            count = prediction_stats["level_counts"][level]
            prediction_stats["level_counts"][f"{level}_percentage"] = round((count / total) * 100, 2) if total > 0 else 0
        
        return prediction_stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint with API and Redis connectivity test"""
    try:
        # Test basic API connectivity
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{BASE_URL}/health") as response:
                api_status = "connected" if response.status == 200 else f"error_{response.status}"
    except:
        api_status = "disconnected"
    
    # Test Redis connectivity
    try:
        redis_conn = await get_redis_client()
        if redis_conn:
            await redis_conn.ping()
            redis_status = "connected"
        else:
            redis_status = "disconnected"
    except Exception as e:
        redis_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "cors_enabled": True,
        "service": "ML Prediction Service with Redis Cache",
        "version": "1.1.0",
        "features": {
            "retry_mechanism": True,
            "max_retries": MAX_RETRIES,
            "retry_delay": RETRY_DELAY,
            "timeout": 60,
            "grouped_predictions": True,
            "prediction_thresholds": PREDICTION_THRESHOLDS,
            "redis_caching": True,
            "cache_ttl": CACHE_TTL
        },
        "external_api_status": api_status,
        "external_api_url": BASE_URL,
        "redis_status": redis_status,
        "redis_url": REDIS_URL,
        "endpoints": {
            "original_predict": "/predict",
            "grouped_predict": "/predict/grouped",
            "prediction_levels": "/prediction-levels",
            "cache_status": "/cache/status",
            "cache_clear": "/cache/clear",
            "cache_keys": "/cache/keys",
            "debug_grouping": "/debug/prediction-grouping"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1256, reload=True)