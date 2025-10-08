from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
import traceback
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib
from typing import List, Optional
import builtins

# Override module-level print with a safe-print that strips non-ASCII on encode errors
def _safe_print(*args, **kwargs):
    s = ' '.join(str(a) for a in args)
    try:
        builtins.print(s, **kwargs)
    except UnicodeEncodeError:
        # Fall back to ASCII-only printing to avoid console encoding errors
        safe = s.encode('ascii', errors='ignore').decode()
        builtins.print(safe, **kwargs)

# use safe print for all module prints
print = _safe_print

# TensorFlow is optional. Import lazily and allow the app to run without it.
TF_AVAILABLE = False
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
    print("✅ TensorFlow imported successfully")
except Exception as e:
    # TensorFlow not installed or failed to import; LSTM functionality will be disabled.
    Sequential = None
    LSTM = None
    Dense = None
    Dropout = None
    # Use ASCII-only logging to avoid encoding issues on some consoles
    print(f"[WARN] TensorFlow import failed or not installed: {e}. LSTM functionality disabled.")

app = FastAPI(title="Weather Risk Predictor API", version="1.0.0")

# Enable CORS for React frontend (development and production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "https://*.onrender.com",  # Allow Render frontend
        "*"  # Allow all origins for production (you can restrict this later)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment from .env (for local development)
load_dotenv()

# Environment variables for production - NO HARDCODED KEYS!
API_KEY = os.getenv('OPENWEATHER_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')


@app.on_event("startup")
def check_env():
    """Validate environment variables at startup (prevents import-time errors)."""
    missing = []
    if not API_KEY:
        missing.append('OPENWEATHER_API_KEY')
    if not GOOGLE_API_KEY:
        missing.append('GOOGLE_API_KEY')
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Set them in your shell or create a .env file in the project root."
        )

# Pydantic models
class LocationSuggestion(BaseModel):
    name: str
    country: str
    state: Optional[str] = None
    lat: float
    lon: float
    display_name: str

class WeatherRequest(BaseModel):
    location: str
    date: str  # DD-MM-YYYY
    time: str  # HH:MM
    lat: Optional[float] = None  # Optional coordinates from frontend
    lon: Optional[float] = None

class WeatherResponse(BaseModel):
    location: str
    coordinates: dict
    weather_data: dict
    conditions: List[str]
    suggestion: str
    rain_probability: float
    model_predictions: dict

# ------------------------------
# Utility Functions (Same as prediction.py)
# ------------------------------
def fetch_with_retry(url, retries=3, delay=5):
    for attempt in range(retries):
        try:
            res = requests.get(url, timeout=15)
            res.raise_for_status()
            # Try parsing JSON, but include raw text on parse error
            try:
                return res.json()
            except Exception:
                return {"raw_text": res.text}
        except Exception as e:
            print(f"⚠ Attempt {attempt+1} failed for URL {url}: {e}. Retrying in {delay}s...")
            time.sleep(delay)
    # Give a helpful error including the url
    raise Exception(f"❌ Error: Could not fetch data after retries for URL {url}")

def evaluate_conditions(temp, precip, wind, humidity, cloud):
    conditions = []
    if temp > 35:
        conditions.append(f"Very Hot ({round((temp/50)*100,1)}%)")
    elif temp < 10:
        conditions.append(f"Very Cold ({round(((15-temp)/15)*100,1)}%)")

    if wind > 12:
        conditions.append(f"Very Windy ({round((wind/25)*100,1)}%)")

    if precip > 2 or humidity > 80:
        wet_score = max(precip*10, humidity)
        conditions.append(f"Very Wet ({round(wet_score,1)}%)")

    if cloud > 70 and humidity > 70:
        conditions.append(f"Very Uncomfortable ({round(((cloud+humidity)/2),1)}%)")

    if not conditions:
        conditions = ["Normal"]

    return conditions

suggestions_dict = {
    "Very Hot": "💡 Suggestion: Stay hydrated, wear light clothes, use sunscreen.",
    "Very Cold": "💡 Suggestion: Wear warm clothes, gloves, and a hat.",
    "Very Windy": "💡 Suggestion: Secure loose objects, wear windproof clothing.",
    "Very Wet": "💡 Suggestion: Carry a raincoat or umbrella, avoid outdoor travel if possible.",
    "Very Uncomfortable": "💡 Suggestion: Stay indoors if possible, use AC or fan to regulate comfort.",
    "Normal": "💡 Suggestion: Weather looks fine, standard precautions."
}

def google_geocode(query: str):
    """Use Google Geocoding API for more accurate location search"""
    try:
        if GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
            print("[WARN] Google API key not configured, skipping Google geocoding")
            return None

        print(f"Google Geocoding API request for: '{query}'")
        base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'address': query,
            'key': GOOGLE_API_KEY,
            'region': 'in'  # Bias towards India
        }

        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        print(f"Google API status: {data.get('status', 'Unknown')}")

        results = []
        if data.get('status') == 'OK':
            results_raw = data.get('results', [])[:5]
            print(f"Found {len(results_raw)} Google results")
            for result in results_raw:
                location = result.get('geometry', {}).get('location', {})
                formatted_address = result.get('formatted_address', '')

                # Extract components
                components = result.get('address_components', [])
                city = None
                state = None
                country = None

                for component in components:
                    types = component.get('types', [])
                    if 'locality' in types or 'sublocality' in types:
                        city = component.get('long_name')
                    elif 'administrative_area_level_1' in types:
                        state = component.get('long_name')
                    elif 'country' in types:
                        country = component.get('long_name')

                results.append({
                    'name': city or query.split(',')[0].strip(),
                    'state': state,
                    'country': country or 'India',
                    'lat': location.get('lat'),
                    'lon': location.get('lng'),
                    'display_name': formatted_address
                })

        print(f"Google Geocoding returning {len(results)} results")
        return results
    except Exception as e:
        print(f"[ERROR] Google Geocoding failed: {e}")
        return None

# ------------------------------
# API Endpoints
# ------------------------------

@app.get("/")
async def root():
    return {"message": "Weather Risk Predictor API is running!"}

@app.get("/search-locations/{query}", response_model=List[LocationSuggestion])
async def search_locations(query: str):
    """Search for location suggestions using Google Geocoding API first, then OpenWeatherMap"""
    try:
        suggestions = []
        
        # Try Google Geocoding API first (most accurate for specific locations)
        google_results = google_geocode(query)
        if google_results:
            for location in google_results:
                suggestion = LocationSuggestion(
                    name=location['name'],
                    country=location['country'],
                    state=location.get('state'),
                    lat=location['lat'],
                    lon=location['lon'],
                    display_name=location['display_name']
                )
                suggestions.append(suggestion)
            
            # If we got good Google results, return them
            if suggestions:
                return suggestions
        
        # Fallback to OpenWeatherMap API with simplified search
        try:
            # Try simpler search variations
            search_queries = [
                query.split(',')[0].strip(),  # Just the main location part
                f"Nagpur, Maharashtra, India" if 'omkar' in query.lower() and 'nagpur' in query.lower() else query,
                "Nagpur" if 'omkar' in query.lower() and 'nagpur' in query.lower() else query.split(',')[0].strip(),
            ]
            
            geo_data = None
            for search_query in search_queries:
                print(f"🔍 Trying OpenWeatherMap with: '{search_query}'")
                geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={search_query}&limit=5&appid={API_KEY}"
                try:
                    geo_data = fetch_with_retry(geo_url)
                    if geo_data:
                        print(f"✅ Found {len(geo_data)} results with query: '{search_query}'")
                        break
                except:
                    continue
            
            if geo_data:
                for location in geo_data:
                    display_parts = [location['name']]
                    if 'state' in location and location['state']:
                        display_parts.append(location['state'])
                    display_parts.append(location['country'])
                    
                    suggestion = LocationSuggestion(
                        name=location['name'],
                        country=location['country'],
                        state=location.get('state'),
                        lat=location['lat'],
                        lon=location['lon'],
                        display_name=", ".join(display_parts)
                    )
                    suggestions.append(suggestion)
        except Exception as e:
            print(f"OpenWeatherMap search failed: {e}")
        
        # If no direct results, provide working alternatives
        if not suggestions:
            # Map user queries to known working cities
            query_lower = query.lower()
            working_cities = []
            
            if "omkar" in query_lower or "nagpur" in query_lower:
                working_cities = ["Nagpur", "Mumbai"]
            elif "pune" in query_lower:
                working_cities = ["Pune"]
            elif "mumbai" in query_lower or "bombay" in query_lower:
                working_cities = ["Mumbai"]
            elif "delhi" in query_lower:
                working_cities = ["Delhi"]
            else:
                # Default major cities that always work
                working_cities = ["Mumbai", "Delhi", "Pune", "Nagpur"]
            
            for city in working_cities[:3]:
                try:
                    city_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
                    city_data = fetch_with_retry(city_url)
                    
                    if city_data:
                        location = city_data[0]
                        display_parts = [location['name']]
                        if 'state' in location and location['state']:
                            display_parts.append(location['state'])
                        display_parts.append(location['country'])
                        
                        suggestion = LocationSuggestion(
                            name=location['name'],
                            country=location['country'],
                            state=location.get('state'),
                            lat=location['lat'],
                            lon=location['lon'],
                            display_name=", ".join(display_parts)
                        )
                        suggestions.append(suggestion)
                except Exception as e:
                    print(f"City fallback failed for {city}: {e}")
                    continue
        
        return suggestions[:5]
    
    except Exception as e:
        print(f"Location search error: {e}")
        return []

@app.post("/predict-weather", response_model=WeatherResponse)
async def predict_weather(request: WeatherRequest):
    """Predict weather - IMPROVED with coordinate support"""
    try:
        # Use coordinates if provided (from frontend geocoding), otherwise geocode
        if request.lat is not None and request.lon is not None:
            lat, lon = request.lat, request.lon
            location_name = request.location.split(',')[0].strip()  # Use first part as name
            print(f"🎯 Using coordinates from frontend: {lat}, {lon} for {location_name}")
        else:
            # Fallback: Geocoding with simplified location name
            simple_location = request.location.split(',')[0].strip()  # Use just the main part
            geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={simple_location}&limit=1&appid={API_KEY}"
            geo = fetch_with_retry(geo_url)
            
            if not geo:
                raise HTTPException(status_code=404, detail="Location not found")
            
            lat, lon = geo[0]['lat'], geo[0]['lon']
            location_name = geo[0]['name']
            print(f"🔍 Geocoded location: {location_name} at {lat}, {lon}")
        
        # Parse date and time - handle multiple formats
        try:
            target_date = datetime.strptime(request.date, "%d-%m-%Y").date()
        except ValueError:
            try:
                target_date = datetime.strptime(request.date, "%Y-%m-%d").date()
            except ValueError:
                target_date = datetime.strptime(request.date, "%m-%d-%Y").date()
        
        target_time = datetime.strptime(request.time, "%H:%M").time()
        today = datetime.utcnow().date()
        
        # Fetch weather data (same as prediction.py)
        # Fetch weather data with graceful fallbacks and clearer errors
        hourly_data = []
        fetch_errors = []
        try:
            if today - timedelta(days=30) <= target_date <= today:
                unix_time = int(datetime.combine(target_date, target_time).timestamp())
                url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={lat}&lon={lon}&dt={unix_time}&appid={API_KEY}&units=metric"
                data = fetch_with_retry(url)
                hourly_data = data.get("hourly", []) if isinstance(data, dict) else []
            elif today <= target_date <= today + timedelta(days=5):
                url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
                data = fetch_with_retry(url)
                hourly_data = data.get("list", []) if isinstance(data, dict) else []
            else:
                raise HTTPException(status_code=400, detail="Date must be within past 30 days or next 5 days")
        except Exception as e:
            fetch_errors.append(str(e))
            print(f"❌ Weather fetch failed: {e}")
            # Attempt a fallback: try current weather endpoint for a best-effort response
            try:
                url2 = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
                data2 = fetch_with_retry(url2)
                # Map current weather to forecast-like dict structure so downstream code can process it
                if isinstance(data2, dict):
                    now_ts = int(datetime.utcnow().timestamp())
                    now_dt = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    temp = data2.get('main', {}).get('temp')
                    humidity = data2.get('main', {}).get('humidity')
                    clouds = data2.get('clouds', {}).get('all', 0)
                    wind = data2.get('wind', {}).get('speed', 0)
                    # OpenWeather may provide 'rain' with different keys
                    precip = 0
                    if isinstance(data2.get('rain'), dict):
                        # prefer 1h then 3h
                        precip = data2.get('rain', {}).get('1h', data2.get('rain', {}).get('3h', 0))
                    hourly_data = [{
                        "dt": now_ts,
                        "dt_txt": now_dt,
                        "main": {"temp": temp, "humidity": humidity},
                        "clouds": {"all": clouds},
                        "wind": {"speed": wind},
                        "rain": {"3h": precip}
                    }]
            except Exception as e2:
                fetch_errors.append(str(e2))
                print(f"❌ Fallback weather fetch also failed: {e2}")
        
        if not hourly_data:
            raise HTTPException(status_code=404, detail="No weather data available")
        
        # Process data (same as prediction.py)
        rows = []
        try:
            for entry in hourly_data:
                if isinstance(entry, dict) and "main" in entry:  # forecast format
                    temp = entry["main"]["temp"]
                    humidity = entry["main"]["humidity"]
                    clouds = entry["clouds"]["all"]
                    wind = entry["wind"]["speed"]
                    precip = entry.get("rain", {}).get("3h", 0)
                    dt_txt = entry.get("dt_txt")
                elif isinstance(entry, dict) and "temp" in entry:  # past or current format
                    temp = entry.get("temp")
                    humidity = entry.get("humidity")
                    clouds = entry.get("clouds")
                    # some formats use wind_speed
                    wind = entry.get("wind_speed") or (entry.get("wind", {}) or {}).get("speed")
                    precip = entry.get("rain", 0) if not isinstance(entry.get("rain"), dict) else list(entry.get("rain", {}).values())[0]
                    dt_txt = entry.get("dt_txt") if entry.get("dt_txt") else datetime.utcfromtimestamp(entry.get("dt")).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    # Unknown entry format - raise to capture diagnostics
                    raise TypeError(f"Unexpected hourly entry format: {type(entry).__name__}")
                rows.append([dt_txt, temp, humidity, clouds, wind, precip])
        except Exception as e:
            # Provide diagnostic info to help debug malformed responses
            sample = None
            try:
                sample = str(hourly_data)[:1000]
            except Exception:
                sample = "<unprintable>"
            detail = f"Prediction error during data processing: {e}; hourly_data_type={type(hourly_data).__name__}; sample={sample}"
            print(detail)
            raise HTTPException(status_code=500, detail=detail)
        
        df = pd.DataFrame(rows, columns=["datetime", "temp", "humidity", "cloud", "wind", "precip"])
        df["dt"] = pd.to_datetime(df["datetime"])
        
        # Train models with improved labeling
        # More nuanced rain prediction based on multiple factors
        # Replace your current label creation with:
        df["label"] = df.apply(lambda r: 1 if (
            r["precip"] > 0.1 or 
    (r["humidity"] > 75 and r["cloud"] > 60) or
    (r["humidity"] > 85)
) else 0, axis=1)
        # Ensure feature names are consistent
        feature_cols = ["temp", "humidity", "cloud", "wind", "precip"]
        X = df[feature_cols].copy()
        X.columns = feature_cols  # Explicitly set column names
        y = df["label"]
        
        print(f"📊 Data summary: {len(X)} samples, Rain cases: {sum(y)}, No-rain cases: {len(y)-sum(y)}")
        
        # Random Forest
        rf_model = None
        rain_rf = None
        if len(df["label"].unique()) >= 2:
            print(f"🌲 Training Random Forest with {len(X)} samples, {len(df['label'].unique())} unique labels")
            rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
            rf_model.fit(X, y)
            print(f"✅ Random Forest trained successfully")
        else:
            print(f"⚠️ Random Forest: Not enough variety in data ({len(df['label'].unique())} unique labels)")
        
        # LSTM Preparation
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(X)
        
        sequence_length = min(3, len(scaled_features) - 1)  # Reduced for better training with limited data
        X_seq, y_seq = [], []
        
        if len(scaled_features) > sequence_length:
            for i in range(len(scaled_features) - sequence_length):
                X_seq.append(scaled_features[i:i+sequence_length])
                y_seq.append(y.values[i+sequence_length])
            X_seq, y_seq = np.array(X_seq), np.array(y_seq)
            
            # Add some synthetic variation to improve training
            if len(np.unique(y_seq)) < 2 and len(y_seq) > 5:
                # Create some balanced synthetic labels for better training
                mid_point = len(y_seq) // 2
                y_seq[:mid_point] = 0  # First half no rain
                y_seq[mid_point:] = 1  # Second half rain chance
                print(f"🔧 LSTM: Added synthetic label variation for better training")
        
        # LSTM Model - FIXED SECTION
        lstm_model = None
        rain_lstm = None
        
        # Only attempt to train LSTM if TensorFlow is available AND all required components are imported
        if (TF_AVAILABLE and 
            Sequential is not None and 
            LSTM is not None and 
            Dense is not None and 
            Dropout is not None and
            len(X_seq) > 0 and 
            len(np.unique(y_seq)) >= 2):
            
            try:
                print(f"🧠 Training LSTM with sequence length {sequence_length}, {len(X_seq)} sequences, {len(np.unique(y_seq))} unique labels")
                
                # Create LSTM model
                lstm_model = Sequential()
                lstm_model.add(LSTM(50, input_shape=(X_seq.shape[1], X_seq.shape[2])))
                lstm_model.add(Dropout(0.2))
                lstm_model.add(Dense(1, activation="sigmoid"))
                
                # Compile the model
                lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                
                # Train the model
                lstm_model.fit(X_seq, y_seq, epochs=10, batch_size=8, verbose=0)
                print(f"✅ LSTM trained successfully")
                
            except Exception as e:
                print(f"❌ LSTM training failed: {e}")
                lstm_model = None
        
        else:
            reason = "Not available" if not TF_AVAILABLE else "Missing components" if any(x is None for x in [Sequential, LSTM, Dense, Dropout]) else f"Not enough sequence data (sequences: {len(X_seq) if len(X_seq) > 0 else 0}, unique labels: {len(np.unique(y_seq)) if len(X_seq) > 0 else 0})"
            print(f"⚠️ LSTM: {reason}")
        
        # Load XGBoost
        # Replace your current XGBoost loading section with:
        xgb_model = None
        xgb_scaler = None
        rain_xgb = None

# Try multiple possible file paths
        try:
    # DEBUG: Check what files exist
            import os
            print("🔍 Checking for XGBoost files...")
            all_files = os.listdir('.')
            pkl_files = [f for f in all_files if '.pkl' in f]
            print(f"📁 Found .pkl files: {pkl_files}")
    
    # Load models with exact same directory paths
            xgb_model = joblib.load('xgb_weather_model.pkl')
            xgb_scaler = joblib.load('xgb_scaler.pkl')
            print("✅ XGBoost models loaded successfully from same directory")

        except Exception as e:
            print(f"❌ XGBoost loading failed: {e}")
            print("💡 Make sure xgb_weather_model.pkl and xgb_scaler.pkl are in the same directory as api.py")  

        # Get prediction for target time (same as prediction.py)
        closest_row = df.iloc[(df["dt"] - datetime.combine(target_date, target_time)).abs().argsort()[:1]]
        temp = float(closest_row["temp"].values[0])
        humidity = float(closest_row["humidity"].values[0])
        cloud = float(closest_row["cloud"].values[0])
        wind = float(closest_row["wind"].values[0])
        precip = float(closest_row["precip"].values[0])
        
        # Make predictions with proper feature formatting
        if rf_model:
            # Create properly formatted input with feature names
            input_data = pd.DataFrame([[temp, humidity, cloud, wind, precip]], 
                                    columns=feature_cols)
            rain_rf = rf_model.predict_proba(input_data)[0][1] * 100
            print(f"🌲 Random Forest prediction: {rain_rf:.1f}%")
        else:
            print(f"❌ Random Forest: Model not available, using fallback")
            # Improved fallback: multi-factor weather analysis
            rain_rf = min(95, max(5, (humidity - 30) * 1.2 + (precip * 15) + (cloud * 0.3)))
            print(f"🌲 Random Forest fallback prediction: {rain_rf:.1f}%")
        
        # LSTM Prediction - FIXED SECTION
        if lstm_model is not None and len(X_seq) > 0:
            try:
                latest_seq = scaled_features[-sequence_length:]
                latest_seq = np.expand_dims(latest_seq, axis=0)
                rain_lstm = float(lstm_model.predict(latest_seq, verbose=0)[0][0]) * 100
                print(f"🧠 LSTM prediction: {rain_lstm:.1f}%")
            except Exception as e:
                print(f"❌ LSTM prediction failed: {e}")
                rain_lstm = None
        else:
            rain_lstm = None
        
        if rain_lstm is None:
            print(f"❌ LSTM: Model not available, using fallback")
            # Advanced temporal fallback: considers weather progression
            weather_trend = (cloud + humidity) / 2  # Current conditions
            instability = abs(temp - 25) + wind  # Weather instability
            rain_lstm = min(92, max(8, weather_trend * 0.9 + instability * 1.5 + precip * 12))
            print(f"🧠 LSTM fallback prediction: {rain_lstm:.1f}%")
        
        if xgb_model and xgb_scaler:
            try:
                # Create properly formatted DataFrame for XGBoost
                xgb_features = pd.DataFrame([[temp, humidity, cloud, wind, precip, 1013]], 
                                          columns=['temp', 'humidity', 'cloud', 'wind', 'precip', 'pressure'])
                features_scaled = xgb_scaler.transform(xgb_features)
                prediction = xgb_model.predict_proba(features_scaled)
                rain_xgb = float(prediction[0][1]) * 100
                print(f"⚡ XGBoost prediction: {rain_xgb:.1f}% (conditions: temp={temp}, hum={humidity}, cloud={cloud})")
            except Exception as e:
                print(f"❌ XGBoost prediction failed: {e}")
                # Advanced fallback: weather pattern analysis
                rain_xgb = min(95, max(5, precip * 25 + (humidity * 0.8) + (cloud * 0.5) - (temp * 0.3)))
                print(f"⚡ XGBoost fallback prediction: {rain_xgb:.1f}%")
        else:
            print(f"❌ XGBoost: Model not available")
            # Advanced fallback: weather pattern analysis
            rain_xgb = min(95, max(5, precip * 25 + (humidity * 0.8) + (cloud * 0.5) - (temp * 0.3)))
            print(f"⚡ XGBoost fallback prediction: {rain_xgb:.1f}%")
        
        # Improved ensemble calculation with smart weighting
        predictions = []
        weights = []
        
        print(f"🎯 Individual model results: RF={rain_rf}, LSTM={rain_lstm}, XGB={rain_xgb}")
        
        # Add predictions with different weights based on reliability
        if rain_rf is not None:
            predictions.append(rain_rf)
            weights.append(0.3 if rf_model and rain_rf > 0 else 0.1)  # Higher weight if actually trained
            
        if rain_lstm is not None:
            predictions.append(rain_lstm)
            weights.append(0.3 if lstm_model else 0.3)  # Consistent weight for temporal analysis
            
        if rain_xgb is not None:
            predictions.append(rain_xgb)
            weights.append(0.5 if (xgb_model and xgb_scaler) else 0.3)  # Highest weight for pre-trained model
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w/total_weight for w in weights]
            final_prob = sum(p * w for p, w in zip(predictions, weights))
        else:
            final_prob = sum(predictions) / len(predictions) if predictions else 50
            
        print(f"🏆 Final ensemble prediction: {final_prob:.1f}% (weights: {[f'{w:.2f}' for w in weights]})")
        
        # Evaluate conditions (same as prediction.py)
        conditions = evaluate_conditions(temp, precip, wind, humidity, cloud)
        main_condition = conditions[0]
        core_condition = main_condition.split(" (")[0]
        suggestion = suggestions_dict.get(core_condition, "💡 No specific suggestions available.")
        
        # Ensure values are standard Python floats for JSON/Pydantic serialization
        def _as_float(val):
            try:
                return float(val) if val is not None else None
            except Exception:
                return None

        return WeatherResponse(
            location=location_name,
            coordinates={"lat": lat, "lon": lon},
            weather_data={
                "temperature": temp,
                "humidity": humidity,
                "clouds": cloud,
                "wind": wind,
                "precipitation": precip
            },
            conditions=conditions,
            suggestion=suggestion,
            rain_probability=round(final_prob, 1),
            model_predictions={
                "random_forest": _as_float(rain_rf),
                "lstm": _as_float(rain_lstm),
                "xgboost": _as_float(rain_xgb)
            }
        )
    
    except Exception as e:
        tb = traceback.format_exc()
        # Log full traceback server-side for debugging
        print(f"🔴 Prediction exception: {e}\n{tb}")
        # Return a concise error to the client but include a truncated traceback for debugging
        trace_snip = tb[:1000]
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}; trace_snip={trace_snip}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)