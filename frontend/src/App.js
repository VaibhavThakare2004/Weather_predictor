import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { 
  WiDaySunny, 
  WiRain, 
  WiCloudy, 
  WiStrongWind, 
  WiHumidity, 
  WiThermometer
} from 'react-icons/wi';
import { 
  FaMapMarkerAlt, 
  FaCalendarAlt, 
  FaClock, 
  FaRobot, 
  FaBrain, 
  FaTree, 
  FaBolt,
  FaSearch,
  FaSpinner
} from 'react-icons/fa';

import { BsStars, BsLightning } from 'react-icons/bs';
import './App.css';

// Use environment variable for API URL, fallback to localhost for development
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8002';

// Floating particles component
const FloatingParticles = () => {
  return (
    <div className="floating-particles">
      {[...Array(15)].map((_, i) => (
        <motion.div
          key={i}
          className="particle"
          initial={{ 
            x: Math.random() * window.innerWidth,
            y: Math.random() * window.innerHeight,
            opacity: 0 
          }}
          animate={{ 
            y: [null, -20, 20],
            x: [null, Math.random() * 100],
            opacity: [0, 0.6, 0] 
          }}
          transition={{ 
            duration: Math.random() * 10 + 5,
            repeat: Infinity,
            delay: Math.random() * 5
          }}
        />
      ))}
    </div>
  );
};

// Weather icon mapper
const getWeatherIcon = (condition, size = "2rem") => {
  const iconMap = {
    'Very Hot': <WiDaySunny size={size} color="#ff6b35" />,
    'Very Cold': <WiThermometer size={size} color="#4fc3f7" />,
    'Very Windy': <WiStrongWind size={size} color="#9c27b0" />,
    'Very Wet': <WiRain size={size} color="#2196f3" />,
    'Very Uncomfortable': <WiCloudy size={size} color="#607d8b" />,
    'Normal': <WiDaySunny size={size} color="#4caf50" />
  };
  
  return iconMap[condition] || <WiDaySunny size={size} color="#4caf50" />;
};

function App() {
  const [locationQuery, setLocationQuery] = useState('');
  const [locationSuggestions, setLocationSuggestions] = useState([]);
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [date, setDate] = useState('');
  const [time, setTime] = useState('');
  const [weatherResult, setWeatherResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  
  const suggestionsRef = useRef(null);
  const searchTimeoutRef = useRef(null);

  // Search for location suggestions with debouncing
  useEffect(() => {
    if (locationQuery.length > 2) {
      setIsSearching(true);
      
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
      
      searchTimeoutRef.current = setTimeout(async () => {
        try {
          const response = await axios.get(`${API_BASE_URL}/search-locations/${encodeURIComponent(locationQuery)}`);
          setLocationSuggestions(response.data);
          setShowSuggestions(true);
        } catch (err) {
          console.error('Error fetching location suggestions:', err);
          setLocationSuggestions([]);
        } finally {
          setIsSearching(false);
        }
      }, 300);
    } else {
      setLocationSuggestions([]);
      setShowSuggestions(false);
      setIsSearching(false);
    }

    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
    };
  }, [locationQuery]);

  // Close suggestions when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (suggestionsRef.current && !suggestionsRef.current.contains(event.target)) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleLocationSelect = (location) => {
    setSelectedLocation(location);
    setLocationQuery(location.display_name);
    setShowSuggestions(false);
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    
    // Check if user typed a location but didn't select from dropdown
    if (!selectedLocation && locationQuery.length > 2) {
      // Auto-search and select first suggestion if available
      try {
        const response = await axios.get(`${API_BASE_URL}/search-locations/${encodeURIComponent(locationQuery)}`);
        if (response.data && response.data.length > 0) {
          setSelectedLocation(response.data[0]);
          setLocationQuery(response.data[0].display_name);
        } else {
          setError('Please select a location from the suggestions ✨');
          return;
        }
      } catch (err) {
        setError('Could not find location. Please try again 🔍');
        return;
      }
    }
    
    if (!selectedLocation || !date || !time) {
      setError('Please fill in all fields ✨');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const dateObj = new Date(date);
      
      // Validate date
      if (isNaN(dateObj.getTime())) {
        setError('Please enter a valid date 📅');
        return;
      }
      
      const formattedDate = `${String(dateObj.getDate()).padStart(2, '0')}-${String(dateObj.getMonth() + 1).padStart(2, '0')}-${dateObj.getFullYear()}`;
      
      console.log('Sending request:', {
        location: selectedLocation.display_name,
        date: formattedDate,
        time: time,
        lat: selectedLocation.lat,
        lon: selectedLocation.lon
      });
      
      const response = await axios.post(`${API_BASE_URL}/predict-weather`, {
        location: selectedLocation.display_name,
        date: formattedDate,
        time: time,
        lat: selectedLocation.lat,
        lon: selectedLocation.lon
      });
      
      setWeatherResult(response.data);
    } catch (err) {
      console.error('Prediction error:', err);
      const errorMessage = err.response?.data?.detail || err.message || 'Something went wrong! Please try again 🔄';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const getRiskLevel = (probability) => {
    if (probability < 20) return { 
      level: 'Low', 
      color: 'linear-gradient(135deg, #4ade80, #10b981)', 
      icon: '🌤️',
      bgColor: 'rgba(74, 222, 128, 0.1)'
    };
    if (probability < 50) return { 
      level: 'Medium', 
      color: 'linear-gradient(135deg, #fbbf24, #f59e0b)', 
      icon: '⛅',
      bgColor: 'rgba(251, 191, 36, 0.1)'
    };
    if (probability < 70) return { 
      level: 'High', 
      color: 'linear-gradient(135deg, #f97316, #ea580c)', 
      icon: '🌦️',
      bgColor: 'rgba(249, 115, 22, 0.1)'
    };
    return { 
      level: 'Very High', 
      color: 'linear-gradient(135deg, #ef4444, #dc2626)', 
      icon: '🌧️',
      bgColor: 'rgba(239, 68, 68, 0.1)'
    };
  };

  const formatDate = () => {
    const today = new Date();
    const thirtyDaysAgo = new Date(today.getTime() - (30 * 24 * 60 * 60 * 1000));
    const fiveDaysFromNow = new Date(today.getTime() + (5 * 24 * 60 * 60 * 1000));
    
    const formatDateString = (date) => {
      return date.toISOString().split('T')[0];
    };
    
    return {
      min: formatDateString(thirtyDaysAgo),
      max: formatDateString(fiveDaysFromNow)
    };
  };

  const dateRange = formatDate();

  return (
    <div className="App">
      <FloatingParticles />
      
      <div className="container">
        {/* Hero Section */}
        <motion.div 
          className="hero-section"
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          <motion.div 
            className="hero-icon"
            animate={{ 
              rotate: [0, 10, -10, 0],
              scale: [1, 1.1, 1]
            }}
            transition={{ 
              duration: 4,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          >
            🌦️
          </motion.div>
          <motion.h1 
            className="hero-title"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3, duration: 0.8 }}
          >
            Weather Risk Predictor
          </motion.h1>
          <motion.p 
            className="hero-subtitle"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.8 }}
          >
            <BsStars className="star-icon" />
            AI-powered weather prediction with stunning accuracy
            <BsStars className="star-icon" />
          </motion.p>
        </motion.div>
        
        {/* Prediction Form */}
        <motion.div 
          className="prediction-form-container"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          <form onSubmit={handlePredict} className="prediction-form">
            {/* Location Search */}
            <motion.div 
              className="form-group location-search"
              ref={suggestionsRef}
              whileHover={{ scale: 1.02 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <label htmlFor="location">
                <FaMapMarkerAlt className="form-icon" />
                Location
                {!selectedLocation && locationQuery.length > 2 && (
                  <small style={{ color: '#fbbf24', marginLeft: '10px' }}>
                    👆 Please select from suggestions below
                  </small>
                )}
                {selectedLocation && (
                  <small style={{ color: '#4ade80', marginLeft: '10px' }}>
                    ✓ Location selected
                  </small>
                )}
              </label>
              <div className="input-container">
                <input
                  type="text"
                  id="location"
                  value={locationQuery}
                  onChange={(e) => {
                    setLocationQuery(e.target.value);
                    setSelectedLocation(null);
                  }}
                  onBlur={() => {
                    // If user didn't select from dropdown but typed something valid, auto-search
                    if (!selectedLocation && locationQuery.length > 2) {
                      setTimeout(async () => {
                        try {
                          const response = await axios.get(`${API_BASE_URL}/search-locations/${encodeURIComponent(locationQuery)}`);
                          if (response.data && response.data.length > 0) {
                            setSelectedLocation(response.data[0]);
                            setLocationQuery(response.data[0].display_name);
                          }
                        } catch (err) {
                          console.log('Auto-search failed');
                        }
                      }, 500);
                    }
                  }}
                  placeholder="Search for area/city (e.g., Pune, Omkar Nagar)"
                  required
                  style={{
                    borderColor: selectedLocation ? '#4ade80' : locationQuery.length > 2 ? '#fbbf24' : 'rgba(102, 126, 234, 0.2)',
                    background: selectedLocation ? 'rgba(74, 222, 128, 0.1)' : 'rgba(255,255,255,0.8)'
                  }}
                />
                <div className="input-icon">
                  {isSearching ? (
                    <FaSpinner className="spinning" />
                  ) : selectedLocation ? (
                    <span style={{ color: '#4ade80', fontSize: '1.2rem' }}>✓</span>
                  ) : (
                    <FaSearch />
                  )}
                </div>
              </div>
              
              <AnimatePresence>
                {showSuggestions && (
                  <motion.div 
                    className="suggestions-dropdown"
                    initial={{ opacity: 0, y: -10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -10, scale: 0.95 }}
                    transition={{ duration: 0.2 }}
                  >
                    {locationSuggestions.length > 0 ? (
                      locationSuggestions.map((location, index) => (
                        <motion.div
                          key={index}
                          className="suggestion-item"
                          onClick={() => handleLocationSelect(location)}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.1 }}
                          whileHover={{ 
                            backgroundColor: "rgba(102, 126, 234, 0.1)",
                            x: 5
                          }}
                        >
                          <div className="suggestion-main">
                            <FaMapMarkerAlt className="suggestion-icon" />
                            {location.name}
                          </div>
                          <div className="suggestion-details">
                            {location.state && `${location.state}, `}{location.country}
                          </div>
                        </motion.div>
                      ))
                    ) : (
                      <div className="suggestion-item no-results">
                        <div className="suggestion-main">
                          🔍 Try: "Nagpur", "Mumbai", "Delhi", or "Pune"
                        </div>
                        <div className="suggestion-details">
                          No results found for "{locationQuery}"
                        </div>
                      </div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* Date and Time Row */}
            <div className="form-row">
              <motion.div 
                className="form-group"
                whileHover={{ scale: 1.02 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <label htmlFor="date">
                  <FaCalendarAlt className="form-icon" />
                  Date
                </label>
                <input
                  type="date"
                  id="date"
                  value={date}
                  onChange={(e) => setDate(e.target.value)}
                  min={dateRange.min}
                  max={dateRange.max}
                  required
                />
                <small>Past 30 days to next 5 days</small>
              </motion.div>

              <motion.div 
                className="form-group"
                whileHover={{ scale: 1.02 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <label htmlFor="time">
                  <FaClock className="form-icon" />
                  Time (Whole Hours Only)
                </label>
                <select
                  id="time"
                  value={time}
                  onChange={(e) => setTime(e.target.value)}
                  required
                  className="time-select"
                >
                  <option value="">Select Time</option>
                  <option value="00:00">12:00 AM</option>
                  <option value="01:00">1:00 AM</option>
                  <option value="02:00">2:00 AM</option>
                  <option value="03:00">3:00 AM</option>
                  <option value="04:00">4:00 AM</option>
                  <option value="05:00">5:00 AM</option>
                  <option value="06:00">6:00 AM</option>
                  <option value="07:00">7:00 AM</option>
                  <option value="08:00">8:00 AM</option>
                  <option value="09:00">9:00 AM</option>
                  <option value="10:00">10:00 AM</option>
                  <option value="11:00">11:00 AM</option>
                  <option value="12:00">12:00 PM</option>
                  <option value="13:00">1:00 PM</option>
                  <option value="14:00">2:00 PM</option>
                  <option value="15:00">3:00 PM</option>
                  <option value="16:00">4:00 PM</option>
                  <option value="17:00">5:00 PM</option>
                  <option value="18:00">6:00 PM</option>
                  <option value="19:00">7:00 PM</option>
                  <option value="20:00">8:00 PM</option>
                  <option value="21:00">9:00 PM</option>
                  <option value="22:00">10:00 PM</option>
                  <option value="23:00">11:00 PM</option>
                </select>
                <small>Select time in whole hours only</small>
              </motion.div>
            </div>

            <motion.button 
              type="submit" 
              disabled={loading} 
              className="predict-button"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              {loading ? (
                <>
                  <FaSpinner className="spinning" />
                  Predicting Magic...
                </>
              ) : (
                <>
                  <BsLightning />
                  Predict Weather Risk
                </>
              )}
            </motion.button>
          </form>
        </motion.div>

        {/* Error Message */}
        <AnimatePresence>
          {error && (
            <motion.div 
              className="error-message"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              ⚠️ {error}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Weather Results */}
        <AnimatePresence>
          {weatherResult && (
            <motion.div 
              className="weather-result"
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -50 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              {/* Results Header */}
              <motion.div 
                className="result-header"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
              >
                <h2>Weather Prediction Results</h2>
                <div className="location-info">
                  <FaMapMarkerAlt />
                  <span>{weatherResult.location}</span>
                  <small>
                    {weatherResult.coordinates.lat.toFixed(4)}, {weatherResult.coordinates.lon.toFixed(4)}
                  </small>
                </div>
              </motion.div>

              {/* Main Risk Card */}
              <motion.div 
                className="main-risk-card"
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.6, type: "spring" }}
                style={{ 
                  background: getRiskLevel(weatherResult.rain_probability).bgColor,
                  border: `2px solid ${getRiskLevel(weatherResult.rain_probability).color.split(',')[0].replace('linear-gradient(135deg, ', '')}`
                }}
              >
                <div className="risk-content">
                  <div className="risk-icon-large">
                    {getRiskLevel(weatherResult.rain_probability).icon}
                  </div>
                  <div className="risk-details">
                    <div 
                      className="probability-number"
                      style={{ 
                        background: getRiskLevel(weatherResult.rain_probability).color,
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent'
                      }}
                    >
                      {weatherResult.rain_probability}%
                    </div>
                    <div className="risk-level">
                      Rain Probability - {getRiskLevel(weatherResult.rain_probability).level} Risk
                    </div>
                  </div>
                </div>
              </motion.div>

              {/* Weather Data Cards Grid */}
              <div className="weather-cards-grid">
                <motion.div 
                  className="weather-card"
                  initial={{ opacity: 0, x: -50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.8 }}
                >
                  <div className="card-header">
                    <WiThermometer size="2rem" color="#ff6b35" />
                    <h4>Temperature</h4>
                  </div>
                  <div className="card-value">
                    {weatherResult.weather_data.temperature.toFixed(1)}°C
                  </div>
                </motion.div>

                <motion.div 
                  className="weather-card"
                  initial={{ opacity: 0, x: 50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.9 }}
                >
                  <div className="card-header">
                    <WiHumidity size="2rem" color="#2196f3" />
                    <h4>Humidity</h4>
                  </div>
                  <div className="card-value">
                    {weatherResult.weather_data.humidity}%
                  </div>
                </motion.div>

                <motion.div 
                  className="weather-card"
                  initial={{ opacity: 0, x: -50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 1.0 }}
                >
                  <div className="card-header">
                    <WiCloudy size="2rem" color="#607d8b" />
                    <h4>Clouds</h4>
                  </div>
                  <div className="card-value">
                    {weatherResult.weather_data.clouds}%
                  </div>
                </motion.div>

                <motion.div 
                  className="weather-card"
                  initial={{ opacity: 0, x: 50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 1.1 }}
                >
                  <div className="card-header">
                    <WiStrongWind size="2rem" color="#9c27b0" />
                    <h4>Wind</h4>
                  </div>
                  <div className="card-value">
                    {weatherResult.weather_data.wind.toFixed(1)} m/s
                  </div>
                </motion.div>
              </div>

              {/* Conditions */}
              <motion.div 
                className="conditions-section"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.2 }}
              >
                <h4>Weather Conditions</h4>
                <div className="conditions-list">
                  {weatherResult.conditions.map((condition, index) => (
                    <motion.span 
                      key={index} 
                      className="condition-badge"
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 1.3 + index * 0.1 }}
                    >
                      {getWeatherIcon(condition.split(' (')[0], "1.2rem")}
                      {condition}
                    </motion.span>
                  ))}
                </div>
              </motion.div>

              {/* Suggestion */}
              <motion.div 
                className="suggestion-box"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.4 }}
              >
                <h4>💡 AI Recommendation</h4>
                <p>{weatherResult.suggestion}</p>
              </motion.div>

              {/* AI Model Predictions */}
              <motion.div 
                className="model-predictions"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.5 }}
              >
                <h4>
                  <FaRobot className="model-icon" />
                  AI Model Predictions
                </h4>
                <div className="model-grid">
                  {weatherResult.model_predictions.random_forest && (
                    <motion.div 
                      className="model-item"
                      whileHover={{ scale: 1.05 }}
                    >
                      <span className="model-name">
                        <FaTree className="model-icon" />
                        Random Forest
                      </span>
                      <span className="model-value">
                        {weatherResult.model_predictions.random_forest.toFixed(1)}%
                      </span>
                    </motion.div>
                  )}
                  {weatherResult.model_predictions.lstm && (
                    <motion.div 
                      className="model-item"
                      whileHover={{ scale: 1.05 }}
                    >
                      <span className="model-name">
                        <FaBrain className="model-icon" />
                        LSTM Neural Net
                      </span>
                      <span className="model-value">
                        {weatherResult.model_predictions.lstm.toFixed(1)}%
                      </span>
                    </motion.div>
                  )}
                  {weatherResult.model_predictions.xgboost && (
                    <motion.div 
                      className="model-item"
                      whileHover={{ scale: 1.05 }}
                    >
                      <span className="model-name">
                        <FaBolt className="model-icon" />
                        XGBoost
                      </span>
                      <span className="model-value">
                        {weatherResult.model_predictions.xgboost.toFixed(1)}%
                      </span>
                    </motion.div>
                  )}
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

export default App;