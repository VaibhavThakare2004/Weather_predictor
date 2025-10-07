# 🌤️ AI Weather Predictor

[![Deploy](https://img.shields.io/badge/Deploy-Render-46E3B7)](https://render.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![React](https://img.shields.io/badge/React-18+-61DAFB)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688)](https://fastapi.tiangolo.com)

> **Advanced weather prediction system using ensemble AI models with beautiful React frontend**

## ✨ Features

- 🎯 **Pinpoint Location Search** - Google Geocoding API for exact locations
- 🤖 **Triple AI Ensemble** - Random Forest, LSTM Neural Network & XGBoost
- 🌍 **Global Weather Data** - OpenWeatherMap integration
- 💫 **Beautiful UI** - Glassmorphism design with smooth animations
- ⚡ **Real-time Predictions** - Instant weather risk assessment
- 📱 **Responsive Design** - Works on all devices

## 🚀 Live Demo

- **Frontend**: [Your App URL]
- **API Docs**: [Your API URL]/docs

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Scikit-learn** - Random Forest classifier
- **TensorFlow** - LSTM neural networks
- **XGBoost** - Gradient boosting models
- **Pandas/NumPy** - Data processing

### Frontend
- **React 18** - Modern UI library
- **Framer Motion** - Smooth animations
- **Axios** - API communication
- **CSS3** - Glassmorphism styling

## 📋 Prerequisites

- Python 3.10+
- Node.js 16+
- OpenWeatherMap API key
- Google Geocoding API key

## 🔧 Local Development

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/weather-predictor.git
cd weather-predictor
```

### 2. Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your API keys

# Start backend server
uvicorn api:app --host 0.0.0.0 --port 8002 --reload
```

### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

### 4. Open Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8002
- **API Docs**: http://localhost:8002/docs

## 🌍 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

### Quick Deploy to Render

1. **Fork this repository**
2. **Add environment variables** in Render dashboard:
   - `OPENWEATHER_API_KEY`
   - `GOOGLE_API_KEY`
3. **Connect to Render** and deploy!

## 🎯 How It Works

### AI Model Ensemble
1. **Random Forest** - Learns from current weather patterns
2. **LSTM Network** - Captures temporal weather sequences  
3. **XGBoost** - Uses pre-trained historical data
4. **Smart Weighting** - Combines predictions intelligently

### Weather Analysis
- **Location Precision** - Google Geocoding for exact coordinates
- **Multi-factor Analysis** - Temperature, humidity, clouds, wind, precipitation
- **Risk Assessment** - Comprehensive weather condition evaluation
- **Smart Suggestions** - Actionable weather advice

## 📊 Model Performance

- **Ensemble Accuracy**: ~85%
- **Response Time**: <2 seconds
- **Location Coverage**: Global
- **Data Sources**: OpenWeatherMap + Historical datasets

## 🔐 Environment Variables

```bash
OPENWEATHER_API_KEY=your_openweather_api_key
GOOGLE_API_KEY=your_google_geocoding_key
REACT_APP_API_URL=your_backend_url
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- OpenWeatherMap for weather data
- Google Maps for geocoding services
- Render for hosting platform
- React community for amazing tools

## 📞 Support

- 📧 Email: your-email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/YOUR_USERNAME/weather-predictor/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/YOUR_USERNAME/weather-predictor/discussions)

---

**Built with ❤️ by [Your Name]**