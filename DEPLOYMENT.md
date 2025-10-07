# 🚀 Weather Predictor - Render Deployment Guide

## 📋 Prerequisites
- GitHub account
- Render account (free tier available)
- Your project pushed to GitHub

## 🛠️ Deployment Steps

### 1. **Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit - Weather Predictor"
git remote add origin https://github.com/YOUR_USERNAME/weather-predictor.git
git push -u origin main
```

### 2. **Deploy Backend API on Render**

1. **Go to [Render Dashboard](https://dashboard.render.com/)**
2. **Click "New +" → "Web Service"**
3. **Connect your GitHub repository**
4. **Configure the service:**
   - **Name**: `weather-predictor-api`
   - **Environment**: `Python 3`
   - **Build Command**: `chmod +x build.sh && ./build.sh`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free` (for testing)

5. **Add Environment Variables in Render Dashboard:**
   - `OPENWEATHER_API_KEY`: `your_actual_openweather_key`
   - `GOOGLE_API_KEY`: `your_actual_google_key` 
   - `PYTHON_VERSION`: `3.10.0`
   
   ⚠️ **NEVER commit API keys to GitHub!**

6. **Click "Create Web Service"**

### 3. **Deploy Frontend on Render (Optional)**

1. **Create another service**: "Static Site"
2. **Configure:**
   - **Name**: `weather-predictor-frontend` 
   - **Build Command**: `cd frontend && npm install && npm run build`
   - **Publish Directory**: `frontend/build`
   - **Environment Variable**:
     - `REACT_APP_API_URL`: `https://weather-predictor-api.onrender.com`

### 4. **Alternative: Deploy Frontend on Netlify/Vercel**

**For better performance, deploy frontend separately:**

#### **Netlify:**
1. Drag & drop your `frontend/build` folder to Netlify
2. Set environment variable: `REACT_APP_API_URL`

#### **Vercel:**
1. Connect GitHub repo
2. Set root directory to `frontend`
3. Add environment variable: `REACT_APP_API_URL`

## 🔧 **Production URLs**

- **Backend API**: `https://weather-predictor-api.onrender.com`
- **Frontend**: `https://weather-predictor-frontend.onrender.com`
- **API Docs**: `https://weather-predictor-api.onrender.com/docs`

## 🧪 **Testing Production**

1. **Test API**: Visit `https://your-api-url.onrender.com/docs`
2. **Test Frontend**: Visit your frontend URL
3. **Test Location Search**: Try "Omkar Nagar Nagpur"
4. **Test Weather Prediction**: Complete flow

## 📊 **Performance Notes**

- **Free Tier**: Apps sleep after 15 mins of inactivity
- **Cold Start**: ~30 seconds to wake up
- **Upgrade**: Paid plans for always-on service

## 🔐 **Security Best Practices**

1. **Use Environment Variables** for all API keys
2. **Restrict CORS** origins to your domains
3. **Add rate limiting** for production use
4. **Monitor usage** to avoid quota limits

## 🎯 **Success Criteria**

✅ Backend API responding at `/docs`  
✅ Frontend loads without errors  
✅ Location search works (Google Geocoding)  
✅ Weather predictions show all 3 AI models  
✅ Beautiful UI with animations working  

## 🆘 **Troubleshooting**

- **Build fails**: Check `requirements.txt` versions
- **API keys not working**: Verify environment variables
- **CORS errors**: Check allowed origins
- **Slow response**: Use paid tier for better performance

Your Weather Predictor is now live! 🌟