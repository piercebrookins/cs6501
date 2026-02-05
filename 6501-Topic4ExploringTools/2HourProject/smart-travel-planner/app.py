#!/usr/bin/env python3
"""Flask API server for Smart Travel Planner website.

Serves the static website AND provides an API endpoint for weather data.
"""

import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our weather tool and OpenAI service
from src.tools.weather import get_weather_forecast
from src.services.openai_service import generate_travel_recommendations

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Enable CORS for local development


# ==========================================================================
# STATIC FILE SERVING
# ==========================================================================

@app.route('/')
def serve_index():
    """Serve the main index.html page."""
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS, images)."""
    return send_from_directory('.', path)


# ==========================================================================
# API ENDPOINTS
# ==========================================================================

@app.route('/api/weather', methods=['GET'])
def get_weather():
    """
    Get weather forecast for a city.
    
    Query Parameters:
        city (str): Name of the city (required)
        units (str): 'metric' or 'imperial' (default: 'metric')
    
    Returns:
        JSON with weather data or error message
    """
    city = request.args.get('city', '').strip()
    units = request.args.get('units', 'metric')
    
    if not city:
        return jsonify({
            'error': 'City parameter is required',
            'example': '/api/weather?city=Paris&units=metric'
        }), 400
    
    if units not in ['metric', 'imperial']:
        units = 'metric'
    
    try:
        # Call our weather tool
        result = get_weather_forecast.invoke({
            'city': city,
            'units': units
        })
        
        # Check if result contains an error
        if result.startswith('❌'):
            return jsonify({
                'error': result.replace('❌ ', ''),
                'city': city
            }), 400
        
        return jsonify({
            'success': True,
            'city': city,
            'units': units,
            'weather': result
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to fetch weather: {str(e)}',
            'city': city
        }), 500


@app.route('/api/travel-plan', methods=['GET'])
def get_travel_plan():
    """
    Get a full AI-powered travel plan (weather + packing + activities).

    Query Parameters:
        city (str): Destination city (required)
        units (str): 'metric' or 'imperial' (default: 'metric')

    Returns:
        JSON with weather data and AI-generated recommendations.
        Falls back gracefully if OpenAI is unavailable.
    """
    city = request.args.get('city', '').strip()
    units = request.args.get('units', 'metric')

    if not city:
        return jsonify({
            'error': 'City parameter is required',
            'example': '/api/travel-plan?city=Paris&units=metric'
        }), 400

    if units not in ('metric', 'imperial'):
        units = 'metric'

    # --- 1. Fetch weather ---------------------------------------------------
    try:
        weather_text = get_weather_forecast.invoke({
            'city': city,
            'units': units,
        })
    except Exception as e:
        return jsonify({'error': f'Failed to fetch weather: {e}', 'city': city}), 500

    if weather_text.startswith('❌'):
        return jsonify({
            'error': weather_text.replace('❌ ', ''),
            'city': city,
        }), 400

    # --- 2. Ask OpenAI for recommendations ----------------------------------
    ai_recs = generate_travel_recommendations(city, weather_text, units)

    # --- 3. Build response --------------------------------------------------
    payload = {
        'success': True,
        'city': city,
        'units': units,
        'weather': weather_text,
        'ai_powered': ai_recs is not None,
    }

    if ai_recs:
        payload['summary'] = ai_recs['summary']
        payload['packing'] = ai_recs['packing']
        payload['activities'] = ai_recs['activities']

    return jsonify(payload)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    weather_key = bool(os.getenv('OPENWEATHER_API_KEY'))
    openai_key = bool(os.getenv('OPENAI_API_KEY'))
    return jsonify({
        'status': 'healthy',
        'weather_api_configured': weather_key,
        'openai_api_configured': openai_key,
    })


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == '__main__':
    # Check for API key
    if not os.getenv('OPENWEATHER_API_KEY'):
        print("\n⚠️  Warning: OPENWEATHER_API_KEY not set!")
        print("   Copy .env.example to .env and add your API key.")
        print("   Get a free key at: https://openweathermap.org/api\n")

    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️  Warning: OPENAI_API_KEY not set!")
        print("   AI recommendations will be disabled (hardcoded fallback used).")
        print("   Get a key at: https://platform.openai.com/api-keys\n")
    
    print("\n🌤️  TripCast - Smart Travel Planner")
    print("=" * 40)
    print("\n🚀 Server starting...")
    print("   Website: http://localhost:3000")
    print("   API:     http://localhost:3000/api/weather?city=Paris")
    print("\n   Press Ctrl+C to stop.\n")
    
    app.run(
        host='0.0.0.0',
        port=3000,
        debug=True
    )
