# Theme Manager for Streamlit App
# Author: Enhanced by AI Assistant
# Date: 2024

import streamlit as st
import json
from typing import Dict, Any

class ThemeManager:
    def __init__(self):
        self.themes = {
            "🌙 Dark Mode": {
                "bg_color": "#0E1117",
                "secondary_bg": "#262730",
                "text_color": "#FAFAFA",
                "accent_color": "#FF6B6B",
                "success_color": "#00CC88",
                "warning_color": "#FFB800",
                "error_color": "#FF4B4B",
                "info_color": "#4B9BFF",
                "card_bg": "#1A1B21",
                "border_color": "#363842",
                "gradient_start": "#1A1B21",
                "gradient_end": "#2E2F3A"
            },
            "☀️ Light Mode": {
                "bg_color": "#FFFFFF",
                "secondary_bg": "#F0F2F6",
                "text_color": "#262730",
                "accent_color": "#FF4B4B",
                "success_color": "#00CC88",
                "warning_color": "#FFA500",
                "error_color": "#FF0000",
                "info_color": "#0066CC",
                "card_bg": "#FFFFFF",
                "border_color": "#E0E0E0",
                "gradient_start": "#F8F9FA",
                "gradient_end": "#E9ECEF"
            },
            "🌊 Ocean Blue": {
                "bg_color": "#0A1929",
                "secondary_bg": "#132F4C",
                "text_color": "#B2BAC2",
                "accent_color": "#5090D3",
                "success_color": "#66BB6A",
                "warning_color": "#FFA726",
                "error_color": "#F44336",
                "info_color": "#29B6F6",
                "card_bg": "#001E3C",
                "border_color": "#265D97",
                "gradient_start": "#0A1929",
                "gradient_end": "#1E3A5F"
            },
            "🌲 Forest Green": {
                "bg_color": "#0F1E0F",
                "secondary_bg": "#1A2F1A",
                "text_color": "#C8E6C9",
                "accent_color": "#4CAF50",
                "success_color": "#8BC34A",
                "warning_color": "#FF9800",
                "error_color": "#F44336",
                "info_color": "#03A9F4",
                "card_bg": "#1B301B",
                "border_color": "#2E7D32",
                "gradient_start": "#0F1E0F",
                "gradient_end": "#1B5E20"
            },
            "🌅 Sunset": {
                "bg_color": "#1A0E1F",
                "secondary_bg": "#2D1B36",
                "text_color": "#F5E6D3",
                "accent_color": "#FF6B9D",
                "success_color": "#C44569",
                "warning_color": "#FFC107",
                "error_color": "#FF5252",
                "info_color": "#E91E63",
                "card_bg": "#2D1B36",
                "border_color": "#723C70",
                "gradient_start": "#1A0E1F",
                "gradient_end": "#723C70"
            },
            "🎨 Custom": {
                "bg_color": "#FFFFFF",
                "secondary_bg": "#F0F2F6",
                "text_color": "#262730",
                "accent_color": "#FF4B4B",
                "success_color": "#00CC88",
                "warning_color": "#FFA500",
                "error_color": "#FF0000",
                "info_color": "#0066CC",
                "card_bg": "#FFFFFF",
                "border_color": "#E0E0E0",
                "gradient_start": "#FFFFFF",
                "gradient_end": "#F0F2F6"
            }
        }
    
    def get_theme(self, theme_name: str) -> Dict[str, str]:
        return self.themes.get(theme_name, self.themes["🌙 Dark Mode"])
    
    def apply_theme(self, theme: Dict[str, str]) -> str:
        return f"""
        <style>
            /* Import Google Fonts */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            /* Main app background */
            .stApp {{
                background: linear-gradient(135deg, {theme['gradient_start']} 0%, {theme['gradient_end']} 100%);
                color: {theme['text_color']};
                font-family: 'Inter', sans-serif;
            }}
            
            /* Sidebar styling */
            section[data-testid="stSidebar"] {{
                background: {theme['secondary_bg']};
                border-right: 2px solid {theme['border_color']};
                box-shadow: 2px 0 10px rgba(0,0,0,0.1);
            }}
            
            section[data-testid="stSidebar"] > div {{
                background: {theme['secondary_bg']};
            }}
            
            /* Headers */
            h1, h2, h3, h4, h5, h6 {{
                color: {theme['text_color']} !important;
                font-weight: 600;
                letter-spacing: -0.02em;
            }}
            
            h1 {{
                background: linear-gradient(90deg, {theme['accent_color']}, {theme['info_color']});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-align: center;
                font-size: 3rem !important;
                margin-bottom: 2rem;
                animation: fadeInScale 0.8s ease-out;
            }}
            
            @keyframes fadeInScale {{
                from {{
                    opacity: 0;
                    transform: scale(0.9);
                }}
                to {{
                    opacity: 1;
                    transform: scale(1);
                }}
            }}
            
            /* Cards and containers */
            .stMetric, .stAlert, div[data-testid="metric-container"] {{
                background: {theme['card_bg']};
                border: 1px solid {theme['border_color']};
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }}
            
            .stMetric:hover, div[data-testid="metric-container"]:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
            }}
            
            /* Buttons */
            .stButton > button {{
                background: linear-gradient(90deg, {theme['accent_color']}, {theme['info_color']});
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.75rem 2rem;
                font-weight: 600;
                font-size: 1rem;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .stButton > button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                filter: brightness(1.1);
            }}
            
            /* Input fields */
            .stTextInput > div > div > input,
            .stNumberInput > div > div > input,
            .stSelectbox > div > div > select {{
                background: {theme['card_bg']};
                color: {theme['text_color']};
                border: 2px solid {theme['border_color']};
                border-radius: 8px;
                padding: 0.5rem;
                transition: all 0.3s ease;
            }}
            
            .stTextInput > div > div > input:focus,
            .stNumberInput > div > div > input:focus,
            .stSelectbox > div > div > select:focus {{
                border-color: {theme['accent_color']};
                box-shadow: 0 0 0 3px {theme['accent_color']}33;
                outline: none;
            }}
            
            /* Sliders */
            .stSlider > div > div > div > div {{
                background: {theme['accent_color']};
            }}
            
            /* Progress bars */
            .stProgress > div > div > div > div {{
                background: linear-gradient(90deg, {theme['accent_color']}, {theme['info_color']});
                border-radius: 10px;
            }}
            
            /* Alerts and messages */
            .stAlert {{
                background: {theme['card_bg']};
                border-left: 4px solid {theme['accent_color']};
                border-radius: 8px;
            }}
            
            /* Tables */
            .stDataFrame {{
                background: {theme['card_bg']};
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            
            /* Plots and charts */
            .js-plotly-plot .plotly {{
                border-radius: 12px;
                overflow: hidden;
            }}
            
            /* File uploader */
            .stFileUploader > div {{
                background: {theme['card_bg']};
                border: 2px dashed {theme['border_color']};
                border-radius: 12px;
                padding: 2rem;
                transition: all 0.3s ease;
            }}
            
            .stFileUploader > div:hover {{
                border-color: {theme['accent_color']};
                background: {theme['secondary_bg']};
            }}
            
            /* Custom animations */
            @keyframes pulse {{
                0% {{
                    box-shadow: 0 0 0 0 {theme['accent_color']}40;
                }}
                70% {{
                    box-shadow: 0 0 0 10px {theme['accent_color']}00;
                }}
                100% {{
                    box-shadow: 0 0 0 0 {theme['accent_color']}00;
                }}
            }}
            
            .pulse-animation {{
                animation: pulse 2s infinite;
            }}
            
            /* Tooltips */
            div[data-baseweb="tooltip"] {{
                background: {theme['card_bg']} !important;
                border: 1px solid {theme['border_color']} !important;
                border-radius: 8px !important;
                color: {theme['text_color']} !important;
            }}
            
            /* Expander */
            .streamlit-expanderHeader {{
                background: {theme['card_bg']};
                border: 1px solid {theme['border_color']};
                border-radius: 8px;
                color: {theme['text_color']} !important;
            }}
            
            .streamlit-expanderHeader:hover {{
                background: {theme['secondary_bg']};
            }}
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
                background: {theme['secondary_bg']};
                padding: 0.5rem;
                border-radius: 12px;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                background: transparent;
                border-radius: 8px;
                color: {theme['text_color']};
                padding: 0.5rem 1rem;
                transition: all 0.3s ease;
            }}
            
            .stTabs [aria-selected="true"] {{
                background: {theme['accent_color']};
                color: white !important;
            }}
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {{
                width: 10px;
                height: 10px;
            }}
            
            ::-webkit-scrollbar-track {{
                background: {theme['secondary_bg']};
                border-radius: 10px;
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: {theme['accent_color']};
                border-radius: 10px;
            }}
            
            ::-webkit-scrollbar-thumb:hover {{
                background: {theme['info_color']};
            }}
        </style>
        """
    
    def save_custom_theme(self, theme: Dict[str, str]):
        if 'custom_themes' not in st.session_state:
            st.session_state.custom_themes = []
        st.session_state.custom_themes.append(theme)
    
    def get_custom_themes(self):
        return st.session_state.get('custom_themes', [])