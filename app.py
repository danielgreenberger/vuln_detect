#!/usr/bin/env python3
"""
Enhanced Dash web application for Mental Health Vulnerability Detection.

Features:
- Visual risk gauge with animated indicator
- Probability breakdown for all classes
- Sample text examples for quick testing
- Prediction history tracking
- Dark/Light theme toggle
- Real-time text statistics
- Model comparison mode
- Export results functionality
"""

import logging
from pathlib import Path
from datetime import datetime
import json

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import plotly.express as px

from predict import Predictor

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Dash app with external stylesheets
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "🧠 Mental Health Vulnerability Detector"

# --- Constants ---
COLORS = {
    "Safe": {"primary": "#28a745", "light": "#d4edda", "icon": "✅"},
    "Depression": {"primary": "#ffc107", "light": "#fff3cd", "icon": "⚠️"},
    "Suicide": {"primary": "#dc3545", "light": "#f8d7da", "icon": "🚨"},
}

SAMPLE_TEXTS = [
    {
        "title": "Neutral/Safe",
        "text": "Had a great day at work today! The weather was beautiful and I enjoyed lunch with my colleagues.",
        "icon": "😊"
    },
    {
        "title": "Mild Concern",
        "text": "I've been feeling a bit down lately. Work has been stressful and I haven't been sleeping well.",
        "icon": "😔"
    },
    {
        "title": "Depression Indicators",
        "text": "I don't see the point anymore. Everything feels empty and I can't remember the last time I felt happy. I just want to stay in bed all day.",
        "icon": "😢"
    },
    {
        "title": "Severe Risk",
        "text": "I can't take this anymore. Nobody would even notice if I was gone. I've been thinking about ending it all.",
        "icon": "🆘"
    },
]

# --- Styles ---
DARK_THEME = {
    "background": "#1a1a2e",
    "card": "#16213e",
    "text": "#eaeaea",
    "border": "#0f3460",
    "accent": "#e94560",
}

LIGHT_THEME = {
    "background": "#f8f9fa",
    "card": "#ffffff",
    "text": "#212529",
    "border": "#dee2e6",
    "accent": "#007bff",
}

# --- Model Discovery ---
def find_available_models(outputs_dir="outputs"):
    """Scans the outputs directory for valid, trained models."""
    models = {}
    base_path = Path(outputs_dir)
    if not base_path.exists():
        logger.warning(f"Outputs directory not found: {outputs_dir}")
        return models

    for run_dir in sorted(base_path.iterdir()):
        if run_dir.is_dir():
            final_model_path = run_dir / "final_model"
            if final_model_path.exists() and final_model_path.is_dir():
                model_name = run_dir.name
                models[model_name] = str(final_model_path)
                logger.info(f"Found valid model: {model_name}")

    return models

AVAILABLE_MODELS = find_available_models()
PREDICTOR_CACHE = {}

# --- Helper Functions ---
def create_gauge_chart(probabilities, theme):
    """Creates a beautiful gauge chart showing risk level."""
    # Calculate weighted risk score (0-100)
    risk_score = (
        probabilities.get("Safe", 0) * 0 +
        probabilities.get("Depression", 0) * 50 +
        probabilities.get("Suicide", 0) * 100
    )
    
    # Determine color based on risk
    if risk_score < 25:
        gauge_color = COLORS["Safe"]["primary"]
    elif risk_score < 60:
        gauge_color = COLORS["Depression"]["primary"]
    else:
        gauge_color = COLORS["Suicide"]["primary"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Risk Level", "font": {"size": 20, "color": theme["text"]}},
        number={"suffix": "%", "font": {"size": 40, "color": theme["text"]}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": theme["text"]},
            "bar": {"color": gauge_color, "thickness": 0.75},
            "bgcolor": theme["card"],
            "borderwidth": 2,
            "bordercolor": theme["border"],
            "steps": [
                {"range": [0, 25], "color": COLORS["Safe"]["light"]},
                {"range": [25, 60], "color": COLORS["Depression"]["light"]},
                {"range": [60, 100], "color": COLORS["Suicide"]["light"]},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": risk_score,
            },
        },
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": theme["text"]},
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig

def create_probability_bars(probabilities, theme):
    """Creates a horizontal bar chart showing all class probabilities."""
    labels = list(probabilities.keys())
    values = [probabilities[l] * 100 for l in labels]
    colors = [COLORS[l]["primary"] for l in labels]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="auto",
        textfont={"color": "white", "size": 14},
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": theme["text"]},
        height=150,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis={"range": [0, 100], "showgrid": False, "showticklabels": False},
        yaxis={"showgrid": False},
        showlegend=False,
    )
    
    return fig

def get_theme(is_dark):
    """Returns the appropriate theme based on toggle state."""
    return DARK_THEME if is_dark else LIGHT_THEME

# --- App Layout ---
def create_layout():
    return html.Div(
        id="main-container",
        children=[
            # Store for theme and history
            dcc.Store(id="theme-store", data={"dark": False}),
            dcc.Store(id="history-store", data=[]),
            
            # Header
            html.Div(
                className="header",
                children=[
                    html.Div(className="header-content", children=[
                        html.H1("🧠 Mental Health Vulnerability Detector", className="title"),
                        html.P("AI-powered analysis for mental health risk assessment", className="subtitle"),
                    ]),
                    html.Button(
                        "🌙 Dark Mode",
                        id="theme-toggle",
                        className="theme-toggle",
                    ),
                ],
            ),
            
            # Main content
            html.Div(
                className="content-grid",
                children=[
                    # Left Panel - Input
                    html.Div(
                        className="panel input-panel",
                        children=[
                            html.H3("📝 Input Text", className="panel-title"),
                            
                            # Model Selection
                            html.Div(className="form-group", children=[
                                html.Label("Select Model:", className="label"),
                                dcc.Dropdown(
                                    id="model-dropdown",
                                    options=[{"label": f"📦 {name}", "value": path} 
                                            for name, path in AVAILABLE_MODELS.items()],
                                    value=next(iter(AVAILABLE_MODELS.values()), None),
                                    clearable=False,
                                    className="dropdown",
                                ),
                            ]),
                            
                            # Text Input
                            html.Div(className="form-group", children=[
                                html.Label("Enter text to analyze:", className="label"),
                                dcc.Textarea(
                                    id="text-input",
                                    placeholder="Type or paste text here for analysis...",
                                    className="textarea",
                                ),
                                html.Div(id="text-stats", className="text-stats"),
                            ]),
                            
                            # Predict Button
                            html.Button(
                                "🔍 Analyze Text",
                                id="predict-button",
                                className="predict-button",
                                n_clicks=0,
                            ),
                            
                            # Sample Texts
                            html.Div(className="samples-section", children=[
                                html.H4("💡 Try Sample Texts:", className="samples-title"),
                                html.Div(
                                    className="samples-grid",
                                    children=[
                                        html.Button(
                                            [html.Span(s["icon"]), html.Span(s["title"])],
                                            id={"type": "sample-btn", "index": i},
                                            className="sample-button",
                                            **{"data-text": s["text"]},
                                        )
                                        for i, s in enumerate(SAMPLE_TEXTS)
                                    ],
                                ),
                            ]),
                        ],
                    ),
                    
                    # Right Panel - Results
                    html.Div(
                        className="panel results-panel",
                        children=[
                            html.H3("📊 Analysis Results", className="panel-title"),
                            html.Div(id="results-container", children=[
                                html.Div(className="placeholder", children=[
                                    html.Span("🔮", className="placeholder-icon"),
                                    html.P("Enter text and click 'Analyze' to see results"),
                                ]),
                            ]),
                        ],
                    ),
                ],
            ),
            
            # History Panel
            html.Div(
                className="panel history-panel",
                children=[
                    html.Div(className="history-header", children=[
                        html.H3("📜 Prediction History", className="panel-title"),
                        html.Button("🗑️ Clear", id="clear-history", className="clear-button"),
                    ]),
                    html.Div(id="history-container", className="history-container"),
                ],
            ),
            
            # Footer
            html.Footer(
                className="footer",
                children=[
                    html.P("⚠️ This tool is for educational purposes only."),
                ],
            ),
        ],
    )

app.layout = create_layout()

# --- CSS Styles ---
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                transition: all 0.3s ease;
            }
            
            #main-container {
                min-height: 100vh;
                padding: 20px;
                transition: all 0.3s ease;
            }
            
            .header {
                display: flex;
                flex-direction: column;
                align-items: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid;
            }
            
            .header-content {
                text-align: center;
                margin-bottom: 15px;
            }
            
            .title {
                font-size: 2rem;
                margin-bottom: 5px;
            }
            
            .subtitle {
                opacity: 0.7;
                font-size: 1rem;
            }
            
            .theme-toggle {
                padding: 10px 20px;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-size: 1rem;
                transition: all 0.3s ease;
            }
            
            .theme-toggle:hover {
                transform: scale(1.05);
            }
            
            .content-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }
            
            @media (max-width: 900px) {
                .content-grid {
                    grid-template-columns: 1fr;
                }
            }
            
            .panel {
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
            }
            
            .panel:hover {
                box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
            }
            
            .panel-title {
                margin-bottom: 20px;
                font-size: 1.3rem;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            .label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
            }
            
            .dropdown {
                width: 100%;
            }
            
            .textarea {
                width: 100%;
                min-height: 150px;
                padding: 15px;
                border: 2px solid;
                border-radius: 10px;
                font-size: 1rem;
                resize: vertical;
                transition: all 0.3s ease;
            }
            
            .textarea:focus {
                outline: none;
                border-color: #007bff;
            }
            
            .text-stats {
                margin-top: 10px;
                font-size: 0.85rem;
                opacity: 0.7;
            }
            
            .predict-button {
                width: 100%;
                padding: 15px;
                border: none;
                border-radius: 10px;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                background: linear-gradient(135deg, #007bff, #0056b3);
                color: white;
            }
            
            .predict-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 123, 255, 0.4);
            }
            
            .predict-button:active {
                transform: translateY(0);
            }
            
            .samples-section {
                margin-top: 25px;
                padding-top: 20px;
                border-top: 1px solid;
            }
            
            .samples-title {
                margin-bottom: 15px;
                font-size: 1rem;
            }
            
            .samples-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 10px;
            }
            
            .sample-button {
                padding: 10px;
                border: 2px solid;
                border-radius: 8px;
                cursor: pointer;
                font-size: 0.85rem;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .sample-button:hover {
                transform: scale(1.02);
            }
            
            .placeholder {
                text-align: center;
                padding: 50px;
                opacity: 0.5;
            }
            
            .placeholder-icon {
                font-size: 4rem;
                display: block;
                margin-bottom: 15px;
            }
            
            .result-card {
                text-align: center;
            }
            
            .result-label {
                font-size: 2.5rem;
                font-weight: bold;
                margin: 20px 0;
                padding: 20px;
                border-radius: 15px;
                display: inline-block;
            }
            
            .result-icon {
                font-size: 4rem;
                margin-bottom: 10px;
            }
            
            .charts-container {
                margin-top: 20px;
            }
            
            .history-panel {
                margin-top: 20px;
            }
            
            .history-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }
            
            .clear-button {
                padding: 8px 15px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 0.9rem;
            }
            
            .history-container {
                display: flex;
                gap: 15px;
                overflow-x: auto;
                padding: 10px 0;
            }
            
            .history-item {
                min-width: 200px;
                padding: 15px;
                border-radius: 10px;
                border: 2px solid;
            }
            
            .history-text {
                font-size: 0.85rem;
                margin-bottom: 10px;
                max-height: 60px;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            
            .history-result {
                font-weight: bold;
                font-size: 1rem;
            }
            
            .history-time {
                font-size: 0.75rem;
                opacity: 0.6;
                margin-top: 5px;
            }
            
            .footer {
                text-align: center;
                padding: 20px;
                margin-top: 30px;
                border-top: 1px solid;
                font-size: 0.9rem;
                opacity: 0.8;
            }
            
            .footer p {
                margin: 5px 0;
            }
            
            /* Light theme */
            .light-theme {
                background-color: #f8f9fa;
                color: #212529;
            }
            
            .light-theme .panel {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
            }
            
            .light-theme .header {
                border-color: #dee2e6;
            }
            
            .light-theme .theme-toggle {
                background-color: #343a40;
                color: white;
            }
            
            .light-theme .textarea {
                background-color: #ffffff;
                border-color: #ced4da;
                color: #212529;
            }
            
            .light-theme .sample-button {
                background-color: #f8f9fa;
                border-color: #dee2e6;
                color: #212529;
            }
            
            .light-theme .clear-button {
                background-color: #dc3545;
                color: white;
            }
            
            .light-theme .history-item {
                background-color: #f8f9fa;
                border-color: #dee2e6;
            }
            
            .light-theme .samples-section {
                border-color: #dee2e6;
            }
            
            .light-theme .footer {
                border-color: #dee2e6;
            }
            
            /* Dark theme */
            .dark-theme {
                background-color: #1a1a2e;
                color: #eaeaea;
            }
            
            .dark-theme .panel {
                background-color: #16213e;
                border: 1px solid #0f3460;
            }
            
            .dark-theme .header {
                border-color: #0f3460;
            }
            
            .dark-theme .theme-toggle {
                background-color: #ffc107;
                color: #212529;
            }
            
            .dark-theme .textarea {
                background-color: #0f3460;
                border-color: #0f3460;
                color: #eaeaea;
            }
            
            .dark-theme .sample-button {
                background-color: #0f3460;
                border-color: #1a1a2e;
                color: #eaeaea;
            }
            
            .dark-theme .clear-button {
                background-color: #e94560;
                color: white;
            }
            
            .dark-theme .history-item {
                background-color: #0f3460;
                border-color: #1a1a2e;
            }
            
            .dark-theme .samples-section {
                border-color: #0f3460;
            }
            
            .dark-theme .footer {
                border-color: #0f3460;
            }
            
            /* Dropdown styling */
            .Select-control {
                border-radius: 10px !important;
            }
        </style>
    </head>
    <body class="light-theme">
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# --- Callbacks ---

@app.callback(
    Output("main-container", "className"),
    Output("theme-toggle", "children"),
    Input("theme-toggle", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def toggle_theme(n_clicks, theme_data):
    """Toggle between dark and light themes."""
    is_dark = not theme_data.get("dark", False)
    if is_dark:
        return "dark-theme", "☀️ Light Mode"
    return "light-theme", "🌙 Dark Mode"

@app.callback(
    Output("theme-store", "data"),
    Input("theme-toggle", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def update_theme_store(n_clicks, theme_data):
    """Update theme store."""
    is_dark = not theme_data.get("dark", False)
    return {"dark": is_dark}

@app.callback(
    Output("text-stats", "children"),
    Input("text-input", "value"),
)
def update_text_stats(text):
    """Update text statistics in real-time."""
    if not text:
        return "📊 Characters: 0 | Words: 0 | Sentences: 0"
    
    char_count = len(text)
    word_count = len(text.split())
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    
    return f"📊 Characters: {char_count} | Words: {word_count} | Sentences: {sentence_count}"

@app.callback(
    Output("text-input", "value"),
    Input({"type": "sample-btn", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def load_sample_text(n_clicks):
    """Load sample text when a sample button is clicked."""
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id:
        try:
            index = json.loads(button_id)["index"]
            return SAMPLE_TEXTS[index]["text"]
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
    
    return dash.no_update

@app.callback(
    Output("results-container", "children"),
    Output("history-store", "data"),
    Input("predict-button", "n_clicks"),
    State("model-dropdown", "value"),
    State("text-input", "value"),
    State("history-store", "data"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def update_prediction(n_clicks, model_path, text, history, theme_data):
    """Run prediction and update results."""
    if not text or not model_path:
        return html.Div(className="placeholder", children=[
            html.Span("⚠️", className="placeholder-icon"),
            html.P("Please enter text and select a model"),
        ]), history
    
    try:
        # Load predictor from cache or create new one
        if model_path not in PREDICTOR_CACHE:
            logger.info(f"Loading predictor for: {model_path}")
            PREDICTOR_CACHE[model_path] = Predictor(model_path)
        predictor = PREDICTOR_CACHE[model_path]
        
        # Get prediction
        result = predictor.predict(text)
        label = result["label"]
        confidence = result["confidence"]
        probabilities = result.get("probabilities", {label: confidence})
        
        # Get theme
        theme = get_theme(theme_data.get("dark", False))
        
        # Create gauge chart
        gauge_fig = create_gauge_chart(probabilities, theme)
        
        # Create probability bars
        prob_fig = create_probability_bars(probabilities, theme)
        
        # Calculate risk score for display
        safe_prob = probabilities.get("Safe", 0)
        depression_prob = probabilities.get("Depression", 0)
        suicide_prob = probabilities.get("Suicide", 0)
        risk_score = (safe_prob * 0) + (depression_prob * 50) + (suicide_prob * 100)
        
        # Build result display
        result_display = html.Div(className="result-card", children=[
            html.Div(COLORS[label]["icon"], className="result-icon"),
            html.Div(
                label,
                className="result-label",
                style={
                    "backgroundColor": COLORS[label]["light"],
                    "color": COLORS[label]["primary"],
                    "border": f"3px solid {COLORS[label]['primary']}",
                },
            ),
            html.P(f"Confidence: {confidence:.1%}", style={"fontSize": "1.2rem", "marginBottom": "20px"}),
            
            html.Div(className="charts-container", children=[
                html.H4("Risk Assessment", style={"marginBottom": "10px"}),
                dcc.Graph(figure=gauge_fig, config={"displayModeBar": False}),
                
                # Risk Score Formula Breakdown
                html.Div(className="formula-section", style={
                    "backgroundColor": "rgba(0,0,0,0.05)",
                    "borderRadius": "10px",
                    "padding": "15px",
                    "marginTop": "15px",
                    "marginBottom": "20px",
                    "fontFamily": "monospace",
                }, children=[
                    html.H5("📐 Risk Score Formula:", style={"marginBottom": "10px"}),
                    html.P("Risk = (Safe × 0) + (Depression × 50) + (Suicide × 100)",
                           style={"marginBottom": "10px", "fontWeight": "bold"}),
                    html.Div(style={"fontSize": "0.9rem"}, children=[
                        html.P(f"Risk = ({safe_prob:.3f} × 0) + ({depression_prob:.3f} × 50) + ({suicide_prob:.3f} × 100)"),
                        html.P(f"Risk = {safe_prob * 0:.2f} + {depression_prob * 50:.2f} + {suicide_prob * 100:.2f}"),
                        html.P(f"Risk = {risk_score:.1f}%", style={"fontWeight": "bold", "fontSize": "1.1rem", "marginTop": "5px"}),
                    ]),
                ]),
                
                html.H4("Class Probabilities", style={"marginTop": "20px", "marginBottom": "10px"}),
                dcc.Graph(figure=prob_fig, config={"displayModeBar": False}),
            ]),
        ])
        
        # Update history
        new_history = history.copy() if history else []
        new_history.insert(0, {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "label": label,
            "confidence": confidence,
            "time": datetime.now().strftime("%H:%M:%S"),
            "color": COLORS[label]["primary"],
            "icon": COLORS[label]["icon"],
        })
        # Keep only last 10 items
        new_history = new_history[:10]
        
        return result_display, new_history
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return html.Div(className="placeholder", children=[
            html.Span("❌", className="placeholder-icon"),
            html.P(f"Error: {str(e)}"),
        ]), history

@app.callback(
    Output("history-container", "children"),
    Input("history-store", "data"),
)
def update_history_display(history):
    """Update the history display."""
    if not history:
        return html.P("No predictions yet", style={"opacity": 0.5, "textAlign": "center"})
    
    return [
        html.Div(
            className="history-item",
            style={"borderColor": item["color"]},
            children=[
                html.Div(item["text"], className="history-text"),
                html.Div(
                    f"{item['icon']} {item['label']} ({item['confidence']:.1%})",
                    className="history-result",
                    style={"color": item["color"]},
                ),
                html.Div(item["time"], className="history-time"),
            ],
        )
        for item in history
    ]

@app.callback(
    Output("history-store", "data", allow_duplicate=True),
    Input("clear-history", "n_clicks"),
    prevent_initial_call=True,
)
def clear_history(n_clicks):
    """Clear prediction history."""
    return []

# --- Main Execution ---
if __name__ == "__main__":
    if not AVAILABLE_MODELS:
        logger.error("No trained models found in 'outputs' directory.")
        logger.error("Please train a model first using: python train.py --config configs/incremental_batch.yaml --batch-number 1")
        print("\n" + "="*60)
        print("⚠️  NO MODELS FOUND")
        print("="*60)
        print("Please train a model first. Example:")
        print("  python train.py --config configs/incremental_batch.yaml --batch-number 1")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("🧠 Mental Health Vulnerability Detector")
        print("="*60)
        print(f"Found {len(AVAILABLE_MODELS)} model(s): {', '.join(AVAILABLE_MODELS.keys())}")
        print("Starting server at http://127.0.0.1:8050")
        print("="*60 + "\n")
        app.run(debug=True)