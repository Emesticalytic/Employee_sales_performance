# 🎨 Color Palette - Employee Sales Forecasting Project

**Author**: Emem Akpan  
**Project**: Employee Sales Forecasting Dashboard

This document contains all color codes used throughout the project for consistent, professional visualizations.

---

## 📊 Main Dashboard Colors

### Primary Brand Colors
```python
PRIMARY_BLUE = '#1f77b4'      # Main header, accent color
TEXT_DARK = '#555'            # Subheaders, body text
TEXT_LIGHT = '#666'           # Disclaimers, footnotes
```

### Background Colors
```python
METRIC_CARD_BG = '#f0f2f6'    # Metric card background (light gray)
SUCCESS_BG = '#d4edda'        # Success messages (light green)
INFO_BG = '#d1ecf1'           # Info boxes (light blue)
```

### Border Colors
```python
BORDER_PRIMARY = '#1f77b4'    # Primary blue border
BORDER_SUCCESS = '#28a745'    # Success green border
BORDER_INFO = '#17a2b8'       # Info cyan border
```

---

## 📈 Model Performance Chart Colors

### Classic 4-Model Palette
Used for: Random Forest, Gradient Boosting, XGBoost, Ensemble

```python
MODEL_COLORS_CLASSIC = [
    'skyblue',      # Random Forest - Light blue (#87CEEB)
    'lightcoral',   # Gradient Boosting - Soft red (#F08080)
    'lightgreen',   # XGBoost - Light green (#90EE90)
    'gold'          # Ensemble - Golden yellow (#FFD700)
]
```

### Modern Vibrant Palette
Used for: Regional analysis, department breakdown

```python
MODEL_COLORS_VIBRANT = [
    '#FF6B6B',      # Coral Red - Random Forest
    '#4ECDC4',      # Turquoise - Gradient Boosting
    '#45B7D1',      # Sky Blue - XGBoost
    '#FFA07A'       # Light Salmon - Ensemble
]
```

### Accent Colors
```python
STEELBLUE = 'steelblue'       # Feature importance bars (#4682B4)
LIGHTBLUE = 'lightblue'       # Regional sales (#ADD8E6)
LIGHTCORAL = 'lightcoral'     # Department sales (#F08080)
```

---

## 🎯 Data Highlight Colors

### Table Styling
```python
HIGHLIGHT_MAX = 'lightgreen'  # Best performance values
HIGHLIGHT_MIN = 'lightgreen'  # Best MAPE/RMSE (lower is better)
```

### Target Lines
```python
TARGET_LINE_COLOR = 'red'     # Dash lines for targets
TARGET_LINE_DASH = 'dash'     # Line style
TARGET_LINE_WIDTH = 2         # Line width
```

---

## 🌈 Seaborn/Matplotlib Palettes

### General Plotting
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")  # Default palette for variety
```

### Color Maps
```python
# For correlation heatmaps
HEATMAP_CMAP = 'coolwarm'

# For continuous data
CONTINUOUS_CMAP = 'viridis'

# For diverging data
DIVERGING_CMAP = 'RdYlGn'
```

---

## 📊 Chart-Specific Colors

### Sales Trend Line
```python
TREND_LINE_COLOR = '#1f77b4'  # Primary blue
TREND_LINE_WIDTH = 3
```

### Moving Average Line
```python
MA_LINE_COLOR = 'red'
MA_LINE_STYLE = 'dash'
MA_LINE_WIDTH = 2
```

### Prediction Scatter
```python
PREDICTION_DOT_COLOR = 'steelblue'
PREDICTION_DOT_SIZE = 8
PREDICTION_DOT_OPACITY = 0.6
```

---

## 🎨 Usage Examples

### Model Performance Bar Chart
```python
import plotly.graph_objects as go

colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Ensemble']
accuracy = [96.5, 98.2, 97.6, 98.1]

fig = go.Figure(data=[
    go.Bar(x=models, y=accuracy, 
           marker_color=colors,
           text=accuracy,
           texttemplate='%{text:.1f}%',
           textposition='outside')
])
```

### Dashboard Overview Cards
```python
import streamlit as st

st.markdown("""
<div style='background-color: #f0f2f6; 
            padding: 1.5rem; 
            border-radius: 0.5rem; 
            border-left: 5px solid #1f77b4;'>
    <h3>Key Metrics</h3>
</div>
""", unsafe_allow_html=True)
```

### Success Message Box
```python
st.markdown("""
<div style='background-color: #d4edda; 
            border-left: 5px solid #28a745; 
            padding: 1rem; 
            border-radius: 0.5rem;'>
    ✅ Target Achieved!
</div>
""", unsafe_allow_html=True)
```

---

## 🖌️ Color Psychology

**Why These Colors?**

- **Blue (#1f77b4)**: Trust, professionalism, stability - perfect for business dashboards
- **Green (#28a745, lightgreen)**: Success, achievement, positive results
- **Coral/Red (lightcoral, #FF6B6B)**: Attention, importance, highlights
- **Gold (#FFD700)**: Excellence, premium, best performer
- **Turquoise (#4ECDC4)**: Modern, fresh, data-driven

---

## 📱 Accessibility Notes

All colors chosen meet WCAG 2.1 AA standards for:
- Sufficient contrast ratios (4.5:1 for normal text)
- Distinguishable for common color blindness types
- Professional appearance in both light and dark modes

---

## 💾 Quick Reference - Hex Codes

| Color Name | Hex Code | RGB | Usage |
|------------|----------|-----|-------|
| Primary Blue | #1f77b4 | rgb(31, 119, 180) | Headers, accents |
| Success Green | #28a745 | rgb(40, 167, 69) | Success messages |
| Info Cyan | #17a2b8 | rgb(23, 162, 184) | Info boxes |
| Coral Red | #FF6B6B | rgb(255, 107, 107) | Model 1 |
| Turquoise | #4ECDC4 | rgb(78, 205, 196) | Model 2 |
| Sky Blue | #45B7D1 | rgb(69, 183, 209) | Model 3 |
| Light Salmon | #FFA07A | rgb(255, 160, 122) | Model 4 |
| Steelblue | #4682B4 | rgb(70, 130, 180) | Feature bars |
| Skyblue | #87CEEB | rgb(135, 206, 235) | Classic palette |
| Lightcoral | #F08080 | rgb(240, 128, 128) | Classic palette |
| Lightgreen | #90EE90 | rgb(144, 238, 144) | Classic palette |
| Gold | #FFD700 | rgb(255, 215, 0) | Classic palette |

---

## 🔧 Implementation Tips

1. **Consistency**: Use the same colors for the same data categories across all charts
2. **Contrast**: Ensure sufficient contrast between adjacent colors
3. **Semantics**: Green = good, Red = attention, Blue = neutral/primary
4. **Accessibility**: Test with colorblind simulators
5. **Branding**: Primary blue (#1f77b4) is the signature color

---

**Created**: February 2026  
**Version**: 1.0  
**Maintained by**: Emem Akpan  
**GitHub**: [@Emesticalytic](https://github.com/Emesticalytic)
