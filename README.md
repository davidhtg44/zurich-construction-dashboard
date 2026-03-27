#  Zurich Construction Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://zurich-construction.streamlit.app/)

**Live Demo:** [Visit the Dashboard](https://zurich-construction.streamlit.app/)

##  Overview
Zurich-construction is an interactive, data dashboard that analyzes and visualizes over a decade of real estate and construction trends across the city of Zurich (2009 - Present). 

Built to provide clear market insights, the tool combines historical open data with machine learning to estimate future construction costs, map urban development, and calculate market inflation.

## Key Features
*  **AI Cost Forecasting:** Utilizes a Random Forest Regression model to predict 10-year future construction volumes and costs based on historical project data.
*  **Interactive Heatmap:** A high-performance PyDeck 3D map visualizing project density and financial volume strictly mapped to Zurich's official neighborhoods.
*  **Market Overview:** Automatically calculates the Compound Annual Growth Rate (CAGR) to track real estate inflation over time.
*  **Urban Development:** Detailed breakdown of residential vs. non-residential building trends, including a temporal analysis of new constructions versus demolitions.

## Tech Stack
* **Language:** Python
* **Framework:** Streamlit
* **Data Processing & ML:** Pandas, Scikit-Learn
* **Visualization:** Plotly, PyDeck

## Data Sources
All data relies on official open datasets provided by **[Open Data Stadt Zürich](https://data.stadt-zuerich.ch/)**. The analysis processes historical construction costs, building typologies, and precise geographical neighborhood borders.
