import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import plotly.express as px
import json
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Zurich Construction Intelligence", layout="wide")
st.title("Zurich Construction Analysis")

#loading construction
@st.cache_data
def load_data():
    df = pd.read_csv("bau501od5011.csv")
    df.columns = df.columns.str.strip()
    # replacing K  with 0 and converting to numeric
    df['BaukostenEffektiv'] = df['BaukostenEffektiv'].apply(lambda x: 0 if x == 'K' else pd.to_numeric(x))
    return df

def load_historical_data():
    #loading demolitions
    df_historic = pd.read_csv("bau506od5064.csv")
    df_historic.columns = df_historic.columns.str.strip()
    #filtering from >= 2009
    df_historic = df_historic[df_historic['StichtagDatJahr'] >= 2009].copy()
    return df_historic

data_historic = load_historical_data()
data = load_data()


# X param and Y target variable to predict
numeric_features = ['StichtagDatJahr', 'QuarSort', 'KreisSort', 'ProjektStatusSSZPubl1Sort', 'ArtArbeitenSort', 'AnzBauprojekte']
X = data[numeric_features]
Y = data['BaukostenEffektiv']

st.sidebar.header("Input parameters")

def user_input_features(data):
    StichtagDatJahr = st.sidebar.slider('Starting Year', int(X['StichtagDatJahr'].min()), int(X['StichtagDatJahr'].max()), int(X['StichtagDatJahr'].mean()))
    
    #mapping neighborhood names to their numeric codes
    df_names_quar = data[['QuarLang', 'QuarSort']].drop_duplicates()
    dictionary_quar = dict(zip(df_names_quar['QuarLang'], df_names_quar['QuarSort']))
    
    names_ord = ["All Zurich"] + sorted(dictionary_quar.keys())
    name_quar_toShow = st.sidebar.selectbox('Select Area', names_ord)
    
    if name_quar_toShow == "All Zurich":
        tutti_i_quartieri = []
        for nome, codice in dictionary_quar.items():
            dq = data[data['QuarSort'] == codice]
            if not dq.empty:
                record_annui_medi = int(round(dq.groupby('StichtagDatJahr').size().mean()))
                record_annui_medi = max(1, record_annui_medi)
                
                for _ in range(record_annui_medi):
                    tutti_i_quartieri.append({
                        'StichtagDatJahr' : StichtagDatJahr,
                        'QuarSort' : codice,
                        'KreisSort' : int(dq['KreisSort'].iloc[0]),
                        'ProjektStatusSSZPubl1Sort' : int(data['ProjektStatusSSZPubl1Sort'].mode()[0]),
                        'ArtArbeitenSort' : int(dq['ArtArbeitenSort'].mode()[0]),
                        'AnzBauprojekte' : int(dq['AnzBauprojekte'].median())
                    })
        features = pd.DataFrame(tutti_i_quartieri)
        
    else:
        QuarSort = dictionary_quar[name_quar_toShow]
        dati_quartiere = data[data['QuarSort'] == QuarSort]
        
        dati_utente = {
            'StichtagDatJahr' : StichtagDatJahr,
            'QuarSort' : QuarSort,
            'KreisSort' : int(dati_quartiere['KreisSort'].iloc[0]),
            'ProjektStatusSSZPubl1Sort' : int(data['ProjektStatusSSZPubl1Sort'].mode()[0]),
            'ArtArbeitenSort' : int(dati_quartiere['ArtArbeitenSort'].mode()[0]),
            'AnzBauprojekte' : int(dati_quartiere['AnzBauprojekte'].median())
        }
        features = pd.DataFrame(dati_utente, index=[0])
        
    return features, name_quar_toShow

df_input, quartiere_scelto = user_input_features(data)

#random forest
model = RandomForestRegressor(random_state=42) 
model.fit(X, Y) 

anno_partenza = int(df_input['StichtagDatJahr'].iloc[0])
anni_futuri = list(range(anno_partenza, anno_partenza + 11)) 

previsioni_base = []
for anno in anni_futuri:
    dati_anno = df_input.copy() 
    dati_anno['StichtagDatJahr'] = anno
    costo_totale_anno = model.predict(dati_anno).sum() * 1000 #prediction
    previsioni_base.append(costo_totale_anno)

previsioni_future = pd.Series(previsioni_base)

#CAGR
costi_annui = data.groupby('StichtagDatJahr')['BaukostenEffektiv'].mean()
anno_min, anno_max = costi_annui.index.min(), costi_annui.index.max()
tasso_crescita = (costi_annui[anno_max] / costi_annui[anno_min]) ** (1 / (anno_max - anno_min)) - 1

#CAGR + AI if anno > anno_max
for i, anno in enumerate(anni_futuri):
    if anno > anno_max:
        anni_di_distanza = anno - anno_max
        previsioni_future.iloc[i] = previsioni_future.iloc[i] * ((1 + tasso_crescita) ** anni_di_distanza)

df_trend = pd.DataFrame({'Year': anni_futuri,'Estimated Cost (CHF)': previsioni_future})
costo_partenza = df_trend['Estimated Cost (CHF)'].iloc[0]

st.markdown("### Market Overview")

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric(label="Selected Area", value=quartiere_scelto)
kpi2.metric(label=f"Est. Starting Cost ({anno_partenza})", value=f"{costo_partenza:,.0f} CHF")
kpi3.metric(label="Historical Inflation (CAGR)", value=f"{tasso_crescita*100:.2f}%")
st.write('---')

if quartiere_scelto == "All Zurich":
    st.subheader("10-Year Cost Forecast (All Zurich)")
    st.write("Aggregated market trend projection for the entire city.")
else:
    st.subheader(f"10-Year Cost Forecast ({quartiere_scelto})")
    st.write("Local market trend projection for the selected neighborhood.")

if quartiere_scelto == "All Zurich":
    df_real = data.groupby('StichtagDatJahr')['BaukostenEffektiv'].sum().reset_index()
else:
    codice_quartiere = df_input['QuarSort'].iloc[0]
    df_real = data[data['QuarSort'] == codice_quartiere].groupby('StichtagDatJahr')['BaukostenEffektiv'].sum().reset_index()

df_real['BaukostenEffektiv'] = df_real['BaukostenEffektiv'] * 1000

fig_trend = go.Figure()

#real data
fig_trend.add_trace(go.Scatter(
    x=df_real['StichtagDatJahr'],
    y=df_real['BaukostenEffektiv'],
    mode='lines+markers',
    name='Real Historical Data',
    line=dict(color='#1f77b4', width=3),
    marker=dict(size=8)
))

#ml data
fig_trend.add_trace(go.Scatter(
    x=df_trend['Year'],
    y=df_trend['Estimated Cost (CHF)'],
    mode='lines+markers',
    name='AI Forecast',
    line=dict(color='#d32f2f', width=3, dash='dash'), 
    marker=dict(size=8)
))

fig_trend.update_layout(
    xaxis_title="Year",
    yaxis_title="Total Cost (CHF)",
    hovermode="x unified",
    margin=dict(t=10, b=0, l=0, r=0),
    legend=dict(
        orientation="h", 
        yanchor="bottom", 
        y=1.02, 
        xanchor="right", 
        x=1,
        bgcolor="rgba(0,0,0,0)" 
    )
)

st.plotly_chart(fig_trend, use_container_width=True)
st.write('---')

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("AI Analysis: Cost Drivers")
    st.write("Key factors influencing final construction costs.")

    traduzione_fattori = {
        'StichtagDatJahr': 'Year',
        'QuarSort': 'Neighborhood',
        'KreisSort': 'District',
        'ProjektStatusSSZPubl1Sort': 'Project Status',
        'ArtArbeitenSort': 'Type of Works',
        'AnzBauprojekte': 'Project Volume'
    }

    importances = model.feature_importances_ #feature_importance
    fattori_tradotti = [traduzione_fattori.get(col, col) for col in X.columns]
    
    df_important = pd.DataFrame({'Factor': fattori_tradotti, 'Importance (%)' : importances * 100})
    
    fig_pie = go.Figure(data=[go.Pie(labels=df_important['Factor'], values=df_important['Importance (%)'], hole=0.4, textinfo='label+percent', hoverinfo='label+percent', marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']))])
    fig_pie.update_layout(showlegend=False, margin=dict(t=20, b=20, l=0, r=0))
    st.plotly_chart(fig_pie, use_container_width=True)

#bar/line chart
with col_right:
    st.subheader('Urban Development')
    st.write("Constructions vs. Demolitions over time.")

    df_type = data_historic.pivot_table(index = 'StichtagDatJahr',columns = 'GbdArtSSZPubl1Lang',values='AnzNeubauGbd_noDM',aggfunc='sum').fillna(0)

    traduzione_edifici = {
        'Wohnhäuser mit Geschäftsräumen': 'Residential with Commercial',
        'Nichtwohngebäude': 'Non-Residential Buildings',
        'Mehrfamilienhäuser ohne Geschäftsräume': 'Multi-Family Residential',
        'Einfamilienhäuser ohne Geschäftsräume': 'Single-Family Residential'
    }
    df_type.rename(columns=traduzione_edifici, inplace=True)

    #total demolitions per year
    df_demolitions = data_historic.groupby('StichtagDatJahr')['AnzAbbruchGbd_noDM'].sum()

    fig = go.Figure()

    for typology in df_type.columns:
        fig.add_trace(go.Bar(x=df_type.index, y=df_type[typology], name=typology))

    # overlaying the demolitions line on the same chart
    fig.add_trace(go.Scatter(x=df_demolitions.index, y=df_demolitions.values, name='Total Demolitions', mode='lines+markers', line=dict(color='red', width=3 ), marker=dict(size=8, color='white', line=dict(width=2, color='red'))))
    
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Number of Buildings',
        barmode='stack', 
        hovermode='x unified', 
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5), 
        margin=dict(t=20, b=20, l=0, r=0)
    )
    st.plotly_chart(fig, use_container_width=True)

st.write('---')

#heatmap
st.subheader("Heatmap: Project Density (2009 - Today)")
st.write("Construction intensity by neighborhood (from white to pure red).")

#mapping numeric district
QUARSORT_TO_NAME = {
    11: 'Altstadt', 12: 'Altstadt', 13: 'Altstadt', 14: 'Altstadt', 
    21: 'Wollishofen', 23: 'Leimbach', 24: 'Enge', 
    31: 'Wiedikon', 33: 'Wiedikon', 34: 'Wiedikon', 
    41: 'Aussersihl', 42: 'Aussersihl', 44: 'Aussersihl', 
    51: 'Industriequartier', 52: 'Industriequartier', 
    61: 'Unterstrass', 63: 'Oberstrass', 
    71: 'Fluntern', 72: 'Hottingen', 73: 'Hirslanden', 74: 'Witikon', 
    81: 'Riesbach', 82: 'Riesbach', 83: 'Riesbach', 
    91: 'Albisrieden', 92: 'Altstetten', 
    101: 'Höngg', 102: 'Wipkingen', 
    111: 'Affoltern', 115: 'Oerlikon', 119: 'Seebach', 
    121: 'Schwamendingen', 122: 'Schwamendingen', 123: 'Schwamendingen' 
}

df_map = data.groupby('QuarSort').agg({'AnzBauprojekte': 'sum','BaukostenEffektiv': 'sum'}).reset_index()

df_map['macro_nome'] = df_map['QuarSort'].map(QUARSORT_TO_NAME)
df_district = df_map.groupby('macro_nome').agg({'AnzBauprojekte': 'sum','BaukostenEffektiv': 'sum'}).reset_index()

map_prog = dict(zip(df_district['macro_nome'], df_district['AnzBauprojekte']))
map_cost = dict(zip(df_district['macro_nome'], df_district['BaukostenEffektiv']))

max_progetti = float(df_district['AnzBauprojekte'].max()) if not df_district.empty else 1.0

#loadinf geoJson map and injects calculated properties into each polygon feature
with open("quartiere2.json", "r", encoding="utf-8") as f:
    geojson = json.load(f)

for feature in geojson['features']:
    props = feature['properties']
    qname = props.get('name', 'Unknown')

    totale_cantieri = map_prog.get(qname, 0)
    totale_costi = map_cost.get(qname, 0) * 1000
    
    if totale_cantieri == 0:
        colore = [40, 40, 40, 150]
    else:
        ratio = totale_cantieri / max_progetti
        colore = [255, int(255 * (1 - ratio)), int(200 * (1 - ratio)), 200]
    
    feature['properties']['fill_color'] = colore
    feature['properties']['totale_progetti'] = int(totale_cantieri)
    feature['properties']['totale_costi_str'] = f"{totale_costi:,.0f} CHF"
    feature['properties']['nome_quartiere_bello'] = qname

view_state = pdk.ViewState(
    latitude=47.3769,
    longitude=8.5417,
    zoom=11.2,
    pitch=0,
    bearing=0
)

layer = pdk.Layer(
    "GeoJsonLayer",
    data=geojson,
    opacity=1.0,
    stroked=True,
    filled=True,
    extruded=False,
    get_fill_color="properties.fill_color",
    get_line_color=[150, 150, 150, 100],
    line_width_min_pixels=1,
    pickable=True
)

tooltip_html = """
    <div style="background-color: #1e1e1e; color: #ffffff; border: 1px solid #555555; border-radius: 4px; padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.5);">
        <p style="margin: 0; font-size: 14px; margin-bottom: 5px;">Neighborhood: <strong>{nome_quartiere_bello}</strong></p>
        <p style="margin: 0; font-size: 14px;">Total Projects: <strong>{totale_progetti}</strong></p>
        <p style="margin: 0; font-size: 14px; color: #ff4b4b;">Total Volume: <strong>{totale_costi_str}</strong></p>
    </div>
"""

st.pydeck_chart(pdk.Deck(
    map_style="carto-dark",
    layers=[layer],
    initial_view_state=view_state,
    tooltip={
        "html": tooltip_html,
        "style": {"border": "none", "backgroundColor": "transparent", "padding": "0"}
    }
))

st.write("Data Source: Open Data Stadt Zürich")