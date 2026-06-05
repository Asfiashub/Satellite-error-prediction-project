"""
Enhanced Visualization Module for OrbitIQ
Features: 3D Globe with satellites, advanced error visualization
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pydeck as pdk
from plotly.subplots import make_subplots


def create_3d_globe(satellite_data):
    """
    Create an interactive 3D globe with satellite positions using PyDeck
    
    Args:
        satellite_data: Dictionary with 'positions' DataFrame containing:
                       - latitude, longitude, altitude
                       - name, error
    
    Returns:
        pydeck.Deck object for rendering in Streamlit
    """
    try:
        df = satellite_data['positions']
        
        # Create the 3D globe layer
        globe_layer = pdk.Layer(
            "GlobeLayer",
            id="base-globe",
            stroked=False,
            filled=True,
            extruded=True,
            wireframe=False,
            material=True,
            material_ambient_color=[20, 20, 40],
            material_diffuse_color=[40, 60, 100],
            material_shininess=0.8,
        )
        
        # Satellite positions as 3D columns
        satellite_layer = pdk.Layer(
            "ColumnLayer",
            df,
            get_position=["longitude", "latitude"],
            get_elevation="altitude",
            elevation_scale=1,
            radius=200000,
            get_fill_color="[255, 140, 0, 200]",
            pickable=True,
            auto_highlight=True,
        )
        
        # Satellite icons/markers
        icon_layer = pdk.Layer(
            "IconLayer",
            df,
            get_position=["longitude", "latitude"],
            get_icon="icon_data",
            get_size=4,
            size_scale=15,
            pickable=True,
        )
        
        # Arcs showing satellite paths (optional)
        arc_layer = pdk.Layer(
            "ArcLayer",
            data=df,
            get_source_position=["longitude", "latitude"],
            get_target_position=["longitude", "latitude"],
            get_source_color=[0, 212, 255, 160],
            get_target_color=[157, 78, 221, 160],
            get_width=2,
        )
        
        # View state for globe
        view_state = pdk.ViewState(
            latitude=20.5937,
            longitude=78.9629,  # Center on India
            zoom=3,
            pitch=45,
            bearing=0,
        )
        
        # Tooltip
        tooltip = {
            "html": "<b>Satellite:</b> {name}<br/>"
                    "<b>Error:</b> {error:.2f} m<br/>"
                    "<b>Altitude:</b> {altitude:,.0f} m<br/>"
                    "<b>Lat:</b> {latitude:.4f}<br/>"
                    "<b>Lon:</b> {longitude:.4f}",
            "style": {
                "backgroundColor": "rgba(10, 14, 39, 0.95)",
                "color": "white",
                "fontSize": "14px",
                "fontFamily": "Rajdhani, sans-serif",
                "border": "2px solid #00d4ff",
                "borderRadius": "10px",
                "padding": "10px",
                "boxShadow": "0 0 20px rgba(0, 212, 255, 0.5)"
            }
        }
        
        # Create deck
        deck = pdk.Deck(
            layers=[globe_layer, satellite_layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="mapbox://styles/mapbox/dark-v10",
        )
        
        return deck
        
    except Exception as e:
        print(f"PyDeck error: {e}")
        return create_fallback_globe(satellite_data)


def create_fallback_globe(satellite_data):
    """
    Fallback 3D globe using Plotly when PyDeck is not available
    """
    df = satellite_data['positions']
    
    fig = go.Figure()
    
    # Add Earth sphere
    theta = np.linspace(0, 2*np.pi, 50)
    phi = np.linspace(0, np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    
    R = 6371000  # Earth radius in meters
    x = R * np.sin(phi) * np.cos(theta)
    y = R * np.sin(phi) * np.sin(theta)
    z = R * np.cos(phi)
    
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, 'rgb(10, 20, 50)'], [1, 'rgb(20, 40, 100)']],
        showscale=False,
        opacity=0.8,
        hoverinfo='skip'
    ))
    
    # Convert lat/lon to 3D coordinates for satellites
    for _, sat in df.iterrows():
        lat_rad = np.radians(sat['latitude'])
        lon_rad = np.radians(sat['longitude'])
        r = R + sat['altitude']
        
        sat_x = r * np.cos(lat_rad) * np.cos(lon_rad)
        sat_y = r * np.cos(lat_rad) * np.sin(lon_rad)
        sat_z = r * np.sin(lat_rad)
        
        # Add satellite marker
        fig.add_trace(go.Scatter3d(
            x=[sat_x],
            y=[sat_y],
            z=[sat_z],
            mode='markers+text',
            marker=dict(
                size=15,
                color='#ff9500',
                symbol='diamond',
                line=dict(color='#00d4ff', width=2)
            ),
            text=sat['name'],
            textposition='top center',
            textfont=dict(size=12, color='white', family='Orbitron'),
            hovertemplate=f"<b>{sat['name']}</b><br>" +
                         f"Error: {sat['error']:.2f} m<br>" +
                         f"Alt: {sat['altitude']:,.0f} m<br>" +
                         f"Lat: {sat['latitude']:.4f}<br>" +
                         f"Lon: {sat['longitude']:.4f}<extra></extra>",
            name=sat['name']
        ))
        
        # Add connection line from Earth surface to satellite
        surface_x = R * np.cos(lat_rad) * np.cos(lon_rad)
        surface_y = R * np.cos(lat_rad) * np.sin(lon_rad)
        surface_z = R * np.sin(lat_rad)
        
        fig.add_trace(go.Scatter3d(
            x=[surface_x, sat_x],
            y=[surface_y, sat_y],
            z=[surface_z, sat_z],
            mode='lines',
            line=dict(color='rgba(0, 212, 255, 0.4)', width=2, dash='dash'),
            hoverinfo='skip',
            showlegend=False
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='rgba(0, 0, 0, 0)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        showlegend=False,
        height=600,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig


def make_globe_placeholder():
    """
    Create a placeholder globe when no data is available
    """
    # Sample satellite data for demonstration
    sample_positions = pd.DataFrame({
        'latitude': [28.6139, 19.0760, 13.0827, 22.5726],
        'longitude': [77.2090, 72.8777, 80.2707, 88.3639],
        'altitude': [35786000, 35786000, 20200000, 20200000],
        'name': ['IRNSS-1A', 'IRNSS-1B', 'IRNSS-1C', 'IRNSS-1D'],
        'error': [156.7, 142.3, 178.2, 134.8]
    })
    
    sample_data = {'positions': sample_positions}
    
    try:
        # Try PyDeck first
        return create_3d_globe(sample_data)
    except:
        # Fallback to Plotly
        return create_fallback_globe(sample_data)


def create_error_timeline(df, satellite_name):
    """
    Create beautiful animated timeline of errors
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('X Error Over Time', 'Y Error Over Time',
                       'Z Error Over Time', 'Clock Error Over Time'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    error_cols = ['x_error (m)', 'y_error (m)', 'z_error (m)', 'satclockerror (m)']
    colors = ['#00d4ff', '#9d4edd', '#ff006e', '#ff9500']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for col, color, pos in zip(error_cols, colors, positions):
        # Main line trace
        fig.add_trace(
            go.Scatter(
                x=df['utc_time'],
                y=df[col],
                mode='lines',
                name=col,
                line=dict(color=color, width=3),
                fill='tozeroy',
                fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)',
                hovertemplate='<b>%{x}</b><br>Error: %{y:.4f} m<extra></extra>'
            ),
            row=pos[0], col=pos[1]
        )
        
        # Add markers at key points
        fig.add_trace(
            go.Scatter(
                x=df['utc_time'][::10],  # Every 10th point
                y=df[col][::10],
                mode='markers',
                marker=dict(size=8, color=color, symbol='diamond',
                           line=dict(color='white', width=1)),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=pos[0], col=pos[1]
        )
        
        fig.update_xaxes(title_text="Time", row=pos[0], col=pos[1],
                        showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(title_text="Error (m)", row=pos[0], col=pos[1],
                        showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    
    fig.update_layout(
        title=f'<b>{satellite_name} - Error Timeline Analysis</b>',
        title_font=dict(size=24, family='Orbitron', color='#00d4ff'),
        height=800,
        showlegend=False,
        hovermode='x unified',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(10, 14, 39, 0.5)',
        font=dict(family='Rajdhani', color='white')
    )
    
    return fig


def create_3d_error_viz(df, title):
    """
    Create stunning 3D visualization of error trajectory with trails
    """
    fig = go.Figure()
    
    # Create color scale based on time/magnitude
    error_magnitude = np.sqrt(df['x_error (m)']**2 + 
                             df['y_error (m)']**2 + 
                             df['z_error (m)']**2)
    
    # Main trajectory
    fig.add_trace(go.Scatter3d(
        x=df['x_error (m)'],
        y=df['y_error (m)'],
        z=df['z_error (m)'],
        mode='lines',
        line=dict(
            color=error_magnitude,
            colorscale='Viridis',
            width=6,
            colorbar=dict(
                title=dict(
                    text='Error<br>Magnitude (m)',
                    font=dict(family='Orbitron', color='white')
                ),
                tickfont=dict(color='white')
            )
        ),
        name='Error Path',
        hovertemplate='<b>Position Error</b><br>' +
                     'X: %{x:.3f} m<br>' +
                     'Y: %{y:.3f} m<br>' +
                     'Z: %{z:.3f} m<br>' +
                     '<extra></extra>'
    ))
    
    # Add markers at regular intervals
    interval = max(1, len(df) // 20)
    fig.add_trace(go.Scatter3d(
        x=df['x_error (m)'][::interval],
        y=df['y_error (m)'][::interval],
        z=df['z_error (m)'][::interval],
        mode='markers',
        marker=dict(
            size=10,
            color=error_magnitude[::interval],
            colorscale='Plasma',
            symbol='diamond',
            line=dict(color='white', width=1)
        ),
        name='Key Points',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add start point
    fig.add_trace(go.Scatter3d(
        x=[df['x_error (m)'].iloc[0]],
        y=[df['y_error (m)'].iloc[0]],
        z=[df['z_error (m)'].iloc[0]],
        mode='markers+text',
        marker=dict(size=15, color='#00ff00', symbol='diamond'),
        text=['START'],
        textposition='top center',
        textfont=dict(size=14, color='#00ff00', family='Orbitron'),
        name='Start',
        showlegend=False
    ))
    
    # Add end point
    fig.add_trace(go.Scatter3d(
        x=[df['x_error (m)'].iloc[-1]],
        y=[df['y_error (m)'].iloc[-1]],
        z=[df['z_error (m)'].iloc[-1]],
        mode='markers+text',
        marker=dict(size=15, color='#ff0000', symbol='diamond'),
        text=['END'],
        textposition='top center',
        textfont=dict(size=14, color='#ff0000', family='Orbitron'),
        name='End',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b>',
            font=dict(size=24, family='Orbitron', color='#00d4ff'),
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(
                title=dict(
                    text='X Error (m)',
                    font=dict(color='white', family='Rajdhani')
                ),
                backgroundcolor='rgba(10, 14, 39, 0.5)',
                gridcolor='rgba(255, 255, 255, 0.1)',
                showbackground=True,
            ),
            yaxis=dict(
                title=dict(
                    text='Y Error (m)',
                    font=dict(color='white', family='Rajdhani')
                ),
                backgroundcolor='rgba(10, 14, 39, 0.5)',
                gridcolor='rgba(255, 255, 255, 0.1)',
                showbackground=True,
            ),
            zaxis=dict(
                title=dict(
                    text='Z Error (m)',
                    font=dict(color='white', family='Rajdhani')
                ),
                backgroundcolor='rgba(10, 14, 39, 0.5)',
                gridcolor='rgba(255, 255, 255, 0.1)',
                showbackground=True,
            ),
            bgcolor='rgba(0, 0, 0, 0)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(family='Rajdhani', color='white'),
        height=700,
        showlegend=True,
        legend=dict(
            font=dict(color='white', family='Rajdhani'),
            bgcolor='rgba(10, 14, 39, 0.7)'
        )
    )
    
    return fig


def create_prediction_comparison(historical_data, prediction, error_type):
    """
    Create beautiful comparison visualization between historical and predicted values
    """
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=list(range(len(historical_data))),
        y=historical_data,
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='#00d4ff', width=3),
        marker=dict(size=8, symbol='circle'),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.1)'
    ))
    
    # Prediction point
    fig.add_trace(go.Scatter(
        x=[len(historical_data)],
        y=[prediction],
        mode='markers+text',
        name='8th Day Prediction',
        marker=dict(size=20, color='#ff006e', symbol='star',
                   line=dict(color='white', width=2)),
        text=['PREDICTION'],
        textposition='top center',
        textfont=dict(size=12, color='#ff006e', family='Orbitron')
    ))
    
    # Connection line
    fig.add_trace(go.Scatter(
        x=[len(historical_data)-1, len(historical_data)],
        y=[historical_data.iloc[-1], prediction],
        mode='lines',
        line=dict(color='#ff006e', width=2, dash='dash'),
        showlegend=False
    ))
    
    fig.update_layout(
        title=f'<b>{error_type} - Prediction vs Historical</b>',
        title_font=dict(size=20, family='Orbitron', color='#00d4ff'),
        xaxis_title='Time Step (Days)',
        yaxis_title='Error (m)',
        hovermode='x unified',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(10, 14, 39, 0.5)',
        font=dict(family='Rajdhani', color='white'),
        height=400,
        showlegend=True,
        legend=dict(
            font=dict(color='white', family='Rajdhani'),
            bgcolor='rgba(10, 14, 39, 0.7)'
        ),
        xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
    )
    
    return fig


def create_heatmap_correlation(df):
    """
    Create correlation heatmap for error components
    """
    error_cols = ['x_error (m)', 'y_error (m)', 'z_error (m)', 'satclockerror (m)']
    corr_matrix = df[error_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=['X Error', 'Y Error', 'Z Error', 'Clock Error'],
        y=['X Error', 'Y Error', 'Z Error', 'Clock Error'],
        colorscale='Viridis',
        text=corr_matrix.values,
        texttemplate='%{text:.3f}',
        textfont=dict(size=14, family='Orbitron'),
        hoverongaps=False,
        colorbar=dict(
            title=dict(
                text='Correlation',
                font=dict(color='white', family='Orbitron')
            ),
            tickfont=dict(color='white')
        )
    ))
    
    fig.update_layout(
        title='<b>Error Components Correlation Matrix</b>',
        title_font=dict(size=20, family='Orbitron', color='#00d4ff'),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(10, 14, 39, 0.5)',
        font=dict(family='Rajdhani', color='white'),
        height=500
    )
    
    return fig