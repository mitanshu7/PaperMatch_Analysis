import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc

# Configuration
DATA_POINTS = 1_00_000  # Number of data points to visualize
OPACITY = 0.6  # Opacity of the markers in the plot
LEGEND_FONT_SIZE = 16  # Font size for the legend
PLOT_MARKER_SIZE = 1  # Size of the markers in the plot
EYE_X = 1  # X coordinate of the camera eye
EYE_Y = -0.125  # Y coordinate of the camera eye
EYE_Z = 0.125  # Z coordinate of the camera eye
HEIGHT = 800  # Height of the plot
PORT = 8050 # Port number for the Dash app

# Initialize the app
app = Dash()
server = app.server

# Read parquet file
parquet_file = 'bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_binary/umap/hamming/all_years.parquet'
df = pd.read_parquet(parquet_file)

# Sample a subset of the data for visualization
df = df.sample(DATA_POINTS)

# Plot the 3D scatter plot using Plotly Express
fig = px.scatter_3d(df, x='x', y='y', z='z', hover_name='title', hover_data='year', color='categories',  opacity=OPACITY)

# Keep legend markers the same size
fig.update_layout(
    legend=dict(
        itemsizing='constant',  # Use 'constant' to fix marker size in legend
        title='Categories',
        font = dict(
            size=LEGEND_FONT_SIZE,  # Set the font size for the legend
        )
    )
)

# Decrease the size of the markers in the plot
fig.update_traces(marker_size = PLOT_MARKER_SIZE)

# Add a title and axis labels, and adjust the camera angle and height
fig.update_layout(
    title='3D Map of arXiv',
    height=HEIGHT,  # Set the height of the plot
    scene_camera=dict(
        eye=dict(x=EYE_X, y=EYE_Y, z=EYE_Z)
    ),
)

fig.update_layout(
    scene=dict(
        xaxis=dict(
            showgrid=False, # Hide grid lines
            zeroline=False, # Hide the zero line
            showbackground=False,   # Hide background grid
            showticklabels=False,   # Hide tick labels
            ticks='',   # Hide ticks
            title=''    # Remove axis title
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showbackground=False,
            showticklabels=False,
            ticks='',
            title=''
        ),
        zaxis=dict(
            showgrid=False,
            zeroline=False,
            showbackground=False,
            showticklabels=False,
            ticks='',
            title=''
        )
    )
)

# App layout
app.layout = [
    dcc.Graph(figure=fig),
]

# Run the app
if __name__ == '__main__':
    app.run(port=PORT)
