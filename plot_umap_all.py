import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc

# Initialize the app
app = Dash()
server = app.server

parquet_file = 'bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_binary/umap/hamming/all_years.parquet'
df = pd.read_parquet(parquet_file)

# Sample a subset of the data for visualization
df_sampled = df.sample(100000)

# Plot the 3D scatter plot using Plotly Express
fig = px.scatter_3d(df_sampled, x='x', y='y', z='z', hover_name='title', hover_data='year', color='categories',  opacity=0.6)

# Keep legend markers the same size
fig.update_layout(
    legend=dict(
        itemsizing='constant',  # Use 'constant' to fix marker size in legend
        title='Categories',
        font = dict(
            size=16,  # Set the font size for the legend
        )
    )
)

# Decrease the size of the markers in the plot
fig.update_traces(marker_size = 1)

# Add a title and axis labels, and adjust the camera angle and height
fig.update_layout(
    title='3D Map of Arxiv Abstracts',
    scene=dict(
        xaxis_title='UMAP X',
        yaxis_title='UMAP Y',
        zaxis_title='UMAP Z'
    ),
    height=800,  # Set the height of the plot
    scene_camera=dict(
        eye=dict(x=1, y=-0.125, z=0.125)
    ),
    scene_dragmode='turntable',  # 'turntable', 'orbit', 'zoom', 'pan', etc
)

# App layout
app.layout = [
    dcc.Graph(figure=fig),
    # html.Div(children='Note: The data is sampled for performance reasons.'),
]

# Run the app
if __name__ == '__main__':
    app.run(port=8050)