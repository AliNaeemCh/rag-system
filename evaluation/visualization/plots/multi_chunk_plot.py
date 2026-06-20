import logging
logger = logging.getLogger("evaluation.visualization.plots.multi_chunk_plot")
logger.info("Loading file...")

import plotly.express as px
from pathlib import Path
import pandas as pd

def create_multi_chunk_plot(df: pd.DataFrame, output_path: Path):
    """
    Expected columns:
    Chunks
    MRR
    Recall
    Coverage
    Faithfulness
    """

    df_long = df.melt(
        id_vars="Chunks",
        var_name="Metric",
        value_name="Score"
    )

    fig = px.line(
        df_long,
        x="Chunks",
        y="Score",
        color="Metric",
        markers=True,
        title="Performance vs Number of Chunks",
        labels={
            "Chunks": "Number of Chunks",
            "Score": "Score (%)"
        }
    )

    fig.update_layout(
        template="plotly_white",
        yaxis_range=[0, 100],
        height=550
    )

    fig.write_image(output_path, scale=2)