import logging
logger = logging.getLogger("evaluation.visualization.plots.heatmap_plot")
logger.info("Loading file...")

import plotly.express as px
from pathlib import Path
import pandas as pd

def create_heatmap_plot(df: pd.DataFrame, output_path: Path):
    """
    Expected index:
        Factual
        Inference
        Multi-2
        Multi-3
        Multi-4
        Multi-5
        Overall

    Expected columns:
        MRR
        Recall@K
        Ref. Coverage
        Faithfulness
    """

    fig = px.imshow(
        df,
        text_auto=True,
        aspect="auto",
        title="Detailed RAG Evaluation Heatmap",
        labels={
            "x": "Metric",
            "y": "Question Category",
            "color": "Score (%)"
        }
    )

    fig.update_layout(
        template="plotly_white",
        height=650
    )

    fig.write_image(output_path, scale=2)