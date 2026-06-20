import logging
logger = logging.getLogger("evaluation.visualization.plots.summary_plot")
logger.info("Loading file...")

import pandas as pd
import plotly.express as px
from pathlib import Path

def create_summary_plot(df: pd.DataFrame, output_path: Path):

    df_long = df.melt(
        id_vars="Category",
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        df_long,
        x="Category",
        y="Score",
        color="Metric",
        barmode="group",
        title="RAG Performance Summary",
        text="Score"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
        textfont_size=15
    )

    fig.update_layout(
        template="plotly_white",

        width=1200,
        height=700,

        yaxis=dict(
            range=[0, 110],
            title="Score (%)",
            title_font_size=18,
            tickfont_size=16
        ),

        xaxis=dict(
            title="",
            tickfont_size=18
        ),

        title=dict(
            text="RAG Performance Summary",
            font_size=28
        ),

        legend=dict(
            font_size=16,
            title_font_size=17
        )
    )

    fig.write_image(
        output_path,
        scale=2
    )