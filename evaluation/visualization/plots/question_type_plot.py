import logging
logger = logging.getLogger("evaluation.visualization.plots.question_type_plot")
logger.info("Loading file...")

import plotly.express as px
from pathlib import Path
import pandas as pd

def create_question_type_plot(df: pd.DataFrame, output_path: Path):

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
        title="RAG Performance by Question Type",
        text="Score"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
        textfont_size=14
    )

    fig.update_layout(
        yaxis_range=[0, 110],
        template="plotly_white",

        width=1200,
        height=700,

        title_font_size=26,

        xaxis=dict(
            title_font_size=18,
            tickfont_size=16
        ),

        yaxis=dict(
            title="Score (%)",
            title_font_size=18,
            tickfont_size=16
        ),

        legend=dict(
            title_font_size=16,
            font_size=15
        )
    )

    fig.write_image(
        output_path,
        scale=2
    )