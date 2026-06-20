from app.core.logger import setup_logging
setup_logging()

from evaluation.visualization.utils import create_question_type_df, create_heatmap_df, create_multi_chunk_df, create_summary_df
from evaluation.visualization.plots.question_type_plot import create_question_type_plot
from evaluation.visualization.plots.multi_chunk_plot import create_multi_chunk_plot
from evaluation.visualization.plots.heatmap_plot import create_heatmap_plot
from evaluation.visualization.plots.summary_plot import create_summary_plot
from app.core.utils import read_json
from app.core.config import settings

import logging
logger = logging.getLogger("evaluation.visualization.run")
logger.info("Loading file...")

from pathlib import Path

def generate_plots(final_result_input: Path, plots_output_directory: Path):

    results = read_json(final_result_input)[0]

    question_type_df = create_question_type_df(results)
    multi_chunk_df = create_multi_chunk_df(results)
    heatmap_df = create_heatmap_df(
        question_type_df,
        multi_chunk_df
    )
    summary_df = create_summary_df(results)

    create_question_type_plot(
        question_type_df,
        plots_output_directory / "question_type_plot.png"
    )

    create_multi_chunk_plot(
        multi_chunk_df,
        plots_output_directory / "multi_chunk_plot.png"
    )

    create_heatmap_plot(
        heatmap_df,
        plots_output_directory / "heatmap_plot.png"
    )

    create_summary_plot(summary_df,
                       plots_output_directory / "summary.png")

    print("Plots generated successfully")

generate_plots(
    final_result_input=settings.EVAL_RESULTS_DIR / "final_result.json",
    plots_output_directory=settings.EVAL_VISUALIZATION_DIR
)