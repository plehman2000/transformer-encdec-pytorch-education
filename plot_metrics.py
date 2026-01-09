import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_metrics(metrics: dict):
    """
    Plot metrics in separate graphs.

    Args:
        metrics: Dict of form {"metric_name": [observations]} or {"metric_name": [[observations], [observations]]}
    """
    n_metrics = len(metrics)

    fig = make_subplots(rows=n_metrics, cols=1, subplot_titles=list(metrics.keys()))

    def convert_to_python(val):
        """Convert torch tensors to python scalars/lists."""
        import torch

        if isinstance(val, torch.Tensor):
            return val.detach().cpu().numpy().tolist()
        return val

    for i, (name, values) in enumerate(metrics.items(), start=1):
        # Check if values contains nested lists
        if values and isinstance(values[0], (list, tuple)):
            # Transpose: group by index across iterations
            num_heads = len(values[0])
            for j in range(num_heads):
                head_values = [
                    convert_to_python(values[iter_idx][j])
                    for iter_idx in range(len(values))
                ]
                fig.add_trace(
                    go.Scatter(y=head_values, mode="lines+markers", name=f"{name}_{j}"),
                    row=i,
                    col=1,
                )
        else:
            converted_values = [convert_to_python(v) for v in values]
            fig.add_trace(
                go.Scatter(y=converted_values, mode="lines+markers", name=name),
                row=i,
                col=1,
            )

    fig.update_layout(
        height=300 * n_metrics, showlegend=False, title_text=" without softmax"
    )
    fig.show()
