import re

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from bokeh.io import curdoc
from bokeh.themes import Theme
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource

from config import theme, palette
curdoc().theme = Theme(json=theme)


def _format_column_name(column_name):
    _BAD_COLUMN_CHARACTERS = "()"
    for char in _BAD_COLUMN_CHARACTERS:
        column_name = column_name.replace(char, "")
    column_name = "_".join(column_name.split())
    return column_name


def make_distribution_plot(
        df,
        n_bins,
        label_column=None,
        fig_kwargs={},
        plot_kwargs={}
    ):
    if label_column is not None:
        target_names = df[label_column].unique()
        feature_name = df.set_index(label_column).columns[0]
    else:
        target_names = [None]
        feature_name = df.columns[0]

    p = figure(tools="", **fig_kwargs)

    bins = np.linspace(df[feature_name].min(), df[feature_name].max(), n_bins+1)
    bin_width = (df[feature_name].max() - df[feature_name].min()) / (n_bins+1)

    for class_name, color in zip(target_names, palette):
        if label_column is not None:
            feature_values = df.set_index(label_column).loc[class_name, feature_name]
        else:
            feature_values = df[feature_name].values

        values, _ = np.histogram(feature_values, bins)

        # do a bit of machine learning to make our plots look
        # better (but less honest, get a lawyer)
        spl = UnivariateSpline(bins[:-1] + bin_width, values, s=2)
        x = np.linspace(bins[0]+bin_width, bins[-1]-bin_width, 100)
        y = spl(x)
        y = np.clip(y, 0, np.inf)
        y *= np.diff(x)[0]

        p.line(x, y, line_color=color, name=class_name, **plot_kwargs)
    return p


def make_2d_scatter_plot(
        df,
        label_column=None,
        fig_kwargs={},
        plot_kwargs={}
    ):
    default_plot_kwargs = {
        "line_color": "color",
        "fill_color": "color",
        "line_alpha": 0.8,
        "fill_alpha": 0.4,
        "line_width": 1.2,
        "size": 6
    }
    default_plot_kwargs.update(plot_kwargs)

    df = df.copy()
    tooltips = []
    for column in df.columns:
        if column in ("x", "y", label_column):
            continue
        formatted_column = _format_column_name(column)
        if formatted_column != column:
            df[formatted_column] = df[column]
            df = df.drop(columns=column)

        tooltip = (column, "@"+formatted_column)
        tooltips.append(tooltip)

    if label_column is not None:
        target_names = df[label_column].unique()
        palette_df = pd.DataFrame(palette, columns=["color"], index=target_names)
        df = df.join(palette_df, on=label_column, how="right")

        formatted_column = _format_column_name(label_column)
        if formatted_column != label_column:
            df[formatted_column] = df[label_column]
            df = df.drop(columns=label_column)

        tooltip = ('Class', "@"+formatted_column)
        tooltips.append(tooltip)

        default_plot_kwargs["legend_group"] = formatted_column
    source = ColumnDataSource(df)

    p = figure(tools="", **fig_kwargs)
    p.circle("x", "y", source=source, name="scatter", **default_plot_kwargs)

    hover = HoverTool(
        tooltips=tooltips,
        renderers=p.select(name="scatter"),
        point_policy="snap_to_data")
    p.add_tools(hover)

    return p