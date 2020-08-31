from bokeh.palettes import RdYlBu10 as palette
palette = [palette[2], palette[6], palette[7]]


theme = {
    "attrs": {
        "Figure": {
            "background_fill_color": '#e8ded5',
            "background_fill_alpha": 0.8,
            "plot_height": 500,
            "plot_width": 500,
        },
        "Grid": {
            "grid_line_color": '#ffffff',
            "grid_line_width": 2
        }
    },
    "line_defaults": {
        "line_width": 2.3,
        "line_alpha": 0.6,
        "line_color": palette[0]
    }
}