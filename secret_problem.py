import numpy as np
import pandas as pd

from bokeh.models import (
    ColumnDataSource,
    CustomJS,
    HoverTool,
    Legend,
    LegendItem,
    LinearColorMapper,
    Slider
)
from bokeh.plotting import figure
from bokeh.layouts import column, row


# the secret parameters that only nature knows
a = 2
b = 4
c = 1
sigma = 4

# the best line parameters that only nature knows
m_star = b
k_star = c + a

# some arbitrary number of observations
N = 250


def true_function(x):
    """
    The oracle!
    """
    return a*x**2 + b*x + c

# generate some observations deterministically
rng = np.random.default_rng(10151988)
x = rng.normal(size=(N,))
eps = rng.normal(size=(N,))
y = true_function(x) + sigma*eps

# combine into a dataframe to take advantage of
# existing plotting libs
data = pd.DataFrame({
    "x": x,
     "input": x,
     "y": y,
    "output": y
})


def _compute_cost_surface(sample_size=None):
    grid_dim = 50
    m = np.linspace(m_star - 5, m_star + 5, grid_dim)
    k = np.linspace(k_star - 5, k_star + 5, grid_dim)
    kk, mm = np.meshgrid(k, m)

    if sample_size is not None:
        img = np.zeros_like(mm)
        for i in range(sample_size):
            img += (mm*x[i] + kk - y[i])**2
        img /= sample_size
    else:
        img = 2*a**2 + (mm - b)**2 + (kk - a - c)**2 + sigma**2

    mapper = LinearColorMapper(
        low=np.log(img).min(),
        high=np.log(img).max(),
        palette="Plasma256"
    )

    source = ColumnDataSource({
        "slope": [mm],
        "intercept": [kk],
        "x": [k_star - 5],
        "y": [m_star - 5],
        "dw": [10],
        "dh": [10],
        "img": [img],
        "log": [np.log(img)]
    })
    return source, mapper


def _init_surface_plot(sample_size, fig_kwargs):
    if sample_size is not None:
        title = "Log empirical error"
    else:
        title = "Log expected error"

    default_fig_kwargs = {
        "height": 400,
        "width": 600
    }
    if fig_kwargs is not None:
        default_fig_kwargs.update(fig_kwargs)
    p = figure(
        title=title,
        x_axis_label="intercept",
        y_axis_label="slope",
        x_range=(k_star - 5, k_star + 5),
        y_range=(m_star - 5, m_star + 5),
        tools="",
        **default_fig_kwargs
    )
    p.grid.grid_line_alpha = 0.0
    return p
    
def plot_cost_surface(sample_size=None, fig_kwargs=None):
    """
    using a, b, c as parameters of quadratic
    and m, k as slope, intercept of line
    """
    p = _init_surface_plot(sample_size, fig_kwargs)
    source, mapper = _compute_cost_surface(sample_size)
    p.image(
        "log",
        x="x",
        y="y",
        dw="dw",
        dh="dh",
        color_mapper=mapper,
        source=source,
        name="img",
        level="image"
    )

    items = []
    r = p.plus(
        [k_star],
        [m_star],
        line_color="#000000",
        line_width=0.8,
        fill_color="#ffffff",
        size=10
    )
    items.append(
        LegendItem(renderers=[r], label="True best parameters")
    )
    if sample_size is not None:
        _x = x[:sample_size]
        _y = y[:sample_size]
        m_hat = (
            (_y.sum()*_x.sum() / sample_size - (_y*_x).sum()) /
            ((_x.sum())**2 / sample_size - (_x**2).sum())
        )
        k_hat = _y.sum() / sample_size - m_hat * _x.sum() / sample_size
        r = p.plus(
            [k_hat],
            [m_hat],
            line_color="#000000",
            line_width=1.5,
            fill_color="#000000",
            size=10
        )
        items.append(
            LegendItem(renderers=[r], label="Best fit on sample")
        )

    p.add_layout(Legend(items=items), "right")
    p.add_tools(HoverTool(
        renderers=p.select("img"),
        tooltips=[
            ("Slope", "$y"),
            ("Intercept", "$x"),
            ("Expected Error", "@img")
        ]
    ))
    return p


def plot_surfaces_side_by_side(fig_kwargs=None):
    sample_size = 10
    _x = x[:sample_size]
    _y = y[:sample_size]
    m_hat = (
        (_y.sum()*_x.sum() / sample_size - (_y*_x).sum()) /
        ((_x.sum())**2 / sample_size - (_x**2).sum())
    )
    k_hat = _y.sum() / sample_size - m_hat * _x.sum() / sample_size
    fit_source = ColumnDataSource({
        "slope": [m_hat],
        "intercept": [k_hat]
    })

    figures = []
    fig_kwargs = fig_kwargs or {}
    for sample_size in [None,  sample_size]:
        width = fig_kwargs.get("width", 400)
        if sample_size is not None:
            width += 200
        fig_kwargs["width"] = width

        p = _init_surface_plot(sample_size, fig_kwargs)
        source, mapper = _compute_cost_surface(sample_size)
        if sample_size is None:
            global_mapper = mapper
        p.image(
            "log",
            x="x",
            y="y",
            dw="dw",
            dh="dh",
            color_mapper=global_mapper,
            source=source,
            name="img",
            level="image"
        )
        r = p.plus(
            [k_star],
            [m_star],
            line_color="#000000",
            line_width=0.8,
            fill_color="#ffffff",
            size=10
        )
        items = [LegendItem(renderers=[r], label="True best parameters")]

        r = p.plus(
            "intercept",
            "slope",
            line_color="#000000",
            line_width=1.5,
            fill_color="#000000",
            size=10,
            source=fit_source
        )
        items.append(LegendItem(renderers=[r], label="Best fit on sample"))

        p.add_tools(HoverTool(
            renderers=p.select("img"),
            tooltips=[
                ("Slope", "$y"),
                ("Intercept", "$x"),
                ("Expected Error", "@img")
            ]
        ))
        figures.append(p)

    p_true, p_emp = figures
    p_emp.add_layout(Legend(items=items), "right")

    js_code = """
        var img_data = img_src.data; // data from the source containing the image
        var sample_data = sample_src.data; // data from the source containing our samples
        var n = cb_obj.value; // current value of the slider
        var img_dim = {}; // format the string with the grid dimension

        // initialize all the variables we'll need
        var pixel;
        var slope;
        var intercept;
        var error;
        var idx;

        // loop through each slope/intercept combination, calculate
        // the average error over the current number of samples, then
        // update that pixel in the image source's data
        for (var i = 0; i < img_dim; i++) {{
            for (var j = 0; j < img_dim; j++) {{
                pixel = 0;
                idx = i*img_dim + j;
                for (var k = 0; k < n; k++) {{
                    slope = img_data["slope"][0][idx]
                    intercept = img_data["intercept"][0][idx]
                    error = slope*sample_data["x"][k] + intercept - sample_data["y"][k];
                    pixel += Math.pow(error, 2);
                }}
                pixel /= n;
                img_data["log"][0][idx] = Math.log(pixel);
                img_data["img"][0][idx] = pixel;
            }}
        }}

        // have the source update all its renderers to reflect the new data
        img_src.change.emit();

        // now do our regression fit in JS land to
        // move the best fit cross around the screen
        var fit_data = fit_src.data;

        // push all n samples to our subsample source
        // compute sums we'll need to do regression fit
        var x;
        var y;
        var xsum = 0;
        var ysum = 0;
        var x2sum = 0;
        var xysum = 0;
        for (var i = 0; i < n; i ++) {{
            x = sample_data["x"][i];
            y = sample_data["y"][i];

            xsum += x;
            ysum += y;
            x2sum += Math.pow(x, 2);
            xysum += x*y;
        }}

        // update best fit values for regression
        var denom = n*x2sum - Math.pow(xsum, 2);
        var m = (n*xysum - xsum*ysum) / denom;
        var b = (ysum*x2sum - xsum*xysum) / denom;
        fit_data["intercept"] = [b];
        fit_data["slope"] = [m];

        // update all source renderers
        fit_src.change.emit();
    """.format(source.data["img"][0].shape[0])

    slider = Slider(
        start=1,
        end=N,
        step=1,
        value=sample_size,
        title="Observations Used for Fit",
        orientation="horizontal",
        direction="ltr"
    )
    callback = CustomJS(
        args={
            "img_src": source,
            "sample_src": ColumnDataSource(data),
            "fit_src": fit_source
        },
        code=js_code
    )
    slider.js_on_change("value", callback)

    return column(row(*figures), slider)
