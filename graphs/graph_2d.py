import numpy as np
import bokeh.plotting as bkp
from bokeh.models.tickers import FixedTicker
from bokeh.io import export_png

bkp.output_notebook(hide_banner=True)

def tweak_fig(fig):
    tight_layout(fig)
    disable_minor_ticks(fig)
    disable_grid(fig)
    fig.toolbar.logo = None

def tight_layout(fig):
    fig.min_border_top    = 35
    fig.min_border_bottom = 35
    fig.min_border_right  = 35
    fig.min_border_left   = 35

def disable_minor_ticks(fig):
    #fig.axis.major_label_text_font_size = value('8pt')
    fig.axis.minor_tick_line_color = None
    fig.axis.major_tick_in = 0

def disable_grid(fig):
    fig.xgrid.grid_line_color = None
    fig.ygrid.grid_line_color = None


def figure(*args, **kwargs):
    fig = bkp.figure(*args, **kwargs)
    tweak_fig(fig)
    return fig



def δ_2d(model, t, fig=None, show=True, filename=None):
    X = range(model.N)
    # X = np.arange(0, model.T, model.T / model.N)

    if fig is None:
        fig = figure(x_range=(min(X), max(X)), y_range=[-0.05, 1.0],
                     plot_width=400, plot_height=200, tools=())
        fig.yaxis.ticker = FixedTicker(ticks=[1])
        fig.yaxis.major_tick_out         = 0
        fig.outline_line_color           = None
        fig.xaxis.major_tick_line_color  = None
        fig.xaxis.major_label_text_color = None
        fig.xaxis.axis_line_color        = None
#        fig.xaxis.ticker = FixedTicker(ticks=list(X))
#        fig.xaxis.major_tick_out         = 2

    fig.line([min(X), max(X)], (0, 0), line_color='#888888', line_width=1)
    for tick in X:
        fig.line([tick, tick], (0, -0.01), line_color='#888888', line_width=1)
    fig.line(X, model.δ_history[t], line_color='black', line_width=5)

    if filename is not None:
        export_png(fig, filename='figures/{}.png'.format(filename))

    if show:
        bkp.show(fig)

def δ_2d_dashed(model, t, dashed_t=None, fig=None, show=True, legend=None, dashed_legend=None,
                filename=None):

    X = range(model.N)
    # X = np.arange(0, model.T, model.T / model.N)

    if fig is None:
        fig = figure(x_axis_label="Time-step", x_range=(min(X), max(X)),
                     y_axis_label="Prediction error", y_range=[-0.1, 1.0],
                     plot_width=400, plot_height=300, tools=())
        fig.yaxis.ticker = FixedTicker(ticks=[0, 1])
        fig.yaxis.major_tick_out = 2
        fig.xaxis.major_tick_line_color  = None
        fig.xaxis.major_label_text_color = None
        fig.xaxis.axis_line_color        = None
        fig.outline_line_color           = None


    fig.line([min(X), max(X)], (0, 0), line_color='#888888', line_width=1)
    for tick in X:
        fig.line([tick, tick], (0, -0.01), line_color='#888888', line_width=1)

    fig.line(X, model.δ_history[t-1], legend=legend, line_color='black', line_width=1)
    if dashed_t is not None:
        fig.line(X, model.δ_history[dashed_t-1], legend=dashed_legend, line_color='black',
                 line_dash='dotted', line_width=1)

    fig.legend.border_line_color = None
    fig.legend.location          = 'top_left'

    if filename is not None:
        export_png(fig, filename='figures/{}.png'.format(filename))

    if show:
        bkp.show(fig)

##p.circle(20*T/N, 1, size=4, line_color="red", fill_color="red")
##p.circle(5*T/N, 1, size=4, line_color="#56BA1B", fill_color="#56BA1B")
##p.circle(15*T/N, 1, size=4, line_color="#56BA1B", fill_color="#56BA1B")
