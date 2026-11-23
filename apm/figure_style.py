"""Physical layout units and print colours for the read-only figure exporters."""
from textwrap import wrap

from matplotlib import rc_context
from matplotlib.patches import Rectangle

INK, TEXT, MUTED = "#1F3650", "#253746", "#52677A"
PANEL, LINE, MISSING = "#F7F9FB", "#DCE3E8", "#A5AFB8"
ACTOR, LEARNED, SEARCH = "#355C7D", "#27866D", "#B27433"
ALIGN, RECOVERY = "#84639E", "#AC605B"
SELECTED = "#E9F4EF"


def surface(fig, x, top, width, height):
    """Position an axes in inches from the top left. Axes coordinates run down the page."""
    w, h = fig.get_size_inches()
    ax = fig.add_axes((x/w, 1-(top+height)/h, width/w, height/h))
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.set_axis_off()
    return ax


def box(ax, x, y, w, h, color, edge="none", lw=.6):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor=edge, linewidth=lw))


def txt(ax, x, y, value, size=8, color=TEXT, bold=False, **kw):
    return ax.text(x, y, value, fontsize=size, color=color, weight="bold" if bold else "normal",
                   va="center", **kw)


def heading(fig, title, detail):
    w, h = fig.get_size_inches()
    fig.text(.13/w, 1-.13/h, title, ha="left", va="top", fontsize=12, weight="bold", color=INK)
    fig.text(.13/w, 1-.37/h, detail, ha="left", va="top", fontsize=8, color=MUTED)


def lines(value, width):
    return wrap(str(value), width=width, break_long_words=False, break_on_hyphens=False) or [""]


def save(fig, path, dpi, tight=True):
    # Preserve editable vector text on export without changing process-wide settings afterward.
    with rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig.savefig(path, dpi=dpi, bbox_inches="tight" if tight else None,
                    pad_inches=.065, facecolor="white")
