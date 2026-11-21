"""Shared Streamlit and Plotly presentation helpers."""
import html
import streamlit as st

COLORS = {
    "background": "#0A1020", "surface": "#101A2E", "card": "#16233A",
    "elevated": "#1D2D49", "border": "#2A3A55", "text": "#E8EEF8",
    "secondary": "#9FB2C8", "sky": "#38BDF8", "sky_hover": "#7DD3FC",
    "violet": "#8B5CF6", "violet_light": "#A78BFA", "teal": "#2DD4BF",
    "amber": "#FBBF24", "rose": "#FB7185",
}
CHART_COLORS = [COLORS[key] for key in ("sky", "violet", "teal", "amber", "rose", "sky_hover")]


def apply_theme():
    st.markdown("""<style>
    :root {--bg:#0A1020;--surface:#101A2E;--card:#16233A;--elevated:#1D2D49;
      --border:#2A3A55;--text:#E8EEF8;--muted:#9FB2C8;--sky:#38BDF8;
      --sky-hover:#7DD3FC;--violet:#8B5CF6;--violet-light:#A78BFA;
      --teal:#2DD4BF;--amber:#FBBF24;--rose:#FB7185;
      --accent:linear-gradient(90deg,#38BDF8,#8B5CF6);}
    [data-testid="stAppViewContainer"] {background:radial-gradient(circle at 50% 0%,rgba(56,189,248,.08),transparent 34%),var(--bg);color:var(--text);}
    .block-container {max-width:1300px;padding-top:1.5rem;padding-bottom:3rem;}
    [data-testid="stSidebar"] {background:var(--surface);border-right:1px solid var(--border);}
    [data-testid="stSidebar"] a {color:var(--sky);text-decoration:none;border-radius:8px;padding:.2rem .35rem;}
    [data-testid="stSidebar"] a:hover {color:var(--violet-light);background:var(--elevated);}
    body:has(#your-data:target) [data-testid="stSidebar"] a[href="#your-data"],
    body:has(#analysis:target) [data-testid="stSidebar"] a[href="#analysis"],
    body:has(#charts:target) [data-testid="stSidebar"] a[href="#charts"] {border-left:3px solid transparent;border-image:var(--accent) 1;background:var(--elevated);}
    h1 {font-size:28px !important;font-weight:600 !important;letter-spacing:-.02em;}
    [data-testid="stHeadingWithActionElements"]:has(h1) {padding-bottom:.6rem;border-bottom:3px solid transparent;border-image:var(--accent) 1;margin-bottom:.65rem;}
    [data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"],
    [data-testid="stExpander"] {background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px;transition:border-color 150ms ease;}
    [data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"]:hover,
    [data-testid="stExpander"]:hover {border-color:rgba(56,189,248,.5);}
    [data-testid="stMetric"] {background:var(--card);border:1px solid var(--border);border-top:3px solid var(--sky);border-radius:12px;padding:12px 16px;}
    [data-testid="stMetricValue"] {font-size:26px;font-weight:600;}
    [data-testid="stMetricLabel"],[data-testid="stCaptionContainer"],.sidebar-status {color:var(--muted);opacity:1;}
    [data-testid="stFileUploaderDropzone"] {background:var(--surface);border:1px dashed var(--border);border-radius:12px;transition:border-color 150ms ease,background 150ms ease;}
    [data-testid="stFileUploaderDropzone"]:hover {border-color:var(--sky);background:var(--elevated);}
    button[kind="primary"] {background:var(--accent);color:var(--bg);border:0;border-radius:8px;font-weight:600;}
    button[kind="primary"]:hover {filter:brightness(1.08);color:var(--bg);}
    button[kind="primary"]:active {filter:brightness(.92);}
    button[kind="secondary"],[data-testid="stFileUploaderDropzone"] button {background:var(--card);color:var(--text);border:1px solid var(--border);border-radius:8px;}
    button[kind="secondary"]:hover,[data-testid="stFileUploaderDropzone"] button:hover {border-color:var(--sky);color:var(--sky-hover);}
    button:disabled {opacity:.52;cursor:not-allowed;}
    button:focus-visible,a:focus-visible {outline:2px solid var(--sky);outline-offset:2px;}
    [data-baseweb="input"] > div,[data-baseweb="select"] > div,textarea {background:var(--surface) !important;border-color:var(--border) !important;color:var(--text) !important;}
    [data-baseweb="input"] > div:hover,[data-baseweb="select"] > div:hover,textarea:hover {border-color:var(--sky) !important;}
    [data-baseweb="input"]:focus-within,[data-baseweb="select"]:focus-within,textarea:focus {border-color:var(--sky) !important;box-shadow:0 0 0 3px rgba(56,189,248,.25) !important;}
    [data-baseweb="popover"],[role="listbox"],[role="tooltip"] {background:var(--surface);color:var(--text);border:1px solid var(--border);}
    [role="option"]:hover {background:var(--elevated);}
    [data-testid="stDataFrame"] {border:1px solid var(--border);border-radius:8px;overflow:hidden;}
    [data-testid="stTabs"] button[aria-selected="true"] {color:var(--sky);border-color:var(--sky);}
    [data-testid="stTabs"] button:hover,[data-testid="stExpander"] summary:hover {color:var(--violet-light);}
    [data-testid="stAlert"] {background:var(--card);color:var(--text);border-left:3px solid var(--sky);}
    [data-testid="stAlert"]:has([data-testid="stNotificationContentWarning"]) {border-left-color:var(--amber);}
    [data-testid="stAlert"]:has([data-testid="stNotificationContentError"]) {border-left-color:var(--rose);}
    [data-testid="stAlert"]:has([data-testid="stNotificationContentSuccess"]) {border-left-color:var(--teal);}
    .section-label,.card-header-badge,.suggested-prompts-header {color:var(--muted);font-size:.78rem;font-weight:500;letter-spacing:.02em;text-transform:none;}
    .result-header-bar {display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;}
    .result-badge {color:var(--teal);}.result-status-banner {padding:12px 16px;background:var(--card);border:1px solid var(--border);border-radius:12px;}
    .validated-tag {display:inline-block;color:var(--teal);border:1px solid var(--teal);border-radius:6px;padding:.15rem .5rem;font-size:.8rem;font-weight:600;}
    [data-testid="stMarkdownContainer"] li::marker {color:var(--sky);}
    [data-testid="stMarkdownContainer"] li:nth-child(3n+2)::marker {color:var(--violet-light);}
    [data-testid="stMarkdownContainer"] li:nth-child(3n+3)::marker {color:var(--teal);}
    .status-warning {color:var(--amber);}.status-refusal {color:var(--rose);}
    .st-key-kpi_row [data-testid="stMetric"] {min-height:104px;}
    .st-key-kpi_row [data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"] {border-top-color:var(--violet);}
    .st-key-kpi_row [data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"] {border-top-color:var(--teal);}
    .st-key-kpi_row [data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"] {border-top-color:var(--amber);}
    *::-webkit-scrollbar {width:8px;height:8px;} *::-webkit-scrollbar-thumb {background:var(--border);border-radius:8px;}
    *::-webkit-scrollbar-thumb:hover {background:var(--sky);} *::-webkit-scrollbar-track {background:transparent;}
    footer {display:none;} button,a,[data-testid="stExpander"] {transition:color 150ms ease,border-color 150ms ease,background 150ms ease;}
    @media(max-width:760px) {.block-container {padding:1rem .75rem 2rem;}h1 {font-size:25px !important;}}
    @media(prefers-reduced-motion:reduce) {* {transition:none !important;animation:none !important;}}
    </style>""", unsafe_allow_html=True)


def card(title, body):
    st.markdown(
        '<div style="background:#16233A;border:1px solid #2A3A55;border-radius:12px;padding:16px">'
        f'<h3>{html.escape(str(title))}</h3><p>{html.escape(str(body))}</p></div>',
        unsafe_allow_html=True,
    )


def style_figure(fig):
    """Change presentation only; preserve trace values and numeric color scales."""
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"], size=12), colorway=CHART_COLORS, title=None,
        legend=dict(font=dict(color=COLORS["secondary"]), orientation="h", y=-.22, x=.5, xanchor="center"),
        hoverlabel=dict(bgcolor=COLORS["elevated"], bordercolor=COLORS["border"], font=dict(color=COLORS["text"])),
        margin=dict(t=16, b=64, l=16, r=16), autosize=True, transition_duration=0,
    )
    for index, trace in enumerate(fig.data):
        color = CHART_COLORS[index % len(CHART_COLORS)]
        if trace.type == "pie":
            trace.marker.colors = CHART_COLORS
        elif hasattr(trace, "marker") and (trace.marker.color is None or isinstance(trace.marker.color, str)):
            trace.marker.color = color
        if hasattr(trace, "marker") and hasattr(trace.marker, "line"):
            trace.marker.line.color = COLORS["border"]
        if hasattr(trace, "line") and (trace.line.color is None or isinstance(trace.line.color, str)):
            trace.line.color = color
    for update_axis in (fig.update_xaxes, fig.update_yaxes):
        update_axis(gridcolor=COLORS["border"], linecolor=COLORS["border"], zeroline=False,
                    tickfont=dict(color=COLORS["secondary"]), title_font=dict(color=COLORS["text"]))
    return fig
