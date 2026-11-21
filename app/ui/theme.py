"""Shared Streamlit and Plotly presentation helpers."""
import html
import streamlit as st

CHART_COLORS = ["#38BDF8", "#7DD3FC", "#34D399", "#FBBF24", "#FB7185", "#9FB2C8"]


def apply_theme():
    st.markdown("""<style>
    .block-container {padding-top:2rem;max-width:1100px;}
    [data-testid="stSidebar"] {background:#111C2E;border-right:1px solid #26364A;}
    button[kind="primary"] {background:#38BDF8;color:#0B1220;border-radius:8px;font-weight:600;border-color:#38BDF8;}
    button[kind="primary"]:hover {background:#7DD3FC;border-color:#7DD3FC;color:#0B1220;}
    [data-testid="stMetric"] {background:#17263A;border:1px solid #26364A;border-radius:12px;padding:12px 16px;}
    [data-testid="stFileUploaderDropzone"] {background:#111C2E;border:1px dashed #26364A;border-radius:12px;}
    [data-testid="stLayoutWrapper"]>[data-testid="stVerticalBlock"] {background:#17263A;border-color:#26364A !important;border-radius:12px !important;}
    [data-testid="stCaptionContainer"],.sidebar-status {color:#9FB2C8;opacity:1;}
    .section-label,.card-header-badge,.suggested-prompts-header {color:#7DD3FC;font-size:.75rem;letter-spacing:.06em;}
    .result-header-bar {display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;}
    .result-badge {color:#34D399;}
    .result-status-banner {padding:12px 16px;background:#17263A;border:1px solid #26364A;border-radius:12px;}
    .status-warning {color:#FBBF24;}.status-refusal {color:#FB7185;}
    button:focus-visible,a:focus-visible {outline:2px solid #7DD3FC;outline-offset:2px;}
    footer {display:none;}
    * {animation:none !important;transition:none !important;scroll-behavior:auto !important;}
    </style>""", unsafe_allow_html=True)


def card(title, body):
    st.markdown(
        '<div style="background:#17263A;border:1px solid #26364A;border-radius:12px;padding:12px 16px">'
        f'<h3>{html.escape(str(title))}</h3><p>{html.escape(str(body))}</p></div>',
        unsafe_allow_html=True,
    )


def style_figure(fig):
    """Change presentation only; preserve trace values and numeric color scales."""
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#17263A", plot_bgcolor="#111C2E",
        font=dict(color="#E5EEF8", size=12), colorway=CHART_COLORS, title=None,
        legend=dict(font=dict(color="#9FB2C8"), orientation="h", y=-.22, x=.5, xanchor="center"),
        hoverlabel=dict(bgcolor="#1E3047", bordercolor="#26364A", font=dict(color="#E5EEF8")),
        margin=dict(t=16, b=64, l=16, r=16), autosize=True, transition_duration=0,
    )
    for index, trace in enumerate(fig.data):
        color = CHART_COLORS[index % len(CHART_COLORS)]
        if trace.type == "pie":
            trace.marker.colors = CHART_COLORS
        elif hasattr(trace, "marker") and (trace.marker.color is None or isinstance(trace.marker.color, str)):
            trace.marker.color = color
        if hasattr(trace, "marker") and hasattr(trace.marker, "line"):
            trace.marker.line.color = "#26364A"
        if hasattr(trace, "line") and (trace.line.color is None or isinstance(trace.line.color, str)):
            trace.line.color = color
    for update_axis in (fig.update_xaxes, fig.update_yaxes):
        update_axis(gridcolor="#26364A", linecolor="#26364A", zeroline=False,
                    tickfont=dict(color="#9FB2C8"), title_font=dict(color="#E5EEF8"))
    return fig
