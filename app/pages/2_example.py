import streamlit as st
from squice.DataLoaders import NumpyNow
import plotly.graph_objs as go


st.set_page_config(
    page_title="squice",
    page_icon="app/static/squice.png",
    layout="wide",
)

data_str = st.text_area(
    "Matrix",
    """[[[0,0,0],[0,3,0],[0,0,0]],
[[0,0,0],[3,5,0],[0,0,0]],
[[0,0,0],[0,0,1],[0,0,0]]]
""",
    height=200,
).strip()


nn = NumpyNow(data_str)
nn.load()

with st.expander("Show mtx data"):
    st.write(nn.mtx)


xs = []
ys = []
zs = []
values = []
minv = None
maxv = None

a, b, c = nn.mtx.shape
for i in range(a):
    for j in range(b):
        for k in range(a):
            val = 0
            if k < c:
                val = nn.mtx[i][j][k]
                if minv is None:
                    minv = val
                    maxv = val
                else:
                    minv = min(minv, val)
                    maxv = max(maxv, val)
            xs.append(i)
            ys.append(j)
            zs.append(k)
            values.append(val)

c0 = "rgba(119,136,153,1)"
c1 = "rgba(240,248,255,0)"
c2 = "rgba(100,149,237,0.5)"
c3 = "rgba(220,20,60,0.9)"
c4 = "rgba(100,0,0,1)"


zero = 1 - (maxv / abs(maxv - minv))
zer0 = max(0.01, zero)
zer1 = max(zer0 * 1.5, 0.5)
zer2 = max(zer0 * 1.9, 0.8)

colorscale = [(0, c0), (zer0, c1), (zer1, c2), (zer2, c3), (1, c4)]

fig = go.Figure(
    data=go.Isosurface(
        x=xs,
        y=ys,
        z=zs,
        value=values,
        colorscale=colorscale,
        showscale=True,
        showlegend=False,
        opacity=0.6,
        surface_count=20,
        caps=dict(x_show=False, y_show=False),
        isomin=minv,
        isomax=maxv,
    )
)

fig.update_xaxes(showticklabels=False, visible=False)  # hide all the xticks
fig.update_yaxes(showticklabels=False, visible=False)  # hide all the xticks
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(scaleanchor="y", scaleratio=1)
# Plot!
st.plotly_chart(fig, use_container_width=True)
