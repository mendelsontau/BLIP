from plotly import graph_objects as go
import plotly.express as px

def set_fig_config(fig: go.Figure,
                   font_size=14,
                   width=500,
                   height=250,
                   margin_l=5,
                   margin_r=5,
                   margin_t=5,
                   margin_b=5,
                   font_family='Serif',
                   remove_background=False):
    if remove_background:
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(width=width, height=height,
                      font=dict(family=font_family, size=font_size),
                      margin_l=margin_l, margin_t=margin_t, margin_b=margin_b, margin_r=margin_r)
    return fig


def save_fig(fig, save_path):
    fig.write_image(save_path, width=2 * 200, height=0.5 * 200)


def plotly_init():
    figure_path = "scene_graph_amount.pdf"

    # 
    fig = go.Figure(data=[go.Scatter(x=[0.1,0.4,0.7,1.0], y=[17.75,19.75,20.75,21.50],
                                    marker={'color': 'rgba(0, 0, 255, 0.4)', 'size': 8},
                                    line={'dash': 'dot', 'color': 'rgba(0, 0, 255, 0.1)'})])
    fig.update_layout(showlegend=False,
                      plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(family="Arial", size=16),
                      xaxis=dict( tickmode='linear', dtick=0.25),
                      yaxis=dict( tickmode='linear', dtick=0.5),
                      margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
                      height=120,
                      width=360
                      )
                      
    fig.update_xaxes(title_text="SG Data Amount",
                     title_standoff=10,
                     title_font={"size": 9},
                     tickfont_size=8,
                     showline=False,
                     linecolor='black',
                     gridcolor='rgb(204,204,204)')
    
    fig.update_yaxes(title_text="Group Score",
                     title_standoff=15,
                     title_font={"size": 9},
                     tickfont_size=8,
                     showline=False,
                     gridcolor='rgb(204,204,204)')


    #garbage graph
    fig1 = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig1.show()
    fig1.write_image("random.pdf")
    

    save_fig(fig, figure_path)
    fig.write_image(figure_path)
    
    fig.show()
    
plotly_init()