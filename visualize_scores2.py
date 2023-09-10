import plotly.graph_objects as go
import numpy as np

scores = np.load("grid_scores.npy", allow_pickle=True)

# Clipping scores to remove potential overfitting
#t = 3000
t = 10000
scores[:, 1][scores[:, 1] > t] = t
scores[:, 2][scores[:, 2] > t] = t
# scores[:, 1][scores[:, 1] < -t] = -t
# scores[:, 2][scores[:, 2] < -t] = -t

def weight(x,y):
    return (x + y) - abs(x - y) + 0.66 * y + 0.33 * x


# periods = 30, 60, 90, 120, 150, 180, 210, 240
# cooldown_types = ["buy", "sell", "both"]
# cooldowns = [0, 1, 2, 4, 8, 16, 32]
# # std_multipliers = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
# std_multipliers = [0,0.25,0.5,0.75,1.0]
# early_cashouts = ["False", "True"]

periods = 60,120
# cooldown_types = ["buy", "sell", "both"]
cooldown_types = ["buy"]
# cooldowns = [0,1,2,3,4]
cooldowns = [4,5,6]
# std_multipliers = [0.5,1,1.5,2,2.5,3,3.5,4]
# std_multipliers = [0,0.25,0.5,0.75,1.0]
std_multipliers = [0,0.25,0.5,0.75]
# early_cashouts = [False, True]
early_cashouts = [False]

divisions = cooldowns
division_categories = {d: [] for d in divisions}

for label, train, test in scores:
    hps = label.split(",")
    # hp = int(hps[0].split(" ")[-1])
    # hp = str(hps[1].split(" ")[-1])
    hp = int(hps[2].split(" ")[-1])
    # hp = float(hps[3].split(" ")[-1])
    # hp = str(hps[4].split(" ")[-1])
    division_categories[hp].append((train, test, f"{label}: Train: {train}, Test: {test}"))  # Adding label here to use in hovertext later

# Create a figure
fig = go.Figure()


# Display the plot


for k, v in division_categories.items():
    v = np.array(v)
    #w = v[:, 0].astype(float) + v[:, 1].astype(float)
    fig.add_trace(go.Scatter(x=v[:, 0].astype(float), y=v[:, 1].astype(float), text=v[:, 2], mode='markers', marker=dict(size=10), name=str(k), hoverinfo='text'))

# overall = {}
# for k, v in division_categories.items():
#     v = np.array(v)
#     w = weight(v[:, 0].astype(float), v[:, 1].astype(float))
#     overall[k] = np.sum(w)
#     #w = w1*v[:, 0].astype(float) + w2*v[:, 1].astype(float)
#
#     fig.add_trace(go.Scatter(x=np.arange(len(w)), y=w, text=v[:,2],mode='markers', marker=dict(size=10), name=str(k), hoverinfo='text'))

# print(overall)
# Add a line y=x
#fig.add_trace(go.Scatter(x=np.arange(0, t), y=np.arange(0, t), mode='lines', name='y=x'))

# Customize the layout of the plot
fig.update_layout(
    title='Scatter Plot with On Hover Labels',
    xaxis_title='Train Score',
    yaxis_title='Test Score',
    hovermode="closest"
)
# Set the tick mode to linear with a specific interval for x and y axes
# fig.update_xaxes(tickmode='linear', tick0=0, dtick=20)
# fig.update_yaxes(tickmode='linear', tick0=0, dtick=20)

fig.update_xaxes(tickmode='linear', tick0=0, dtick=200)
fig.update_yaxes(tickmode='linear', tick0=0, dtick=200)

# Display the plot
fig.show()
