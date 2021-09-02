from flask import Flask, render_template, request
import torch
from torch import nn
from pickle import load
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/checkMushroom", methods=["POST"])
def checkMushroom():
    x = list()
    if request.method == "POST":
        x.append(request.form["cap-shape"])
        x.append(request.form["cap-surface"])
        x.append(request.form["cap-color"])
        x.append(request.form["bruises"])
        x.append(request.form["odor"])
        x.append(request.form["gill-attachment"])
        x.append(request.form["gill-spacing"])
        x.append(request.form["gill-size"])
        x.append(request.form["gill-color"])
        x.append(request.form["stalk-shape"])
        x.append(request.form["stalk-surface-above-ring"])
        x.append(request.form["stalk-surface-below-ring"])
        x.append(request.form["stalk-color-above-ring"])
        x.append(request.form["stalk-color-below-ring"])
        x.append(request.form["veil-color"])
        x.append(request.form["ring-number"])
        x.append(request.form["ring-type"])
        x.append(request.form["spore-print-color"])
        x.append(request.form["population"])
        x.append(request.form["habitat"])

        model = nn.Sequential(nn.Linear(20,30),
                      nn.ReLU(),
                      nn.Linear(30, 20),
                      nn.ReLU(),
                      nn.Linear(20, 10),
                      nn.ReLU(),
                      nn.Linear(10, 1),
                      nn.Sigmoid())

        model.load_state_dict(torch.load("./model.pth"))
        encoder = load(open('./encoder.pkl', 'rb'))
        x = encoder.transform([x])
        mushroom_type = None
        with torch.no_grad():
            mushroom_type = torch.round(model(torch.tensor(x).type(torch.float32))).item()
        
        if mushroom_type == 0.0:
            return render_template("edible_mushroom.html")
        else:
            return render_template("poisonous_mushroom.html")

@app.route("/goToIndex", methods=["POST"])
def goToIndex():
    return render_template("index.html")

if __name__ == "__main__":
    app.run()
