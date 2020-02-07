# Import necessary packages
import pandas as pd
import numpy as np
import os
from PIL.Image import core as image
from PIL import Image

# Get Image Dictionary for Input
img_int_list = [str(x) for x in list(range(1, 54305))] #List 1 to 54304
data_path = 'PlantVillage-Dataset/raw_image_data/color'
img_dict = {} # Dict with img_path and img_class
img_dict_list = [] # List of img_dicts
for folder in os.listdir(data_path):
    for image in os.listdir('%s/%s' % (data_path, folder)):
        img_path = '%s/%s/%s' % (data_path, folder, image)
        img_dict['path'] = img_path
        img_dict['class'] = folder
        img_dict_list.append(img_dict)
img_int_dict = dict(zip(img_int_list, img_dict_list))

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#A7E5FE',
    'text': '#000000'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Crop Leaf Disease Image Classification Project',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='Analysis by Ben Geissel', style={
        'textAlign': 'center',
        'color': colors['text'],
        'fontSize': 20,
    }),
    
    html.Div(children='Pick an Image to Test the Model!', style={
        'textAlign': 'center',
        'color': '#073DD9',
        'fontSize': 30,
    }),
    
    html.Div([
        html.Label('Provide a Number Between 1 and 54304:'),
        dcc.Input(id = 'image-number', placeholder = '1',
                  value = None,
                  type = 'text',
                  style = {'width': '10%', 'textAlign': 'center'}),
        html.Div(id = 'image-number-text-confirmation')],
        style = {'justifyContent': 'center', 'align-items': 'center'})
])

@app.callback(
    Output('image-number-text-confirmation', 'children'),
    [Input('image-number', 'value')])
def update_output_div(input_value):
    return 'You\'ve selected image number "{}"'.format(input_value)




if __name__ == '__main__':
    app.run_server(debug=True)