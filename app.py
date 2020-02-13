# Import necessary packages
import pandas as pd
import numpy as np
import os
from PIL.Image import core as image
from PIL import Image
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import tensorflow as tf
import image_processing

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import base64

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#C7F8B0',
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
    
    html.Br(),
    
    html.Div(children='Pick an Image to Test the Model!', style={
        'textAlign': 'center',
        'color': '#073DD9',
        'fontSize': 30,
    }),
    
    html.Br(),
    
    html.Div([
        html.Label('Provide a Number Between 1 and 54,304:'),
        dcc.Input(id = 'image-number', placeholder = '1',
                  value = None,
                  type = 'text',
                  style = {'width': '10%', 'textAlign': 'center'}),
        html.Button('Calculate', id = 'button', style = {'color': '#E71107'}),
        html.Div([html.Div(id = 'image-number-text-confirmation'),
                  html.Br(),
                  html.Div(id = 'image-class', style = {'fontSize' : 24}),
                  html.Br(),
                  html.Img(id = 'image'),
                  html.Br(),
                  html.Br(),
                  html.Div(id = 'image-pred', style = {'fontSize' : 24}),
                  html.Br(),
                  html.Div(id = 'image-correct', style = {'fontSize' : 24})]),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),],
        style = {'justifyContent': 'center', 'align-items': 'center'})
])

@app.callback(
    [Output('image-number-text-confirmation', 'children'),
    Output('image-class', 'children'),
    Output('image', 'src'),
    Output('image-pred', 'children'),
    Output('image-correct', 'children')],
    [Input('button', 'n_clicks')],
    state = [State(component_id = 'image-number', component_property = 'value')])
def image_function(n_clicks, input_value):
    try:

        # Get Image Dictionary for Input
        img_int_list = [str(x) for x in list(range(1, 54305))] #List 1 to 54304
        data_path = 'PlantVillage-Dataset/raw_image_data/color'
        img_dict_list = [] # List of img_dicts
        for folder in os.listdir(data_path):
            for image in os.listdir('%s/%s' % (data_path, folder)):
                img_dict = {} # Dict with img_path and img_class
                img_path = '%s/%s/%s' % (data_path, folder, image)
                img_dict['path'] = img_path
                img_dict['class'] = folder
                img_dict_list.append(img_dict)
        img_int_dict = dict(zip(img_int_list, img_dict_list))
        
        # Create statement
        statement = 'You are viewing image number {}'.format(input_value)
        
        # Find image path based on user input number
        user_img_path = img_int_dict[input_value]['path']
        
        # Display image
        encoded_image = base64.b64encode(open(user_img_path, 'rb').read())
        image_src = 'data:image/png;base64,{}'.format(encoded_image.decode())
        
        # Display true image class
        user_img_class = img_int_dict[input_value]['class']
        user_img_text = 'This crop leaf is from class: {}'.format(user_img_class)
        
        # Create class_map_dict
        leaf_type_list = []
        for folder in os.listdir(data_path):
            leaf_type_list.append(folder)
        class_list = sorted(leaf_type_list)
        class_map_dict = dict(zip(list(range(38)), class_list))
        
        # Load CNN and Predict result
        model = load_model('../crop_leaves_disease_model.h5')
        pred_class = image_processing.image_prediction(user_img_path, model, class_map_dict)
        correct = (user_img_class == pred_class)
        pred_text = 'The Convolutional Neural Network Model predicts class: {}.'.format(pred_class)
        correct_text = 'This prediction is {}!'.format(correct)
        
        return statement, user_img_text, image_src, pred_text, correct_text
    
    # Error statement
    except:
        return "Error: Please input a number between 1 and 54,304", None, None, None, None



if __name__ == '__main__':
    app.run_server(debug=True)