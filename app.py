import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import dill
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import os
import numpy as np
import gc


# load recommendation table by ALS
df_als = dill.load(open('data/userRecs_pd.pkd', 'rb'))

# load recommendation table by content-based filtering
df_nmf_prod = dill.load(open('data/df_nmf.pkd', 'rb'))
ingred_nn = dill.load(open('data/nn_ingred.pkd', 'rb'))
cbf_matrix = dill.load(open('data/cbf_matrix.pkd', 'rb'))

# load product and rating tables
prod_df = dill.load(open('data/prod_df.pkd', 'rb'))
# prod_ratings = dill.load(open('data/prod_ratings.pkd', 'rb'))
prod_ratings_no_rev = dill.load(open('data/prod_ratings_no_rev.pkd', 'rb'))

# load nearest neighbor model
nn = dill.load(open('data/nn_users.pkd', 'rb'))

# load table of users to look up list of similar users from nearest neighbor result
df_features = dill.load(open('data/df_features.pkd', 'rb'))



# get a list of characteristics for choices
age = ['18 & Under', '19-24', '25-29', '30-35', '36-43', '44-55', '56 & Over', 'Unknown']
skin_type = ['Acne-prone', 'Combination', 'Dry', 'Normal', 'Oily', 'Other', 'Sensitive', 'Very Dry', 'Very Oily']
skin_tone = ['Dark', 'Deep Dark', 'Fair', 'Fair-Medium', 'Medium', 'Medium Brown', 'Olive', 'Other', 'Tan']
skin_undertone = ['Cool', 'Neutral', 'Not Sure', 'Warm']

hair_color = ['Black', 'Blond', 'Brown', 'Brunette', 'Grey', 'Other', 'Red', 'Silver']
hair_type = ['Curly', 'Kinky', 'Other', 'Relaxed', 'Straight', 'Wavy']
hair_texture = ['Coarse', 'Fine', 'Medium', 'Other']
eyes = ['Black', 'Blue', 'Brown', 'Green', 'Grey', 'Hazel', 'Other', 'Violet']

######################################################
# stylesheet for the app
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# if there's a css style sheet in the /assets folder, it will be used even if we specify external_stylesheet here
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
# to use css style sheet in the /assets folder (which will be automatically served)
# app = dash.Dash(__name__)


# must have for dash to deploy
server = app.server


# navigation bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavLink("About", href="#about", external_link=True),
        dbc.NavLink("Start", href="#start", external_link=True),
        dbc.NavLink("Recommend", href="#rec", external_link=True),
        dbc.NavLink("Contact", href="#contact", external_link=True),
    ],
    brand="Home",
    brand_href="#",
    sticky="top",
)

# Landing page and home page, has logo
landing_pg = dbc.Container(
    [
        dbc.Row(
            [html.Div([
                html.Img(src=app.get_asset_url('hg_logo.png'))])
            ],
            # justify='center',
            # className='align-items-center'
        ),

    ],
    # mt-4 is margin top 4px
    # vh-100 is view height 100% of window

    className="vh-100 d-flex align-items-center justify-content-center",
)

# About page, brief introduction about the app
about_pg = dbc.Container(id='about', children=
    [
        dbc.Row(
            dbc.Card(
                [
                    dbc.CardHeader("Holy Grail"),
                    dbc.CardBody(
                        [
                            dbc.CardTitle("Your skincare recommender app"),
                            dbc.CardText("Input your favorite product and let Holy Grail recommend other similar products."
                                         "You can use the dropdown list to see the recommended products and choose one of them to explore ratings for the products."),
                        ]),
                ]),
        )],
    className="vh-100 d-flex align-items-center justify-content-center",
    )

# start page where users specify their characteristics
start_pg = dbc.Container(id='start', children=
        [
            dbc.Row([
                dbc.Col(
                        dbc.FormGroup([dbc.Label('Your age range'),
                       dbc.RadioItems(id='age',
                       options=[
                           dict(label=age[i], value=age[i]) for i in range(len(age))
                       ],
                       )])
                ),
                dbc.Col(
                        dbc.FormGroup([dbc.Label('Your skin type'),
                       dbc.RadioItems(id='skin_type',
                       options=[
                           dict(label=skin_type[i], value=skin_type[i]) for i in range(len(skin_type))
                       ],
                       )])
                ),
                dbc.Col(
                        dbc.FormGroup([dbc.Label('Your skin tone'),
                       dbc.RadioItems(id='skin_tone',
                       options=[
                           dict(label=skin_tone[i], value=skin_tone[i]) for i in range(len(skin_tone))
                       ],
                       )])
                ),
                dbc.Col(
                        dbc.FormGroup([dbc.Label('Your skin undertone'),
                       dbc.RadioItems(id='skin_undertone',
                       options=[
                           dict(label=skin_undertone[i], value=skin_undertone[i]) for i in range(len(skin_undertone))
                       ],
                       )])
                ),
            ], className='mt-5 pt-5'),
# , className="vh-100", align='center', justify='center'
            dbc.Row([
                dbc.Col(
                        dbc.FormGroup([dbc.Label('Your hair color'),
                       dbc.RadioItems(id='hair_color',
                       options=[
                           dict(label=hair_color[i], value=hair_color[i]) for i in range(len(hair_color))
                       ],
                       )])
                ),
                dbc.Col(
                        dbc.FormGroup([dbc.Label('Your hair type'),
                       dbc.RadioItems(id='hair_type',
                       options=[
                           dict(label=hair_type[i], value=hair_type[i]) for i in range(len(hair_type))
                       ],
                       )])
                ),
                dbc.Col(
                        dbc.FormGroup([dbc.Label('Your hair texture'),
                       dbc.RadioItems(id='hair_texture',
                       options=[
                           dict(label=hair_texture[i], value=hair_texture[i]) for i in range(len(hair_texture))
                       ],
                       )])
                ),
                dbc.Col(
                        dbc.FormGroup([dbc.Label('Your eye color'),
                       dbc.RadioItems(id='eyes',
                       options=[
                           dict(label=eyes[i], value=eyes[i]) for i in range(len(eyes))
                       ],
                       )])
                ),
            ], className='mt-5 pt-5'),
        ], className="vh-100"
    )

# Recommendation page
rec_pg = dbc.Container(id='rec', children=
    [
        dbc.Row([
            dbc.Col([
                dbc.Label("Suggest products similar to", className='mt-5 pt-5'),
                dcc.Dropdown(
                        id='input_prod_state'
                        # names of products are stored in index of table df_nmf_prod
                        # using dropdown menu will automatically give the ability of searching for product name
                        , options = [
                        dict(label=prod_df.names.iloc[i], value=prod_df.names.iloc[i]) for i in range(len(prod_df))
                        ]
                        , value='eg. Retin A'
                        ),
                # dbc.Input(id="input_test", placeholder="Type something...", type="text"),
                dbc.Button("Submit", id='submit_button', n_clicks=0, color="secondary", className="mr-1"),
            ], width=8)
        ], justify='center'
            # , className="vh-100", align='center', justify='center'
        ),

        # Recommended products are displayed here
        dbc.Row([
                dbc.Label('Recommended products - Click on product to see ratings'),
                dbc.RadioItems(id='rec_outin')
        ], justify='center', className='mt-5 pt-5', id='prod_rec_toggle'),


        # Graph of ratings is displayed here
        dbc.Row(id='prod_graph_toggle', className='mt-5 pt-5', children=[
            dbc.Col([
                # dbc.Label('Recommended products', className='mt-5 pt-5'),
                dcc.Graph(id='prod_lipies_dist')

            ])

        ]),
    ])



# Contact page
contact_pg = dbc.Container(id='contact', children=
    [
        dbc.Row(
            dbc.Card(
                [
                    dbc.CardHeader("Contact"),
                    dbc.CardBody(
                        [
                            dbc.CardTitle("Natalie Le"),
                            dbc.CardLink("LinkedIn", href="https://www.linkedin.com/in/nataliele"),
                            dbc.CardLink("Read about the process", href="https://nataliele.github.io/2019-01-07-capstone/"),
                        ]),
                ]),
        )],
    className="vh-100 d-flex align-items-center justify-content-center",
    )

# layout of the app, has to be assigned before callback
app.layout = html.Div([navbar, landing_pg, about_pg, start_pg, rec_pg, contact_pg])


# callback controls the interactivity of the app
@app.callback(
    # we are passing the result of function update_output_div into the property
    # 'options' of the component with id 'rec_outin'
    # because property 'options' of the component 'Dropdown' takes a list, that
    # is what update_output_div function returns
    # the input_value for the update_output_div function is from the callback's
    # input, which in this case returns n_clicks and the state's value, which
    # is taken from the input box
    Output(component_id='rec_outin', component_property='options'),
    [Input('submit_button', 'n_clicks'),
    Input('age', 'value'),
    Input('skin_type', 'value'),
    Input('skin_tone', 'value'),
    Input('skin_undertone', 'value'),
    Input('hair_color', 'value'),
    Input('hair_type', 'value'),
    Input('hair_texture', 'value'),
    Input('eyes', 'value'),
    ],
    [State('input_prod_state', 'value')]
    )
# have to put n_clicks in even though dont use it because thats what callback's
# input returns?
# the additional inputs are returned before inputs from states
def update_output_dropdown(n_clicks, input_age, input_skin_type, input_skin_tone, input_skin_undertone,
                           input_hair_color, input_hair_type, input_hair_texture, input_eyes, input_value):

    # function to handle when product name is not found, return nothing
    try:
        ### find products from similar users
        # create array for new user's characteristics
        user = np.zeros(56)

        # populate user array by comparing input and existing list
        for i, age_range in enumerate(age):
            if input_age == age_range:
                user[i] = 1

        for i, type in enumerate(skin_type):
            if input_skin_type == type:
                user[i+16] = 1

        for i, tone in enumerate(skin_tone):
            if input_skin_tone == tone:
                user[i+25] = 1

        for i, undertone in enumerate(skin_undertone):
            if input_skin_undertone == undertone:
                user[i+34] = 1

        for i, h_color in enumerate(hair_color):
            if input_hair_color == h_color:
                user[i+38] = 1

        for i, h_type in enumerate(hair_type):
            if input_hair_type == h_type:
                user[i+46] = 1

        for i, h_texture in enumerate(hair_texture):
            if input_hair_texture == h_texture:
                user[i+52] = 1

        for i, eye_color in enumerate(eyes):
            if input_eyes == eye_color:
                user[i+8] = 1

        # find indices of similar users to new user
        dists, indices = nn.kneighbors(user.reshape(1, -1))

        # list of users similar to our test user
        user_list = df_features.iloc[indices[0]]

        # convert to df to merge
        user_list_df = pd.DataFrame(user_list)

        # list of recommended product ids
        rec = pd.merge(user_list_df, df_als, left_on=user_list_df.uid, right_on=df_als.uid, how='left')[['uid_x', 'rec1']]

        # load descriptions of recommended products
        prod_rec = pd.merge(rec, prod_df, left_on=rec.rec1, right_on=prod_df.pro_ids, how='left')

        prod_rec_dedup = prod_rec.drop(['key_0', 'uid_x', 'rec1'], axis=1).drop_duplicates()


        ### find products similar to user's product - content-based filtering
        product = cbf_matrix.loc[input_value]

        # find 10 products similar to the test product
        # indices are indices of the 10 most similar products, not the actual product id
        ingred_dists, ingred_indices = ingred_nn.kneighbors(product.values.reshape(1, -1))

        # list of 10 products similar to test product, excluding the first one, which is the test product
        top_10 = prod_df.iloc[ingred_indices[0]][1:]
        top10_rec = top_10

        ### find products similar to user's product - cosine similarity
        # product = df_nmf_prod.loc[input_value]
        # # Compute cosine similarities: similarities
        # similarities = df_nmf_prod.dot(product)
        # top_10 = similarities.nlargest(11)[1:]

        ### combine products from similar products and products from similar users
        # top10_df = pd.DataFrame(top_10)
        # top10_rec = pd.merge(top10_df, prod_df, left_on=top10_df.index, right_on=prod_df.names, how='left')

        tot_rec = prod_rec_dedup[['names', 'ratings', 'repurchases', 'pkg_quals', 'prices',
                                  'ingredients', 'brands']].append(
            top10_rec[['names', 'ratings', 'repurchases', 'pkg_quals', 'prices',
                       'ingredients', 'brands']])

        # have to reset index so slicing can work below
        final_rec = tot_rec.sample(10, random_state=123).reset_index()

        # options takes a list of dictionary
        # has to have 'value =' in order to choose value
        return [
            # dict(label = top_10.index[i], value = top_10.index[i]) for i in range(len(top_10))
            dict(label=final_rec.names[i], value=final_rec.names[i]) for i in range(len(final_rec.names))
            ]

    except KeyError:
        return [
            dict(label='', value='')
        ]


## callback function for details of selected recommended product
@app.callback(
    Output(component_id='prod_lipies_dist', component_property='figure'),
    [Input('rec_outin', 'value')]
    )
def update_prod_graph(input_value):
    # get uid of selected product
    selected_uid = prod_df[prod_df['names'] == input_value]['pro_ids']
    # select all rows with reviews for selected product
    selected = prod_ratings_no_rev.merge(selected_uid)

    lipies_dist = pd.DataFrame(selected['lipies'].value_counts()).sort_index()

    # return all the codes that should go into the 'figure' argument
    return {
            'data': [
                go.Bar(
                    x=lipies_dist.index,
                    y=lipies_dist['lipies'],
                    # set width so that bar charts dont spread out when there are fewer values than 5
                    width=0.8,
                    marker=dict(
                        color='#EB89B5'
                    ),
                    opacity=0.75
                             )
                # can add styling like column size, opacity etc
            ],
            'layout': go.Layout(
                title="Counts of ratings for product",
                bargap=0.2,
                # set axis range so that bar charts dont spread out when there are fewer values than 5
                xaxis=dict(range=[0, 6]),
                font=dict(color='#888')
            )
        }


## callback function to hide/display product rec
@app.callback(
    Output('prod_rec_toggle', 'style'),
    [Input('submit_button', 'n_clicks')]
    )
def toggle_graph(input_value):
    if input_value:
        return {'display': 'block'}
    else:
        return {'display': 'none'}

## callback function to hide/display product graph
@app.callback(
    Output('prod_graph_toggle', 'style'),
    [Input('rec_outin', 'value')]
    )
def toggle_graph(input_value):
    if input_value:
        return {'display': 'block'}
    else:
        return {'display': 'none'}



if __name__ == "__main__":
    app.run_server(debug=True)
