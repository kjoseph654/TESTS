# Run this app with `python meta-analysis-dash.py` and
# visit http://127.0.0.1:8050/ in your web browser.
'''
dashboard screen meta analysis (Secondary only)
'''
import os
import json, datetime, subprocess
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import dash
from dash import html, dcc, dash_table
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from utils import collate_manifests, \
    get_css_colors, analysis_dir, parent_dir, get_no_odor_wells, curve_fit
import numpy as np
from sklearn.decomposition import PCA

#%%

def pca_data(df_input, parameters):
 
    #file = 'C:/Users/kjose/Desktop/' + dataset + '.csv'
    df = df_input
    #These are currently the columns that are dropped. R^2 potentially has some uses for pre-filtering if we wanted
    df.drop(columns=['R^2','AUC', 'Saturated?'], inplace=True)

    #Melting the table
    df_melt = df.melt(id_vars=['Plate', 'OR'])
    df_melt['param'] = df_melt['OR'] + '_' + df_melt['variable']

    #Pivoting the table
    df_pivot = pd.pivot(df_melt, index='Plate', columns='param', values='value')

    #Filling the non-response samples with dummy values
    EC50_fill = 1
    Efficacy_fill = 0
    foldchange_fill = 1
    Hill_fill = 1
    
    #Creating the list of the dummy values for the size of the data frame (by column)
    fill_na_vals = [EC50_fill, Efficacy_fill, foldchange_fill, Hill_fill]*len(df_pivot.columns)
    fill_n_dict = {n:v for n,v in zip(df_pivot.columns, fill_na_vals)}
    
    #Filling the dummy values by column
    df_pivot.fillna(value=fill_n_dict, inplace=True)

    #Reindexing the table so that it sorts numerically instead of by string
    df_pivot = df_pivot.reindex(sorted(df_pivot.index,key=lambda d: (len(d), d)))
    
    #Pulling the parameters selected by the user, and removing them from the pivot table
    parameters_all = ['EC50','Efficacy','Fold-Change','HillSlope']
    parameter_input = parameters
    parameters_used = []
    for p in parameters_all:
        param = p in parameter_input
        if param == True:
            parameters_used.append('yes')
        else:
            parameters_used.append('no')
    
    n_factors = int(len(df_pivot.columns))
    parameters_index = list(range(n_factors))
    parameters_remove = parameters_used * int(n_factors/4)
    for idx, n in enumerate(parameters_index):
        if parameters_remove[idx] == 'yes':
            continue
        if parameters_remove[idx] == 'no':
            parameters_index[idx] = -1
    parameters_index = [i for i in parameters_index if i != -1]
        
    df_pivot = df_pivot.iloc[:, parameters_index]
    return df_pivot

def pca_plot(df_update, xsize = 500, ysize = 500, xlabel = 'PC1', 
             ylabel = 'PC2', fontsize = 16, Loadings = 'True', 
             loading_adjust = 1, move_label = 'None', z_score = 'True',
             m_size = 10, whiten = 'True'):
    
    #Performing the PCA
    df_update1 = df_update.iloc[:,2:]
    
    zscore = (z_score == 'True')
    X = df_update1.to_numpy()

    if zscore:
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        X[np.isnan(X)] = 0
        
    pca = PCA(n_components=2, whiten = (whiten == 'True'))
    X_r = pca.fit(X).transform(X)

    # build df for plotting the PCA
    df_pca_plot = pd.DataFrame(data={'PC1': X_r[:,0], 'PC2':X_r[:,1], 
                                     'Plate': df_update.iloc[:,1], 
                                     df_update.columns[0]: df_update.iloc[:,0]})

    #Coefficients of the Loadings
    coeff = np.transpose(pca.components_[0:2, :])
    
    label_x = 'PC1: ' + str(int(pca.explained_variance_ratio_[0]*100)) + '% Variance'
    label_y = 'PC2: ' + str(int(pca.explained_variance_ratio_[1]*100)) + '% Variance'
    
    fig = px.scatter(data_frame = df_pca_plot, x = 'PC1', y = 'PC2', 
                     color = df_update.columns[0], template = "simple_white",
                     width = xsize, height = ysize,
                     labels = {
                         'PC1': label_x,
                         'PC2': label_y})
    
    fig.update_xaxes(ticks="outside", tickwidth=2, tickcolor='black', ticklen=10)
    fig.update_yaxes(ticks="outside", tickwidth=2, tickcolor='black', ticklen=10, col=1)
    fig.update_layout(font = dict(family = "Arial",size = fontsize))
    fig.update_traces(marker={'size': m_size})
    if Loadings == 'True':
        for ind, i in enumerate(coeff):
            ar_input = df_update1.columns.values[ind]
            see = pd.DataFrame({'x': ([0,coeff[ind][0]]), 'y':([0, coeff[ind][1]])})
            fig.add_traces(
                list(px.line(see*loading_adjust, x = 'x', y= 'y').select_traces())
                )
            fullstring = move_label
            substring = ar_input
            if substring in fullstring:
                fig.add_annotation(x = ((loading_adjust+0.25)*(coeff[ind][0])), y = (loading_adjust+0.25)*(coeff[ind][1]),
                                   text = ar_input,
                                   showarrow = False)
            else:
                fig.add_annotation(x = (loading_adjust+0.25)*(coeff[ind][0]), y = (loading_adjust+0.25)*(coeff[ind][1]),
                                   text = ar_input,
                                   showarrow = False)
    return fig
#%%
# Dash setup
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
which_variable_options = [{'label': p, 'value': p} for p in [
    'Normalized', 'Firefly_bgnorm', 'Background_Subtracted', 'Background_Divided', 'Firefly', 'Renilla']]

# colormap
colormap = px.colors.qualitative.Dark24
# shift by 1 color
colormap = np.roll(np.array(colormap),1)

df_list = []

# git commit id and code timestamp
commit_id = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').rstrip()
commit_timestamp = subprocess.check_output('git show -s --format=%ci HEAD'.split(' ')).decode('ascii').rstrip()

# color options
css_color_options = [{'label': c, 'value':c} for c in get_css_colors()]
css_color_options.insert(0,{'label':'Auto', 'value':'Auto'})
xscale_options = [{'label': p, 'value': p}
                  for p in ['Log x scale', 'Linear x scale']]
fit_type_options = [{'label': p, 'value': p}
                  for p in ['3-parameter', '4-parameter']]

marker_options = [{'label': p, 'value': p}
                  for p in ['circle', 'square', 'diamond', 'cross']]

normalization_options = [{'label': p, 'value': p}
                  for p in ['None', 'Minimum concentration or distance', 'Maximum concentration or distance']]

# concentration options
default_conc_options = []

def load_manifest():
    df_manifest, _ = collate_manifests(screen_type='Secondary')
    return df_manifest

EC50 = dcc.Checklist(['EC50','Efficacy','Fold-Change','HillSlope'],
                     ['EC50','Efficacy','Fold-Change','HillSlope'], inline = True)

def get_layout():
    # generate combined manifest file
    df_manifest = load_manifest()
    
    # populate Project items
    project_checklist_options = df_manifest['rel_data_dir'].unique()
    project_checklist_options = [{'label': os.path.normpath(p).split(os.sep)[-2], 'value': p} for p in project_checklist_options]
    
    return html.Div([
        dcc.Tabs([
            dcc.Tab(label='Dose response', children=[
                html.Div(children=[
                    html.H1(children='AR dose response dashboard'),
                    html.Div([
                        html.Div([
                            html.Div('Variable', style={'font-weight': 'bold'}),
                            dcc.RadioItems(  # which_variable selector
                                id='which_variable_radio',
                                options=which_variable_options,
                                value='Normalized'
                            ),
                            html.Br(),
                            html.Div('Plot style', style={'font-weight': 'bold'}),
                            dcc.RadioItems(  # which_variable selector
                                id='single_points_radio',
                                options=[{'label': 'Mean+/-st.dev', 'value': 'std'}, 
                                          {'label':'Mean+/-s.e.m.', 'value':'sem'},
                                          {'label':'Single datapoints', 'value':'Single datapoints'}],
                                           value='sem'
                            ),
                            html.Br(),
                            html.Div('X scale', style={'font-weight': 'bold'}),
                            dcc.RadioItems(  # Log x selector
                                id='log_x_radio',
                                options=xscale_options,
                                value='Log x scale'
                            ),
                            html.Br(),
                            html.Div('Fit type', style={'font-weight': 'bold'}),
                            dcc.RadioItems(  # Log x selector
                                id='fit_type_radio',
                                options=fit_type_options,
                                value='3-parameter'
                            ),
                            html.Br(),
                            html.Div('Subtract...', style={'font-weight': 'bold'}),
                            dcc.RadioItems(  # Log x selector
                                id='norm_type_radio',
                                options=normalization_options,
                                value='None'
                            ),
                            html.Br(),
                            html.Br(),
                            html.Button("Download raw CSV", id="btn-download-raw"),
                            dcc.Download(id="download-raw-csv")
                        ], className="three columns"),
                        html.Div([
                            html.Form(autoComplete='off', children=[
                                html.Div('Project', style={'font-weight': 'bold'}),
                                dcc.Dropdown(  # project dropdown
                                    id='project-dropdown',
                                    options=project_checklist_options,
                                ),
                                html.Br(),
                                html.Div('Experiment', style={'font-weight': 'bold'}),
                                dcc.Dropdown(  # project dropdown
                                    id='experiment-dropdown'
                                ),
                                html.Br(),
                                html.Div('Plate', style={'font-weight': 'bold'}),
                                dcc.Checklist(  # plate dropdown
                                    id='plate-dropdown', inline = True
                                ),
                                html.Br(),
                                html.Div('OR', style={'font-weight': 'bold'}),
                                dcc.Dropdown(  # OR dropdown
                                    id='OR-dropdown',
                                ),
                                html.Br(),
                                html.Div([html.Div([html.Div('Color', style={'font-weight': 'bold'}, id='color-title'),
                                          dcc.Dropdown( # line color dropdow
                                             id='color-dropdown',
                                             options=css_color_options,
                                             value='Auto')], className="six columns"),
                                          html.Div([html.Div('Marker', style={'font-weight': 'bold'}),
                                          dcc.Dropdown( # line color dropdow
                                             id='markersymbol-dropdown',
                                             options=marker_options,
                                             value='circle')], className="six columns")
                                          ], className='row'),
                                ]),
                            html.Br(),
                            html.Div('Plot parameters', style={'font-weight': 'bold'}),
                            html.Div(id='params-left', children=[html.Div('X label'), 
                                               dcc.Input(id='x-axis-label',value='Concentration', placeholder='Concentration'),
                                               html.Div('Y label'),
                                               dcc.Input(id='y-axis-label',value='Activation', placeholder='Activation'),
                                               html.Div(children=[html.Div('Axis fontsize'), 
                                                                  dcc.Input(id='axis-label-fontsize', type='number', value=16),
                                                                  html.Div('Tick fontsize'), 
                                                                  dcc.Input(id='tick-label-fontsize', type='number', value=16)
                                               ])
                                               ], className='six columns'),
                            html.Div([html.Div('Marker size'), 
                                      dcc.Input(id='plot-markersize', type='number', value=10),
                                      html.Div('Linewidth'),
                                      dcc.Input(id='plot-linewidth', type='number', value=2),
                                      html.Div(children=[html.Div('Figure width x height'),
                                               dcc.Input(id='fig-width', type='number', min=900, value=1200),
                                               dcc.Input(id='fig-height', type='number', value=600)])
                                      ]),
                            
                            html.Div([html.Button('Update', id='add-curve-btn', n_clicks=0),
                                      html.Button('Clear all', id='clear-curves-btn', n_clicks=0)
                                      ]),
                            html.Br(),
                            html.Div('Concentrations/Distances to ignore', style={'font-weight': 'bold'}),
                            dcc.Checklist( # concentration dropdown
                                         id='conc-checklist',
                                        options = default_conc_options,
                                        value = default_conc_options,
                                        labelStyle={'display': 'inline-block'}
                                        ),
                            ], className="six columns"),
                    ], className='row'),
                    html.Div(id='fig-panel'),
                    html.Div('Time: ' + str(datetime.datetime.now()), style={'fontSize': 10}),
                    html.Div('Code revision id: ' + commit_id, style={'fontSize': 10}),
                    html.Div('Code timestamp: ' + commit_timestamp, style={'fontSize': 10}),
                    dcc.Store(id='selected-data-all'),
                    dcc.Store(id='export-raw-data'),
                    dcc.Store(id='color-list'),
                    dcc.Store(id='marker-list'),
                    dcc.Store(id='manifest', data=df_manifest.to_json(orient='split'))
                    ])
                ]),
            dcc.Tab(label='PCA', children=[
                html.H1('PCA dashboard'),
                html.Label("Which parameters do you want?"),
                EC50,
                html.Br(),
                html.Button(id='pca-submit',                
                        children='Pull Data from the Dose response table'),
                html.Br(),
                html.Div(id='pca'),
                html.Br(),
                html.Div(id ='pca-table')
                ])
            ])
                    
        ])
    
    
app.layout = get_layout

@app.callback(
    Output("download-raw-csv", "data"),
    Input("btn-download-raw", "n_clicks"),
    State('selected-data-all', 'data'),
    State('which_variable_radio', 'value'),
    State('conc-checklist', 'value'),
    State('log_x_radio', 'value'),
    prevent_initial_call=True,
    suppress_callback_exceptions=True
)
def download_raw_csv(n_clicks, jsonified_data_all, which_variable, conc_selection, log_x_choice):
    '''
    Download raw displayed data as CSV

    Parameters
    ----------
    n_clicks : TYPE
        DESCRIPTION.
    jsonified_data_all : TYPE
        DESCRIPTION.
    conc_selection : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    log_x = (log_x_choice == xscale_options[0]['value'])
    
    export_cols = ['Project', 'Plate', 'OR', 'conc', which_variable]

    df_list = read_json_pd_list(jsonified_data_all)
    df_all = pd.concat(df_list)
    
    # if empty, do not return anything
    if df_all.empty:
        raise PreventUpdate
    df_all = df_all[~df_all['conc'].isin(conc_selection)]
    
    # if log x scale, make conc column log
    if log_x:
        df_all['conc'] = np.log10(df_all['conc'])
    
    # set no-odor concentration to 0
    df_all.loc[df_all['plate-type'] == 'no-odor','conc'] = 0
    df_all = df_all[export_cols]

    # make user friendly table for display
    df_all['Replicate'] = df_all.groupby(['Project', 'Plate', 'OR', 'conc']).cumcount()
    df_all_unstack = df_all.set_index(['Project', 'OR', 'Plate', 'conc', 'Replicate']).unstack()
    
    return dcc.send_data_frame(df_all_unstack.to_csv, "raw_data.csv")

@app.callback(Output('color-title', 'style'),
              Input('color-dropdown', 'value'))
def update_color_sample(color_value):
    return {'background-color': color_value, 'font-weight':'bold'}

@app.callback(Output('experiment-dropdown', 'options'),
              Input('project-dropdown', 'value'),
              State('manifest', 'data'))

def update_experiment_options(selected_project, jsonified_manifest):
    df_manifest = pd.read_json(jsonified_manifest, orient='split')
    if selected_project is not None:
        plates = df_manifest[df_manifest['rel_data_dir'] == selected_project]['proj_dir'].unique()
        plate_checklist_options = [{'label': p, 'value': p} for p in plates]
        return plate_checklist_options
    else:
        return [{'label': 'none', 'value': 'none'}]

@app.callback(Output('plate-dropdown', 'options'),
              Input('experiment-dropdown', 'value'),
              Input('project-dropdown', 'value'),
              State('manifest', 'data'))

def update_plate_options(selected_exp, selected_project, jsonified_manifest):
    df_manifest = pd.read_json(jsonified_manifest, orient='split')
    if selected_exp is not None:
        plates = df_manifest[(df_manifest['rel_data_dir'] == selected_project) & 
                             (df_manifest['proj_dir'] == selected_exp)]['plate_prefix']
        plate_checklist_options = [{'label': p, 'value': p} for p in plates]
        return plate_checklist_options
    else:
        return [{'label': 'none', 'value': 'none'}]


@app.callback(Output('OR-dropdown', 'options'),
              Input('plate-dropdown', 'value'),
              Input('experiment-dropdown', 'value'),
              Input('project-dropdown', 'value'))

def update_OR_options(selected_plate, selected_experiment, selected_project):

    if (selected_project is not None) and (selected_plate is not None):
        df_combo = pd.read_csv(os.path.join(parent_dir, selected_project, analysis_dir, 'combo.csv'))
        df_combo = df_combo[(df_combo['Project'] == selected_experiment) &
                    (df_combo['Plate'] == selected_plate[0])]
        OR_options = df_combo['OR'].unique()
    
        return [{'label': p, 'value': p} for p in OR_options]
    else:
        return [{'label': 'none', 'value': 'none'}]


def read_single_df_from_json(df_json):
    json_read = json.loads(df_json)
    dff = pd.DataFrame(index = json_read['index'], columns=json_read['columns'], data=json_read['data'])
    return dff

def read_json_pd_list(json_str):
    #return [pd.read_json(a, orient='split') for a in json.loads(json_str)]
    return [read_single_df_from_json(a) for a in json.loads(json_str)]

def write_json_pd_list(pd_list):
    return json.dumps([a.to_json(orient='split') for a in pd_list])


@app.callback(
    Output('fig-panel', 'children'),
    Output('selected-data-all', 'data'),
    Output('color-list', 'data'),
    Output('marker-list', 'data'),
    Output('conc-checklist', 'options'),
    Input('add-curve-btn', 'n_clicks'),
    Input('clear-curves-btn', 'n_clicks'),
    Input('which_variable_radio', 'value'),
    Input('single_points_radio', 'value'),
    Input('fit_type_radio', 'value'),
    Input('log_x_radio', 'value'),
    Input('norm_type_radio', 'value'),
    State('conc-checklist', 'value'),
    State('selected-data-all', 'data'),
    State('project-dropdown', 'value'),
    State('experiment-dropdown', 'value'),
    State('plate-dropdown', 'value'),
    State('OR-dropdown', 'value'),
    State('color-list', 'data'),
    State('color-dropdown', 'value'),
    State('x-axis-label', 'value'),
    State('y-axis-label', 'value'),
    State('axis-label-fontsize', 'value'),
    State('tick-label-fontsize', 'value'),
    State('plot-linewidth', 'value'),
    State('plot-markersize', 'value'),
    State('fig-width', 'value'),
    State('fig-height', 'value'),
    State('marker-list', 'data'),
    State('markersymbol-dropdown', 'value')
)





#i am thinking that i will need to move the function to the top and change the return value 






def update_fig(add_curves_btn=0, clear_curves_btn=0, which_variable='Normalized', single_points='sem', fit_type='3-parameter',
               log_x_choice=xscale_options[0]['value'], norm_type=normalization_options[0], conc_selection = default_conc_options, jsonified_data_all='', 
               selected_project=None, selected_experiment=None, selected_plate=None,
               selected_OR=None, jsonified_color_list='', selected_color='Auto', xaxis_label='Concentration', yaxis_label='Activation',
               label_fontsize=14, tick_fontsize=12, plot_linewidth=2, plot_markersize=10,
               fig_width=1200, fig_height=500, jsonified_marker_list='', selected_marker='circle'):

    ctx = dash.callback_context
    
    # un-triggered
    if not ctx.triggered:
        return [html.Div('No data to display'), 
                write_json_pd_list([pd.DataFrame()]),
                json.dumps(['Auto']), 
                json.dumps(['circle']),
                []
                ]
    
    # clear button
    elif 'clear-curves-btn' in ctx.triggered[0]['prop_id']:
        return [html.Div('No data to display'), 
                write_json_pd_list([pd.DataFrame()]),
                json.dumps(['Auto']), 
                json.dumps(['circle']),
                []
                ]
    
    # assume log is first options
    log_x = (log_x_choice == xscale_options[0]['value'])

    if (selected_project is not None) and (selected_experiment is not None) and (selected_plate is not None) and (selected_OR is not None):
        # new data to be plotted
        df_combo = pd.read_csv(os.path.join(parent_dir, selected_project, analysis_dir, 'combo.csv'))
        df_filt = df_combo[(df_combo['OR'] == selected_OR) & (df_combo['Project'] == selected_experiment) &
                 (df_combo['plate-type'] == 'sample') & (df_combo['Plate'] == selected_plate)]
        df_neg = df_combo[(df_combo['OR'] == selected_OR) & (df_combo['plate-type'] == 'no-odor')]
        df_neg.to_csv('neg.csv')
        df_neg = get_no_odor_wells(df_neg, selected_experiment)
        df_filt = df_filt.append(df_neg)
            
            
        if jsonified_data_all == '':  # first plot
            df_list = [df_filt]
            color_list = ['', selected_color]
            marker_list = ['', selected_marker]
        else:  # subsequent plots   
            df_list = read_json_pd_list(jsonified_data_all)  # load df_list
            color_list = json.loads(jsonified_color_list)
            marker_list = json.loads(jsonified_marker_list)
            cols_to_compare = ['OR', 'Project', 'Plate', 'plate-type']
            # if same element as previously, do not add to list
            # have to compare the cols_to_compare columns bc json messes with floats
            if (df_list[-1].empty) or not (df_list[-1][cols_to_compare].equals(df_filt[cols_to_compare])):
                df_list.append(df_filt)
                color_list.append(selected_color)
                marker_list.append(selected_marker)
        fig = go.Figure(data=[go.Scatter(x=[], y=[])])

        stats_list = []
        
        # find all concentration options from df list
        conc_options = np.array([dff['conc'].to_numpy() for dff in df_list[1:]])
        conc_options = np.unique(np.concatenate(conc_options))
        
        for i, df_sub in enumerate(df_list):
            if not df_sub.empty:
                
                # color to use: if auto, use colormap, otherwise use user-selected
                plot_color = color_list[i] if (color_list[i] != 'Auto') else colormap[i]
                plot_markersymbol = marker_list[i]
                
                df_sample = df_sub[(df_sub['plate-type'] != 'no-odor')].copy()
                
                # filter to only include user-selected concentrations
                df_sample = df_sample[~df_sample['conc'].isin(conc_selection)]
                
                # min max tested concentrations
                min_conc, max_conc = (np.min(df_sample['conc']), np.max(df_sample['conc']))
                
                if norm_type == normalization_options[0]['value']: # normalize to no-odor
                    # separate sample and no-odor entries
                    df_neg = df_sub[(df_sub['plate-type'] == 'no-odor')]
                    if which_variable in ['Normalized', 'Firefly', 'Renilla']:
                        mean_noodor = np.mean(df_neg[which_variable])
                    elif which_variable == 'Background_Divided':
                        mean_noodor = 1
                    else:
                        mean_noodor = 0
                elif norm_type == normalization_options[1]['value']: # normalize to min concentration / distance
                    # A = (A / A_min) - 1
                    df_sample[which_variable] = df_sample[which_variable] - (df_sample[df_sample['conc'] == min_conc][which_variable].mean())
                    mean_noodor = 0
                else: # normalize to max concentration / distance
                    # A = (A / A_max) - 1
                    df_sample[which_variable] = df_sample[which_variable] - (df_sample[df_sample['conc'] == max_conc][which_variable].mean())
                    mean_noodor = 0
                    
                df_sample.to_csv('df_sample.csv')
                
                
                # 3- or 4-parameter curve fitting
                curve_fit_4_param_bool = ('4' in fit_type)
                
                x_fit, y_fit, stats = curve_fit(
                    df_sample, 'conc', which_variable, log_x, curve_fit_4_param_bool)
                
                
                df_sub_grouped = df_sample.groupby(
                    'conc')[which_variable].agg(['mean', 'std', 'sem'])
                df_sub_grouped['conc'] = df_sub_grouped.index

                # min max fit concentrations
                min_conc_fit, max_conc_fit = (x_fit[0], x_fit[-1])
                
                '''
                AUC: for now, just sum up points above no-odor
                '''
                stats['AUC'] = (df_sub_grouped['mean']-mean_noodor).sum()
                
                #Kenny addition for OR/Plate Names
                stats['OR'] = df_sample.iloc[0]['OR']
                stats['Plate'] = df_sample.iloc[0]['Plate']
                               
                '''
                calculating saturation
                consider that saturation has been reached if there exist at least 2 concentrations > EC50 that were tested
                '''
                stats['Saturated?'] = (sum(df_sub_grouped.index > stats['EC50'])) > 2
                
                '''
                fold-change = Max response / Response(no-odor)
                if mean_noodor == 0, set to NaN
                '''
                if mean_noodor != 0:
                    stats['Fold-change'] = df_sub_grouped['mean'].max() / mean_noodor
                else:
                    stats['Fold-change'] = np.nan
                '''
                Top-Bottom of fit (efficacy)
                '''
                stats['Efficacy (fit)'] = stats['Top'] - stats['Bottom']
                
                '''
                Hill coefficient
                 - if already exists, list it; otherwise, set to 1 (means it was a 3-param fit)
                '''
                if 'HillSlope' not in stats.keys():
                    stats['HillSlope'] = 1
                    
                
                stats_list.append(stats)
                
                if single_points == 'Single datapoints':  # single points
                    scatterplot = go.Scatter(
                        x=np.log10(df_sample['conc']
                                   ) if log_x else df_sample['conc'],
                        y=df_sample[which_variable],
                        mode='markers',
                        name=stats['id'],
                        marker=dict(color=plot_color, size=plot_markersize, symbol=plot_markersymbol)
                    )
                else:  # errorbars
                    scatterplot = go.Scatter(# datapoints
                        x=np.log10(
                            df_sub_grouped['conc']) if log_x else df_sub_grouped['conc'],
                        y=df_sub_grouped['mean'],
                        error_y=dict(
                            type='data',
                            array=(df_sub_grouped[single_points]),
                            color=plot_color,
                            thickness=plot_linewidth),
                        name=stats['id'],
                        mode='markers',
                        marker=dict(color=plot_color, size=plot_markersize, symbol=plot_markersymbol))

                fig.add_trace(scatterplot).add_trace(go.Scatter(x=np.log10(x_fit) if log_x else x_fit, # fit line
                                                                y=y_fit,
                                                                mode='lines',
                                                                name=stats['id'],
                                                                line=dict(
                                                                    color=plot_color,
                                                                    width=plot_linewidth)
                                                                )
                                                     ).add_trace(go.Scatter(x=np.log10([min_conc_fit, max_conc_fit]) if log_x else [min_conc, max_conc],  # no-odor horizontal line
                                                                            y=[mean_noodor]*2,
                                                                            mode='lines',
                                                                            name=stats['id'],
                                                                            line=dict(
                                                                                    color=plot_color, 
                                                                                    width=plot_linewidth, 
                                                                                    dash='dot')
                                                     )
                ).update_layout(width=fig_width,
                                height=fig_height,
                                margin=dict(r=700),
                                plot_bgcolor='rgba(0, 0, 0, 0)',
                                paper_bgcolor='rgba(0, 0, 0, 0)',
                                legend=dict(x=1,
                                            y=1,
                                            xanchor='left',
                                            yanchor='top'
                                            ),
                                xaxis={'title': dict(text=xaxis_label, font_size=label_fontsize, font=dict(family='Arial'))},
                                yaxis={'title': dict(text=yaxis_label, font_size=label_fontsize, font=dict(family='Arial'))},
                                uirevision='constant')
        fig=fig.update_yaxes(showline=True, linewidth=3, tickwidth=3,
                         ticklen=7, linecolor='black', ticks='outside', showgrid=False, tickfont=dict(size=tick_fontsize, family='Arial'))
        fig=fig.update_xaxes(ticks='outside', showline=True, linewidth=3,
                         tickwidth=3, ticklen=10, linecolor='black', showgrid=False, tickfont=dict(size=tick_fontsize, family='Arial'))
        
        config = {
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            'filename': 'image',
            'scale': 3, # Multiply title/legend/axis/canvas sizes by this factor
            # 'height': 400,
            # 'width': 400
            }
        }
        
#                 *[html.Div(s['id'] + ': EC50={:.5e}, Top-Bottom of fit={:.3f}, fold-change (min to max concentration): {:.2f}'.format(
#                     s['EC50'], s['Top']-s['Bottom'], s['fold-change'])) for s in stats_list],

        hits_table = pd.DataFrame(data=stats_list)
        
        cols = ['Plate', 'OR', 'EC50', 'Fold-change', 'HillSlope', 'Efficacy (fit)', 'AUC', 'R^2', 'Saturated?']
        hits_table = hits_table[cols]
        
        # with open('df_list.json', 'w') as f:
        #     f.write(write_json_pd_list(df_list))
        
        return (html.Div(children=[
            dcc.Graph(figure=fig, config=config, id='dose-response-fig'),
                                   dash_table.DataTable(id = 'dose-response-table', data=hits_table.to_dict('records'),
                                                        columns=[{"name": i, "id": i} for i in hits_table.columns],
                                                        export_format='csv'
                                                        ),
                                   html.Br(),
                                   html.Button(id='pca-submit',                
                                           children='Submit'),
                                   html.Br()
                ]),
                write_json_pd_list(df_list),
                json.dumps(color_list),
                json.dumps(marker_list),
                [{'label': c ,'value':c} for c in conc_options]
                )
    else:
        return [html.Div(), 
                write_json_pd_list([pd.DataFrame()]),
                json.dumps(['Auto']), 
                json.dumps(['circle']),
                []
                ]
    

@app.callback(
    Output('pca-table','children'),
    Input('pca-submit','n_clicks'),
    State(EC50, 'value'),
    State('dose-response-table', 'data'),
    State('dose-response-table', 'columns')
)
def update_table(clicks, parameters, rows, columns):
    if clicks is not None:
        df_input = pd.DataFrame(rows, columns=[c['name'] for c in columns])
        df_pivot = pca_data(df_input, parameters)
        df_pivot.to_csv('df_input.csv')
        return dash_table.DataTable(id = 'pca-input',
                                columns = (
                                    [{'id': 'Groups', 'name': 'Groups'}] +
                                    [{'id': 'Sample', 'name': 'Sample'}] +
                                    [{'id': p, 'name': p, 'deletable': True} for p in df_pivot.columns]
                                    ),
                                data = [
                                    dict(Groups = df_pivot.index[i-1], Sample = df_pivot.index[i-1], **{p: df_pivot[p][i-1] for p in df_pivot.columns})
                                    for i in range(1, len(df_pivot.index)+1)], editable = True,
                                column_selectable = "multi")

@app.callback(
    Output('pca', 'children'),
    Input('pca-input', 'data'),
    Input('pca-input', 'columns')
    )

def update_graph(rows, columns):
    df_new = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    fig = pca_plot(df_new)
    return html.Div(children =[
        dcc.Graph(id = 'pca-graph', figure = fig)
        ])

if __name__ == '__main__':
    app.run_server(debug=True)
