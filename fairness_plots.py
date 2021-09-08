import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, false_positive_rate, true_positive_rate, \
                            selection_rate, count, _mean_overprediction, _mean_underprediction



def predictions_in_accuracy_score(y_true, predictions, sensitive_features):
    # Predictions in accuracy_score
    st.markdown("<h2 style='text-align: center; color: black;'>Disparity in performance</h2>", unsafe_allow_html=True)
    gm = MetricFrame(accuracy_score, y_true, predictions, sensitive_features=sensitive_features)
    gm_overall = str(round(gm.overall * 100, 2))
    print('gm overall :', gm_overall)
    disparity_in_gm = str(round(gm.difference() * 100, 2))
    print(disparity_in_gm)
    gm_sensitive_feature_dict = gm.by_group.to_dict()
    print(gm_sensitive_feature_dict)
    y = ["{} : {}%".format(name, str(round(value * 100, 2))) for name, value in gm_sensitive_feature_dict.items()]
    print(y)
    under = MetricFrame(_mean_underprediction, y_true, predictions, sensitive_features=sensitive_features)
    under_sensitive_feature_dict = under.by_group.to_dict()
    under_x = [-value * 100 for value in under_sensitive_feature_dict.values()]
    under_x_text = [str(round(value, 2)) + "%" for value in under_x]
    print(under_sensitive_feature_dict)
    print(under_x)
    print(under_x_text)
    over = MetricFrame(_mean_overprediction, y_true, predictions, sensitive_features=sensitive_features)
    over_sensitive_feature_dict = over.by_group.to_dict()
    over_x = [value * 100 for value in over_sensitive_feature_dict.values()]
    over_x_text = [str(round(value, 2)) + "%" for value in over_x]
    print(over_sensitive_feature_dict)
    print(over_x)
    print(over_x_text)
    st.markdown("<h3 style='text-align: center; color: black;'>Overall accuracy: {}%  |  Disparity in accuracy: {}% </h3>".format(gm_overall, disparity_in_gm), unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=y,
                         x=under_x,
                         name="Underprediction \n (predicted=0, true=1)",
                         orientation="h",
                         marker=dict(color="#FF7F0E"),
                         text=under_x_text,
                         textposition='inside'
                         ))
    fig.add_trace(go.Bar(y=y,
                          x=over_x,
                          name="Overprediction \n (predicted=1, true=0)",
                          orientation="h",
                          marker=dict(color="#1F77B4"),
                          text=over_x_text,
                          textposition='inside'
                          ))
    fig.add_hline(y=0.5, line_color="lightgrey")
    fig.add_vline(x=0, line_width=1)
    fig.update_layout(barmode="relative",
                      xaxis=dict(ticksuffix="%"),
                      width=1000,
                      height=500,
                      title={'text':"<sup>The bar chart shows the distribution of errors in each group.<br>Errors are split into overprediction errors (predicting 1 when the true label is 0), and underprediction errors (predicting 0 when the true label is 1). <br>The reported rates are obtained by dividing the number of errors by the overall group size.</sup>",
                             'font': {'color': 'black',
                                      'size':18},
                              'xanchor':'auto',
                              'yanchor':'auto',
                            })
    st.plotly_chart(fig)


def predictions_in_selection_rate(y_true, predictions, sensitive_features):
    # Predictions in selection_rate
    st.markdown("<h2 style='text-align: center; color: black;'>Disparity in predictions</h2>", unsafe_allow_html=True)
    sr = MetricFrame(selection_rate, y_true, predictions, sensitive_features=sensitive_features)
    sr_sensitive_feature_dict = sr.by_group.to_dict()
    print('selection rate: ', sr_sensitive_feature_dict)
    y = ["{} : {}%".format(name, str(round(value * 100, 2))) for name, value in sr_sensitive_feature_dict.items()]
    print(y)
    sr_overall = str(round(sr.overall * 100 , 2))
    print('overall :', sr_overall)
    disparity_in_sr = str(round(sr.difference() * 100 , 2))
    print(disparity_in_sr)
    sr_x = [value * 100 for value in sr_sensitive_feature_dict.values()]
    sr_x_text = [str(round(value, 2)) + "%" for value in sr_x]
    print(sr_x)
    print(sr_x_text)
    st.markdown("<h3 style='text-align: center; color: black;'>Overall selection rate: {}%  |  Disparity in selection rate: {}% </h3>".format(sr_overall, disparity_in_sr), unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=y,
                         x=sr_x,
                         orientation="h",
                         marker=dict(color="#1F77B4"),
                         text=sr_x_text,
                         textposition='inside'
                         ))
    fig.update_layout(barmode="relative",
                      xaxis=dict(ticksuffix="%"),
                      width=800,
                      height=500,
                      title={'text':"<sup>The bar chart shows the selection rate in each group, meaning the fraction of points classified as 1.</sup>",
                             'font': {'color': 'black',
                                      'size':18},
                             'xanchor':'auto',
                             'yanchor':'auto',
                             })
    st.plotly_chart(fig)


def plot_metric(metric, metric_name, y_true, predictions, sensitive_features, text):
    st.markdown("<h2 style='text-align: center; color: black;'>{}</h2>".format(text),
                                    unsafe_allow_html=True)
    metric_frame = MetricFrame(metrics=metric,
                               y_true=y_true,
                               y_pred=predictions,
                               sensitive_features=sensitive_features)
    metric_frame_dict = dict(metric_frame.by_group)
    print('metric frame dict: ', metric_frame_dict)
    x = list(metric_frame_dict.keys())
    print(x)
    y = list(metric_frame_dict.values())
    print(y)
    y_text = [str(round(value * 100, 2)) + "%" if metric_name != 'count' else value for value in y]
    print(y_text)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=y,
                         x=x,
                         orientation="v",
                         marker=dict(color="#1F77B4"),
                         text=y_text,
                         textposition='inside'
                         ))
    fig.update_layout(width=550,
                      height=550,
                      yaxis_tickformat="%" if metric_name != 'count' else "",
                      title={'text':"Performance in {}".format(metric_name),
                             'y': 0.85,
                             'x': 0.5,
                             'font': {'color': 'black',
                                      'size':18}
                              })
    st.plotly_chart(fig)


def subplots_in_accuracy_score(y_true, unmitigated_predictions, mitigated_predictions, sensitive_features):
    # Predictions in accuracy_score
    st.markdown("<h2 style='text-align: center; color: black;'>Disparity in performance</h2>", unsafe_allow_html=True)

    unmitigated_gm = MetricFrame(accuracy_score, y_true, unmitigated_predictions, sensitive_features=sensitive_features)
    mitigated_gm = MetricFrame(accuracy_score, y_true, mitigated_predictions, sensitive_features=sensitive_features)

    unmitigated_gm_overall = str(round(unmitigated_gm.overall * 100, 2))
    mitigated_gm_overall = str(round(mitigated_gm.overall * 100, 2))

    unmitigated_disparity_in_gm = str(round(unmitigated_gm.difference() * 100, 2))
    mitigated_disparity_in_gm = str(round(mitigated_gm.difference() * 100, 2))

    unmitigated_gm_sensitive_feature_dict = unmitigated_gm.by_group.to_dict()
    mitigated_gm_sensitive_feature_dict = mitigated_gm.by_group.to_dict()

    unmitigated_y = ["{} : {}%".format(name, str(round(value * 100, 2))) for name, value in
                     unmitigated_gm_sensitive_feature_dict.items()]
    mitigated_y = ["{} : {}%".format(name, str(round(value * 100, 2))) for name, value in
                   mitigated_gm_sensitive_feature_dict.items()]

    unmitigated_under = MetricFrame(_mean_underprediction, y_true, unmitigated_predictions,
                                    sensitive_features=sensitive_features)
    mitigated_under = MetricFrame(_mean_underprediction, y_true, mitigated_predictions,
                                  sensitive_features=sensitive_features)

    unmitigated_under_sensitive_feature_dict = unmitigated_under.by_group.to_dict()
    mitigated_under_sensitive_feature_dict = mitigated_under.by_group.to_dict()

    unmitigated_under_x = [-value * 100 for value in unmitigated_under_sensitive_feature_dict.values()]
    mitigated_under_x = [-value * 100 for value in mitigated_under_sensitive_feature_dict.values()]

    unmitigated_under_x_text = [str(round(value, 2)) + "%" for value in unmitigated_under_x]
    mitigated_under_x_text = [str(round(value, 2)) + "%" for value in mitigated_under_x]

    unmitigated_over = MetricFrame(_mean_overprediction, y_true, unmitigated_predictions,
                                   sensitive_features=sensitive_features)
    mitigated_over = MetricFrame(_mean_overprediction, y_true, mitigated_predictions,
                                 sensitive_features=sensitive_features)

    unmitigated_over_sensitive_feature_dict = unmitigated_over.by_group.to_dict()
    mitigated_over_sensitive_feature_dict = mitigated_over.by_group.to_dict()

    unmitigated_over_x = [value * 100 for value in unmitigated_over_sensitive_feature_dict.values()]
    mitigated_over_x = [value * 100 for value in mitigated_over_sensitive_feature_dict.values()]

    unmitigated_over_x_text = [str(round(value, 2)) + "%" for value in unmitigated_over_x]
    mitigated_over_x_text = [str(round(value, 2)) + "%" for value in mitigated_over_x]

    st.markdown("<h3 style='text-align: center; color: black;'>After Mitigation: Overall accuracy: {}%  |  Disparity in accuracy: {}% </h3>".format(mitigated_gm_overall, mitigated_disparity_in_gm), unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: black;'>Before Mitigation: Overall accuracy: {}%  |  Disparity in accuracy: {}% </h3>".format(unmitigated_gm_overall, unmitigated_disparity_in_gm), unsafe_allow_html=True)
    #fig = go.Figure()
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=0, shared_xaxes=True, shared_yaxes=True, subplot_titles=("Mitigated Model","", "Unmitigated Model","") )
    fig.append_trace(go.Bar(y=mitigated_y,
                         x=mitigated_under_x,
                         # name="Underprediction after mitigation\n (predicted=0, true=1)",
                         name="Underprediction after mitigation",
                         orientation="h",
                         marker=dict(color="#DC3912"),
                         text=mitigated_under_x_text,
                         textposition='inside'
                         ),row=1, col=1)
    fig.append_trace(go.Bar(y=mitigated_y,
                            x=mitigated_over_x,
                            name="Overprediction after mitigation",
                            orientation="h",
                            marker=dict(color="#0099C6"),
                            text=mitigated_over_x_text,
                            textposition='inside'
                            ), row=1, col=2)
    fig.append_trace(go.Bar(y=unmitigated_y,
                            x=unmitigated_under_x,
                            name="Underprediction before mitigation",
                            orientation="h",
                            marker=dict(color="#FF7F0E"),
                            text=unmitigated_under_x_text,
                            textposition='inside'
                            ), row=2, col=1)
    fig.append_trace(go.Bar(y=unmitigated_y,
                            x=unmitigated_over_x,
                            #name="Overprediction \n (predicted=1, true=0)",
                            name="Overprediction before mitigation",
                            orientation="h",
                            marker=dict(color="#1F77B4"),
                            text=unmitigated_over_x_text,
                            textposition='inside'
                            ), row=2, col=2)
    fig.add_hline(y=0.5, line_color="lightgrey")
    fig.add_vline(x=0, line_width=1)
    fig.update_layout(barmode="relative",
                      xaxis=dict(ticksuffix="%"),
                      width=1000,
                      height=500,
                      # title={'text':"<sup>The bar chart shows the distribution of errors in each group.<br>Errors are split into overprediction errors (predicting 1 when the true label is 0), and underprediction errors (predicting 0 when the true label is 1). <br>The reported rates are obtained by dividing the number of errors by the overall group size.</sup>",
                      #        'font': {'color': 'black',
                      #                 'size':18},
                      #         'xanchor':'auto',
                      #         'yanchor':'auto',
                      #       }
                      )
    x = [0.5, 0.5]
    y = [1, 0.375]
    for annotation, x, y in zip(fig['layout']['annotations'], x, y):
        annotation['x']= x
        annotation['y']= y
    st.plotly_chart(fig)


def subplots_in_selection_rate(y_true, unmitigated_predictions, mitigated_predictions, sensitive_features):

    # Predictions in selection_rate
    st.markdown("<h2 style='text-align: center; color: black;'>Disparity in predictions</h2>", unsafe_allow_html=True)

    unmitigated_sr = MetricFrame(selection_rate, y_true, unmitigated_predictions, sensitive_features=sensitive_features)
    mitigated_sr = MetricFrame(selection_rate, y_true, mitigated_predictions, sensitive_features=sensitive_features)

    unmitigated_sr_sensitive_feature_dict = unmitigated_sr.by_group.to_dict()
    mitigated_sr_sensitive_feature_dict = mitigated_sr.by_group.to_dict()

    unmitigated_y = ["{} : {}%".format(name, str(round(value * 100, 2))) for name, value in unmitigated_sr_sensitive_feature_dict.items()]
    mitigated_y = ["{} : {}%".format(name, str(round(value * 100, 2))) for name, value in mitigated_sr_sensitive_feature_dict.items()]

    unmitigated_sr_overall = str(round(unmitigated_sr.overall * 100, 2))
    mitigated_sr_overall = str(round(mitigated_sr.overall * 100, 2))

    unmitigated_disparity_in_sr = str(round(unmitigated_sr.difference() * 100, 2))
    mitigated_disparity_in_sr = str(round(mitigated_sr.difference() * 100, 2))

    unmitigated_sr_x = [value * 100 for value in unmitigated_sr_sensitive_feature_dict.values()]
    mitigated_sr_x = [value * 100 for value in mitigated_sr_sensitive_feature_dict.values()]


    unmitigated_sr_x_text = [str(round(value, 2)) + "%" for value in unmitigated_sr_x]
    mitigated_sr_x_text = [str(round(value, 2)) + "%" for value in mitigated_sr_x]

    st.markdown("<h3 style='text-align: left; color: black;'>After mitigation:Overall selection rate: {}%  |  Disparity in selection rate: {}% </h3>".format(mitigated_sr_overall, mitigated_disparity_in_sr), unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: black;'>Before mitigation:Overall selection rate: {}%  |  Disparity in selection rate: {}% </h3>".format(unmitigated_sr_overall, unmitigated_disparity_in_sr), unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Mitigated Model", "Unmitigated Model"))
    fig.add_trace(go.Bar(y=mitigated_y,
                         x=mitigated_sr_x,
                         orientation="h",
                         marker=dict(color="#0099C6"),
                         text=mitigated_sr_x_text,
                         textposition='inside'
                         ), row =1, col=1)
    fig.add_trace(go.Bar(y=unmitigated_y,
                         x=unmitigated_sr_x,
                         orientation="h",
                         marker=dict(color="#1F77B4"),
                         text=unmitigated_sr_x_text,
                         textposition='inside'
                         ), row=2, col=1)
    fig.update_layout(barmode="relative",
                      xaxis=dict(ticksuffix="%"),
                      width=800,
                      height=500,
                      showlegend=False
                      )
    st.plotly_chart(fig)



def subplots_in_metric(metric, metric_name, y_true, unmitigated_predictions, mitigated_predictions, sensitive_features, text):
    st.markdown("<h2 style='text-align: center; color: black;'>{}</h2>".format(text),
                                    unsafe_allow_html=True)
    unmitigated_metric_frame = MetricFrame(metrics=metric,
                                           y_true=y_true,
                                           y_pred=unmitigated_predictions,
                                           sensitive_features=sensitive_features)
    unmitigated_metric_frame_dict = dict(unmitigated_metric_frame.by_group)
    print('metric frame dict: ', unmitigated_metric_frame_dict)
    unmitigated_x = list(unmitigated_metric_frame_dict.keys())
    print(unmitigated_x)
    unmitigated_y = list(unmitigated_metric_frame_dict.values())
    print(unmitigated_y)
    unmitigated_y_text = [str(round(value * 100, 2)) + "%" if metric_name != 'count' else value for value in unmitigated_y]
    print(unmitigated_y_text)
    mitigated_metric_frame = MetricFrame(metrics=metric,
                                         y_true=y_true,
                                         y_pred=mitigated_predictions,
                                         sensitive_features=sensitive_features)
    mitigated_metric_frame_dict = dict(mitigated_metric_frame.by_group)
    print('metric frame dict: ', mitigated_metric_frame_dict)
    mitigated_x = list(mitigated_metric_frame_dict.keys())
    print(mitigated_x)
    mitigated_y = list(mitigated_metric_frame_dict.values())
    print(mitigated_y)
    mitigated_y_text = [str(round(value * 100, 2)) + "%" if metric_name != 'count' else value for value in mitigated_y]
    print(mitigated_y_text)

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.025, shared_yaxes=True, subplot_titles=("Mitigated Model", "Unmitigated Model"))
    fig.add_trace(go.Bar(y=mitigated_y,
                         x=mitigated_x,
                         orientation="v",
                         marker=dict(color="#0099C6"),
                         text=mitigated_y_text,
                         textposition='inside',
                         ), row=1, col=1)
    fig.add_trace(go.Bar(y=unmitigated_y,
                         x=unmitigated_x,
                         orientation="v",
                         marker=dict(color="#1F77B4"),
                         text=unmitigated_y_text,
                         textposition='inside'
                         ), row=1, col=2)
    fig.update_layout(barmode='relative',
                      width=600,
                      height=550,
                      showlegend=False,
                      yaxis_tickformat="%" if metric_name != 'count' else "",
                      )
    st.plotly_chart(fig)


def model_comparison_plot(metrics_mitigated,accuracy_mitigated,metrics_unmitigated,accuracy_unmitigated,disparity_metric):
    st.markdown("<h2 style='text-align: center; color: black;'>Model Comparison</h2>", unsafe_allow_html=True)
    fig = go.Figure()

    min_dif_mitigated = 100
    max_dif_mitigated = 0

    most_accurate_model_accuracy = accuracy_unmitigated
    most_accurate_model_disparity = metrics_unmitigated.difference()

    lowest_disparity_model_accuracy = accuracy_unmitigated
    lowest_disparity_model_disparity = metrics_unmitigated.difference()

    for i in range(len(metrics_mitigated)):
        fig.add_trace(go.Scatter(
            y=[100 * round(metrics_mitigated[i].difference() ,3)],
            x=[100 * round(accuracy_mitigated[i] ,3)],
            name="dominant_model_{0}".format(i)
        ))
        if (metrics_mitigated[i].difference() < min_dif_mitigated):
            min_dif_mitigated = metrics_mitigated[i].difference()
        if (metrics_mitigated[i].difference() > max_dif_mitigated):
            max_dif_mitigated = metrics_mitigated[i].difference()

        if accuracy_mitigated[i] > most_accurate_model_accuracy:
            most_accurate_model_accuracy = accuracy_mitigated[i]
            most_accurate_model_disparity = metrics_mitigated[i].difference()

        if metrics_mitigated[i].difference() < lowest_disparity_model_disparity:
            lowest_disparity_model_accuracy = accuracy_mitigated[i]
            lowest_disparity_model_disparity = metrics_mitigated[i].difference()

    fig.add_trace(go.Scatter(
        y=[100 * round(metrics_unmitigated.difference() ,3)],
        x=[100 * round(accuracy_unmitigated ,3)],
        name='unmitigated model',
        marker=dict(color="Black")

    ))
    fig.update_layout(xaxis_title='Accuracy', yaxis_title=disparity_metric)
    fig.update_layout(barmode="relative",
                      xaxis=dict(ticksuffix="%"),
                      yaxis=dict(ticksuffix="%"),
                      width=800,
                      height=600,
                      margin_t = 200,
                      title={
                          'text': "<sup>This chart represents each of the {} models as a selectable point."
                                  "The x-axis represents accuracy, with higher being better."
                                  "<br>The y-axis represents disparity, with lower being better."
                                  "<br><b>INSIGHTS: </b>"
                                  "Accuracy ranges from {}% to {}%. The disparity ranges from {}% to {}%"
                                  "<br>The most accurate model achieves accuracy of {}% and a disparity of {}%"
                                  "<br>The lowest-disparity model achieves accuracy of {}% and a disparity of {}% </sup>".format(
                                    len(metrics_mitigated)+1,
                                    round(100 * min(min(accuracy_mitigated),accuracy_unmitigated),2),
                                    round(100 * max(max(accuracy_mitigated),accuracy_unmitigated),2),
                                    round(100 * min(min_dif_mitigated,metrics_unmitigated.difference()),2),
                                    round(100 * max(max_dif_mitigated,metrics_unmitigated.difference()),2),
                                    round(100 * most_accurate_model_accuracy, 2),
                                    round(100 * most_accurate_model_disparity, 2),
                                    round(100 * lowest_disparity_model_accuracy,2),
                                    round(100 * lowest_disparity_model_disparity, 2)),
                          'font': {'color': 'black',
                                   'size': 18},
                          'xanchor': 'auto',
                          'yanchor': 'auto',
                          })
    fig.update_traces(marker_size=15)
    st.plotly_chart(fig)