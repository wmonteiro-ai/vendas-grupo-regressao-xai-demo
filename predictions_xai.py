import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

##########
#Imports
##########
import streamlit as st
import pandas as pd
import numpy as np
import pickle

import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime
from evalml.model_understanding.prediction_explanations import explain_predictions

##########
#Configuração do Streamlit
##########
st.set_page_config(page_title='Inteledge - Simulador de Crédito', page_icon="💡", layout="centered", initial_sidebar_state="auto", menu_items=None)

##########
#Funções para as previsões e para a organização da pãgina
##########
@st.cache(allow_output_mutation=True)
def get_pickles():
	_, logical_types = pickle.load(open('sample.pkl', 'rb'))
	pipeline, expected_value, pdp, pdp_relationship = pickle.load(open('model.pkl', 'rb'))
	
	return pipeline, logical_types, expected_value, pdp, pdp_relationship

@st.cache(allow_output_mutation=True)
def get_samples(target):
	# carregando uma amostra da base de dados
	df = pickle.load(open('df_resampled.pkl', 'rb'))
	columns = [target] + df.drop(target, axis=1).columns.tolist()

	# carregando as predições
	df[target] = best_pipeline.predict(df.drop(target, axis=1))
	df = df.replace({target: {1: 'Aprovado', 0: 'Reprovado'}})
	
	df_negados = df[df[target]=='Reprovado'].tail(5).reset_index(drop=True)
	df_negados = df_negados[columns]

	df_aprovados = df[df[target]=='Aprovado'].tail(5).reset_index(drop=True)
	df_aprovados = df_aprovados[columns]
	
	return df, df_negados, df_aprovados

@st.cache(allow_output_mutation=True)
def get_global_explanations(pdp, pdp_relationship):
	figures = []
	
	# Por UF
	fig = px.bar(pdp['UF'].sort_values(by='partial_dependence', ascending=False),
				x='feature_values', y='partial_dependence',
				color_discrete_sequence=[px.colors.qualitative.Plotly[3]],
				template='plotly_white',
				labels={
					"partial_dependence": "Alteração na predição",
					"feature_values": 'UF'
				})
	fig.update_layout(title=f'Influência dos diferentes valores de <em>UF</em> na previsão')
	figures.append(fig)
	
	# Por variáveis numéricas
	keys = ['Inscritos', 'Isentos', 'Confirmados', 'Dia do Pedido', 'Mês do Pedido']
	for i in range(len(keys)):
		fig = px.line(pdp[keys[i]], x='feature_values', y='partial_dependence',
					color_discrete_sequence=[px.colors.qualitative.Plotly[i]], template='plotly_white',
					labels={
						"partial_dependence": "Alteração na predição",
						"feature_values": f"Valor de {keys[i]}"
					})
		fig.update_layout(title=f'Influência dos diferentes valores de <em>{keys[i]}</em> na previsão')
		figures.append(fig)
	
	# Por 2 variáveis
	for key in [('Inscritos', 'Confirmados'), ('Inscritos', 'Isentos'), ('Confirmados', 'Isentos')]:
		def get_values(df):
			y = df.columns
			x = df.index
			z = df.values
			return x, y, z
		
		labels = key
		try:
			x, y, z = get_values(pdp_relationship[key])
		except:
			x, y, z = get_values(pdp_relationship[(key[1], key[0])])
		    
		fig = go.Figure(data=go.Contour(z=z, x=x, y=y, line_smoothing=0.85))
		fig.update_layout(title=f'Influência dos diferentes valores de <em>{labels[0]}</em> e <em>{labels[1]}</em> na previsão',
		                  xaxis_title=labels[0], yaxis_title=labels[1])
		figures.append(fig)
	
	return figures

def plot_importances(best_pipeline, df):
	# predictions
	pred = max(0, int(best_pipeline.predict(df).values[0]))

	df_plot = explain_predictions(pipeline=best_pipeline, input_features=df.reset_index(drop=True),
							y=None, top_k_features=len(df.columns), indices_to_explain=[0],
							include_explainer_values=True, output_format='dataframe')

	df_plot = df_plot.sort_values(by='quantitative_explanation')
	df_plot['Soma'] = expected_value+df_plot['quantitative_explanation'].cumsum()
	df_plot = df_plot.sort_values(by='quantitative_explanation')
	df_plot['Influencia para este resultado?'] = df_plot['quantitative_explanation']<0
	df_plot = df_plot.round(2)

	col_names = []
	for col in df_plot['feature_names'].values:
		if col == 'Data do Pedido':
			col_names.append(f'{col}<br><em>({df[col].astype(str).values[0]})</em>')
		elif col == 'Empresa':
			nome_empresa = df[col].values[0]
			nome_empresa = nome_empresa if len(nome_empresa) < 25 else nome_empresa[:22] + '...'
			col_names.append(f'{col}<br><em>({nome_empresa})</em>')
		else:
			col_names.append(f'{col}<br><em>({df[col].values[0]})</em>')

	fig_xai = go.Figure(go.Waterfall(
		name='Projeção',
		base=0,
		orientation="h",
		y=['Previsão inicial'] + col_names + ['Previsão final'],
		x=[expected_value] + df_plot['quantitative_explanation'].values.tolist() + [0],
		measure=['absolute'] + ['relative']*len(df_plot) + ['total'],
		text=[expected_value] + [f'{str(int(x))}' for x in df_plot['Soma'].values] + [pred],
		textposition = "outside",
		connector = {"line":{"color":"rgb(63, 63, 63)"}},
	))

	fig_xai.update_xaxes(range=[max(0, df_plot['Soma'].min()*0.7),
								max(expected_value, pred*1.3, df_plot['Soma'].max()*1.3)])
    
	fig_xai.update_layout(
		title=f'Principais influenciadores para o resultado:<br>(Previsão: <b>R$ {max(0, int(pred))}</b>)',
		showlegend=False,
		width=420,
		template='plotly',
		height=900
	)

	return fig_xai, pred

##########
#Preparando o simulador
##########
# carregando o modelo preditivo
best_pipeline, logical_types, expected_value, pdp, pdp_relationship = get_pickles()

# carregando uma amostra da base de dados
target = 'Total Pedido (R$)'
df, df_negados, df_aprovados = get_samples(target)

##########
#Seção 1 - Histõrico
##########
col1, _, _ = st.columns(3)
with col1:
	st.image('inteledge.png')

st.title('Simulador de Vendas')
st.markdown('Imagine o seguinte caso: você é um gerente de vendas de uma empresa que vende treinamentos e capacitações para outras empresas de todo o país. Esses treinamentos podem ser para empresas privadas ou estatais; podem ser treinamentos para pequenas ou grandes empresas; podem ter um grande número de pagantes ou de isentos. Uma coisa é oferecer um curso para uma empresa grande - outro, é garantir que você terá um grande número de **participantes**: afinal, é assim que você ganha o seu dinheiro.')
st.markdown('E agora, como prever comportamentos futuros? Como simular **quanto** você realmente receberá? É para isso que desenvolvemos uma Inteligência artificial para previsão de vendas. Imagine que isto pode ser aplicado também para outros tipos de vendas -- já imaginou as possibilidades para você? Ficou interessado em fazer algo parecido para o seu negócio? Entre em contato conosco no [@inteledge.lab](https://instagram.com/inteledge.lab) no Instagram!')
st.markdown('Confira também [algumas análises que fizemos em outra base de dados](https://share.streamlit.io/wmonteiro92/vendas-concessao-credito-analise-demo/main/exploration.py) e [outro algoritmo de IA que também criamos para você simular à vontade](https://share.streamlit.io/wmonteiro92/vendas-concessao-credito-xai-demo/main/predictions_xai.py)!')
##########
#Seção 2 - Simulador
##########
st.header('Teste você mesmo!')
st.markdown('Veja como seriam as previsões para novas oportunidades. Teste diferentes configurações e veja ao lado como o algoritmo chegou a esta conclusão para esta base de dados fictícia. As previsões atualizam **em tempo real**.')
col1, col2 = st.columns(2)

with col1:
	# variáveis 
	empresa = st.selectbox('Empresa',
		tuple(np.sort(df['Empresa'].unique())))

	uf = st.selectbox('UF',
		tuple(np.sort(df['UF'].unique())))
		 
	inscritos = st.slider('Número de inscritos',
		int(df['Inscritos'].min()), int(df['Inscritos'].max()),
		int(df['Inscritos'].median()))
		 
	isentos = st.slider('Número de isentos',
		int(df['Isentos'].min()), int(df['Isentos'].max()),
		int(df['Isentos'].median()))
		 
	confirmados = st.slider('Número de confirmados',
		int(df['Confirmados'].min()), int(df['Confirmados'].max()),
		int(df['Confirmados'].median()))

	data_pedido = st.date_input('Quando aconteceu o pedido?',
		datetime.now(), datetime.strptime('2021-01-01', '%Y-%m-%d'), datetime.now())

	mais_10k = st.checkbox('A empresa possui mais de 10 mil funcionários?')

	privada = st.checkbox('É empresa privada?')

	atua_todo_pais = st.checkbox('A empresa atua em todo o país?')

with col2:
	# inference
	df_inference = pd.DataFrame([[empresa, uf, inscritos, isentos, confirmados,
		mais_10k, privada, atua_todo_pais, data_pedido.month,
		data_pedido.day, data_pedido.year]],
		columns=['Empresa', 'UF', 'Inscritos', 'Isentos', 'Confirmados',
				'Mais de 10 mil funcionários?', 'Empresa Privada?', 'Atua em todo o país?',
				'Mês do Pedido', 'Dia do Pedido', 'Ano do Pedido'])
	df_inference = df_inference[logical_types.keys()]
	df_inference.ww.init()
	df_inference.ww.set_types(logical_types=logical_types)

	fig_xai, predicao = plot_importances(best_pipeline, df_inference)
	st.plotly_chart(fig_xai)

##########
#Seção 3 - Influenciadores
##########
st.header('Entendendo como o algoritmo aprendeu')
st.write('Do ponto de vista de Ciência de Dados você precisa saber **como** um algoritmo chegou a uma decisão. Naturalmente, nenhum algoritmo será 100% preciso. Por outro lado, será que ele está aprendendo corretamente as relações entre os dados? Será que há algum tipo de erro nos dados, ou alguma coisa não faz sentido? Veja abaixo os comportamentos aprendidos pela Inteligência Artificial a partir de uma base de dados de treinamento.')

for fig in get_global_explanations(pdp, pdp_relationship):
	st.plotly_chart(fig)
    
st.markdown('Instagram e Contato: [@inteledge.lab](https://instagram.com/inteledge.lab)')