import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import altair as alt

st.set_page_config(page_title="Análise de Churn", layout="wide")

# Carregamento dos dados
@st.cache_data
def load_data():
    df = pd.read_csv("src/Churn_Modelling.csv")
    df.rename(columns={'Exited': 'Churn'}, inplace=True)
    return df

df = load_data()

# Sidebar
st.sidebar.title("🔧 Navegação")
show_dados = st.sidebar.checkbox("Exibir dados", value=False)
show_eda = st.sidebar.checkbox("Exibir Análise Exploratória", value=False)
show_model = st.sidebar.checkbox("Exibir Modelagem com Regressão Logística", value=False)

# Observação extra sobre SMOTE
st.sidebar.markdown("---")
st.sidebar.markdown("📌 **Observação:**")
st.sidebar.info("Na etapa **Diagnóstico Final**, você poderá escolher se deseja aplicar SMOTE para balancear as classes. Essa opção será exibida somente ao ativar a modelagem.")


st.title("🔍 Painel de Churn – Retenção de Clientes do Banco")


# -----------------------------
# Função de validação de pressupostos
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from imblearn.over_sampling import SMOTE

def validar_pressupostos_logistica(X, y, model_pipeline):
    resultados = {}

    # 1. Balanceamento da variável dependente
    churn_counts = y.value_counts(normalize=True)
    resultados['balanceamento'] = churn_counts.to_dict()

    # 2. Verificar Multicolinearidade com VIF
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_scaled = StandardScaler().fit_transform(X_encoded)
    vif_data = pd.DataFrame()
    vif_data["Variável"] = X_encoded.columns
    vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    resultados['vif_table'] = vif_data

    #3. Linearidade do logit (Box-Tidwell simplificado)
    # Ajustar modelo logístico usando statsmodels
        # Codificar variável categórica
    preprocessor = ColumnTransformer(
        transformers=[('geo', OneHotEncoder(drop='first'), ['Geography'])],
        remainder='passthrough'
    )

    X_encoded = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names)

    # Adicionar constante e ajustar modelo logístico
    X_logit = sm.add_constant(X_encoded_df).reset_index(drop=True)
    y_alinhado = y.reset_index(drop=True)
    logit_model = sm.Logit(y_alinhado, X_logit)
    result = logit_model.fit()

    # Obter resíduos e valores ajustados
    residuos = result.resid_pearson
    fitted = result.fittedvalues

    # Gráfico: resíduos vs valores ajustados
   # Criar DataFrame com resíduos e valores ajustados
    df_residuos = pd.DataFrame({
        'Valores Ajustados': fitted,
        'Resíduos': residuos
    })

    # Gráfico interativo com Altair
    fig_linearidade = alt.Chart(df_residuos).mark_circle(size=60, opacity=0.5).encode(
        x=alt.X('Valores Ajustados', title='Valores Ajustados'),
        y=alt.Y('Resíduos', title='Resíduos'),
        tooltip=['Valores Ajustados', 'Resíduos']
    ).properties(
        title='Resíduos Pearson vs Logit Ajustado',
        width=600,
        height=400
    ).interactive()
      # linha de suavização (LOWESS)
    linha = fig_linearidade.transform_loess('Valores Ajustados', 'Resíduos').mark_line(color='red')
    fig_linearidade = fig_linearidade + linha

    resultados['fig_linearidade'] = fig_linearidade

    # 4. Verificar necessidade de SMOTE
    resultados['aplicar_smote'] = True if churn_counts.min() < 0.3 else False

    # 5. Verificar necessidade de regularização
    resultados['recomendar_regularizacao'] = True if vif_data['VIF'].max() > 5 else False

    return resultados


# ------------------------------
# diagnóstico de agrupamento

import pandas as pd
import numpy as np
import altair as alt

def diagnostico_agrupamentos(pipeline, X_test, y_test):
    # Pega os nomes das variáveis após transformação
    feature_names = pipeline.named_steps['prep'].get_feature_names_out()

    # Obtém X_test processado (OneHot + numéricas)
    X_encoded = pipeline.named_steps['prep'].transform(X_test)
    X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names).reset_index(drop=True)
    y_aligned = y_test.reset_index(drop=True)

    # Ajusta modelo com statsmodels (para obter logit e resíduos)
    import statsmodels.api as sm
    X_logit = sm.add_constant(X_encoded_df)
    logit_model = sm.Logit(y_aligned, X_logit).fit(disp=0)
    residuos = logit_model.resid_pearson
    fitted = logit_model.fittedvalues

    # Adiciona ao DataFrame
    X_encoded_df["logit"] = fitted
    X_encoded_df["residuos"] = residuos

    st.subheader("🔎 Análise de Agrupamentos por Variável Categórica")
    for col in X_encoded_df.columns:
        if col in ["logit", "residuos"]:
            continue
        if set(X_encoded_df[col].unique()) == {0, 1}:
            chart = alt.Chart(X_encoded_df).mark_circle(size=60, opacity=0.5).encode(
                x='logit',
                y='residuos',
                color=alt.Color(col, legend=alt.Legend(title=col)),
                tooltip=[col, 'logit', 'residuos']
            ).properties(
                width=600,
                height=300,
                title=f"Resíduos vs Logit (Agrupado por: {col})"
            )
            st.altair_chart(chart)

    st.subheader("📈 Linearidade dos Preditores Numéricos")
    variaveis_numericas = ['remainder__Age', 'remainder__Balance', 'remainder__NumOfProducts']

    for col in variaveis_numericas:
        scatter = alt.Chart(X_encoded_df).mark_circle(size=60, opacity=0.5).encode(
            x=alt.X(col, title=col.split("__")[-1]),
            y=alt.Y('residuos', title='Resíduos'),
            tooltip=['logit', col]
        ).properties(
            width=600,
            height=300,
            title=f"Resíduos vs {col.split('__')[-1]}"
        )

        linha = scatter.transform_loess(col, 'residuos').mark_line(color='red')

        st.altair_chart(scatter + linha)

    # Pega os nomes das variáveis após transformação
    feature_names = pipeline.named_steps['prep'].get_feature_names_out()

    # Obtém X_test processado (OneHot + numéricas)
    X_encoded = pipeline.named_steps['prep'].transform(X_test)
    X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names).reset_index(drop=True)
    y_aligned = y_test.reset_index(drop=True)

    ## Ajusta modelo com statsmodels (para obter logit e resíduos)
    #import statsmodels.api as sm
    #X_logit = sm.add_constant(X_encoded_df)
    #logit_model = sm.Logit(y_aligned, X_logit).fit(disp=0)
    #residuos = logit_model.resid_pearson
    #fitted = logit_model.fittedvalues

    # Adiciona ao DataFrame
    #X_encoded_df["logit"] = fitted
    #X_encoded_df["residuos"] = residuos

   # st.subheader("🔎 Análise de Agrupamentos por Variável")

    #for col in X_encoded_df.columns:
     #   if col in ["logit", "residuos"]:
      #      continue  # pula variáveis já usadas no gráfico
        # Verifica se é dummy
       ## if set(X_encoded_df[col].unique()) == {0, 1}:
         #   chart = alt.Chart(X_encoded_df).mark_circle(size=60, opacity=0.5).encode(
          #      x='logit',
          #      y='residuos',
          #      color=alt.Color(col, legend=alt.Legend(title=col)),
          #      tooltip=[col, 'logit', 'residuos']
          #  ).properties(
          #      width=600,
          #      height=300,
          #      title=f"Resíduos vs Logit colorido por: {col}"
           # )
           # st.altair_chart(chart)





# ------------------------------
# VISUALIZAÇÃO GERAL DOS DADOS
# ------------------------------
if show_dados:
    st.header("📊 Visualização dos Dados Gerais")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("---")

# ------------------------------
# ANÁLISE EXPLORATÓRIA
# ------------------------------
if show_eda:
    st.header("📈 Análise Exploratória de Churn")

    # Distribuição
    st.subheader("Distribuição da variável alvo (Churn)")
    fig_churn = px.histogram(df, x='Churn', color='Churn',
                             category_orders={'Churn': [0, 1]},
                             labels={'Churn': 'Cliente saiu'},
                             title='Distribuição de Clientes que Saíram (Churn)')
    st.plotly_chart(fig_churn, use_container_width=True)

    # Numéricas
    st.subheader("📊 Variáveis Numéricas vs Churn")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.plotly_chart(px.box(df, x='Churn', y='Age', color='Churn', title='Idade vs Churn'), use_container_width=True)
        st.plotly_chart(px.box(df, x='Churn', y='CreditScore', color='Churn', title='Score de Crédito vs Churn'), use_container_width=True)

    with col2:
        st.plotly_chart(px.box(df, x='Churn', y='Balance', color='Churn', title='Saldo vs Churn'), use_container_width=True)
        st.plotly_chart(px.box(df, x='Churn', y='EstimatedSalary', color='Churn', title='Salário Estimado vs Churn'), use_container_width=True)

    with col3:
        st.plotly_chart(px.box(df, x='Churn', y='Tenure', color='Churn', title='Tempo de Permanência vs Churn'), use_container_width=True)
        st.plotly_chart(px.box(df, x='Churn', y='NumOfProducts', color='Churn', title='Número de Produtos vs Churn'), use_container_width=True)

    # Categóricas
    st.subheader("📦 Variáveis Categóricas vs Churn")
    col4, col5, col6, col7 = st.columns(4)

    with col4:
        st.plotly_chart(px.histogram(df, x='NumOfProducts', color='Churn', barmode='group',
                                     title='Número de Produtos vs Churn'), use_container_width=True)

    with col5:
        st.plotly_chart(px.histogram(df, x='IsActiveMember', color='Churn', barmode='group',
                                     title='Cliente Ativo vs Churn',
                                     labels={'IsActiveMember': 'Ativo (1=Sim, 0=Não)'}), use_container_width=True)

    with col6:
        st.plotly_chart(px.histogram(df, x='Geography', color='Churn', barmode='group',
                                     title='Geografia vs Churn'), use_container_width=True)

    with col7:
        st.plotly_chart(px.histogram(df, x='Gender', color='Churn', barmode='group',
                                     title='Gênero vs Churn'), use_container_width=True)

    st.markdown("---")
    st.caption("Desenvolvido por Silvia Laryssa – Streamlit + Plotly 📊")
    
    
    
def treinar_modelo_logistico(df):
    # Seleção das variáveis
    X = df[['Age', 'Balance', 'IsActiveMember', 'NumOfProducts', 'Geography']]
    y = df['Churn']

    # Treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Pipeline
    preprocessor = ColumnTransformer(
        transformers=[('geo', OneHotEncoder(drop='first'), ['Geography'])],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('logreg', LogisticRegression(max_iter=1000))
    ])

    # Treinamento
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Avaliação no Streamlit
    st.subheader("📋 Avaliação do Modelo")
    cm = confusion_matrix(y_test, y_pred)
    st.write("**Matriz de Confusão:**")
    st.dataframe(pd.DataFrame(cm, columns=["Previsto 0", "Previsto 1"], index=["Real 0", "Real 1"]))

    st.markdown("**Relatório de Classificação:**")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    # Coeficientes e interpretação
    st.subheader("📈 Coeficientes do Modelo")
    feature_names = pipeline.named_steps['prep'].get_feature_names_out()
    coef = pipeline.named_steps['logreg'].coef_[0]
    odds = np.exp(coef)

    coef_table = pd.DataFrame({
        'Variável': feature_names,
        'Coeficiente': coef.round(3),
        'Odds Ratio': odds.round(3)
    }).sort_values(by='Coeficiente', ascending=False)

    st.dataframe(coef_table)

    return X_test, y_test, pipeline   

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import streamlit as st

def reajustar_modelo_logistico(X, y):
    """
    Ajusta um novo modelo de regressão logística com dados já codificados (X e y).

    Retorna:
    - pipeline treinado
    - y_pred (predições)
    - coef_table (tabela com coeficientes e odds)
    """

    # Pipeline apenas com scaler + regressão
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=1000))
    ])

    # Treina o modelo
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)

    # Avaliação do modelo
    st.subheader("📋 Avaliação do Modelo Ajustado com Dados Transformados")
    st.write("**Matriz de Confusão:**")
    cm = confusion_matrix(y, y_pred)
    st.dataframe(pd.DataFrame(cm, columns=["Previsto 0", "Previsto 1"], index=["Real 0", "Real 1"]))

    st.markdown("**Relatório de Classificação:**")
    report = classification_report(y, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    # Coeficientes
    coef = pipeline.named_steps['logreg'].coef_[0]
    odds = np.exp(coef)
    feature_names = X.columns if hasattr(X, 'columns') else [f"X{i}" for i in range(len(coef))]

    coef_table = pd.DataFrame({
        'Variável': feature_names,
        'Coeficiente': coef.round(3),
        'Odds Ratio': odds.round(3)
    }).sort_values(by='Coeficiente', ascending=False)

    st.subheader("📈 Coeficientes do Modelo")
    st.dataframe(coef_table)

    return pipeline, y_pred, coef_table
 
 
# #############################################
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, confusion_matrix
import plotly.graph_objects as go

def avaliar_modelo_classificacao(y_true, y_pred, y_prob):
    st.subheader("D) 📊 Avaliação Preditiva do Modelo")

    # Métricas principais
    acuracia = accuracy_score(y_true, y_pred)
    precisao = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    # Especificidade
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    especificidade = tn / (tn + fp)

    # Exibição das métricas
    st.markdown("### 🔢 Métricas de Desempenho")
    st.write(f"**🎯 Acurácia:** {acuracia:.2f}")
    st.write(f"**📌 Precisão:** {precisao:.2f}")
    st.write(f"**📈 Sensibilidade (Recall):** {recall:.2f}")
    st.write(f"**🛡️ Especificidade:** {especificidade:.2f}")
    st.write(f"**📉 AUC (Área sob a Curva ROC):** {auc:.2f}")

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='Curva ROC', line=dict(color='blue')))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aleatório', line=dict(color='gray', dash='dash')))
    fig_roc.update_layout(
        title='Curva ROC',
        xaxis_title='Taxa de Falsos Positivos (1 - Especificidade)',
        yaxis_title='Taxa de Verdadeiros Positivos (Sensibilidade)',
        width=700,
        height=500
    )
    st.plotly_chart(fig_roc, use_container_width=True)


############################################## 

# ------------------------------
# MODELAGEM COM REGRESSÃO LOGÍSTICA
# ------------------------------
if show_model:
    st.header("🧠 Questão A e C - Modelagem com Regressão Logística (Retenção de Clientes)")

    st.markdown("Selecionando variáveis explicativas mais relevantes:")
    st.code("Age, Balance, IsActiveMember, NumOfProducts, Geography")

    X_test, y_test, pipeline = treinar_modelo_logistico(df)


    # ------------------------------
    # VALIDAÇÃO DOS PRESSUPOSTOS DA REGRESSÃO LOGÍSTICA
    # ------------------------------
    st.header("B) 📏 Validação dos Pressupostos da Regressão Logística")

    val_result = validar_pressupostos_logistica(X_test, y_test, pipeline)
    diagnostico_agrupamentos(pipeline, X_test, y_test)

    # 1. Balanceamento
    st.subheader("📊 Balanceamento da variável Churn")
    st.write(val_result['balanceamento'])
    if min(val_result['balanceamento'].values()) < 0.3:
        st.warning("🔴 Churn desbalanceado – considere aplicar SMOTE ou técnicas de balanceamento.")

    # 2. Multicolinearidade
    st.subheader("📌 VIF – Multicolinearidade")
    st.dataframe(val_result['vif_table'])
    if val_result['vif_table']['VIF'].max() > 5:
        st.warning("⚠️ VIF elevado detectado – considere aplicar regularização L1 ou L2.")

    # 3. Linearidade do logit
    st.subheader("📈 Linearidade entre preditores e logit")
    st.altair_chart(val_result['fig_linearidade'], use_container_width=True) 

    # 4. Recomendações automáticas
    st.subheader("🧪 Diagnóstico Final")
    if val_result['aplicar_smote']:
        st.markdown("✅ **Sugestão:** Aplicar SMOTE para balancear as classes.")

    if val_result['recomendar_regularizacao']:
        st.markdown("✅ **Sugestão:** Aplicar regularização **L1 (Lasso)** ou **L2 (Ridge)** para mitigar multicolinearidade.")

    aplicar_smote_usuario = st.checkbox("Deseja aplicar SMOTE para balancear as classes?")
    if aplicar_smote_usuario:
        if min(val_result['balanceamento'].values()) < 0.3:
            st.info("🔄 Aplicando SMOTE após codificação dos dados...")

            # Transforma X_test com o mesmo pré-processador do pipeline
            X_test_encoded = pipeline.named_steps['prep'].transform(X_test)
            feature_names = pipeline.named_steps['prep'].get_feature_names_out()
            X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=feature_names)

            # Aplica SMOTE com os dados numéricos
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_test_encoded_df, y_test)

            st.success("✅ SMOTE aplicado com sucesso!")
            st.write("Distribuição após SMOTE:")
            st.dataframe(y_resampled.value_counts(normalize=True))

            # Atualize os valores usados no restante do app, se necessário
            st.markdown("🔄 Atualizando modelo com dados balanceados...")
            # Reajusta o modelo com os dados balanceados
            pipeline, y_pred, coef_table = reajustar_modelo_logistico(X_resampled, y_resampled)
            
            st.subheader("📈 Coeficientes do Modelo Ajustado com SMOTE")
            data_interpretativa = {
                "Variável": [
                    "geo__Geography_Germany",
                    "geo__Geography_Spain",
                    "remainder__Age",
                    "remainder__Balance",
                    "remainder__NumOfProducts",
                    "remainder__IsActiveMember"
                ],
                "Coeficiente": [
                    "+0.384",
                    "–0.096",
                    "+0.925",
                    "+0.165",
                    "–0.132",
                    "–0.552"
                ],
                "Odds Ratio": [
                    "1.468",
                    "0.908",
                    "2.523",
                    "1.179",
                    "0.877",
                    "0.576"
                ],
                "Impacto sobre o Churn (%)": [
                    "+46,8%",
                    "–9,2%",
                    "+152,3% por ano",
                    "+17,9%",
                    "–12,3% por produto",
                    "–42,4%"
                ],
                "Interpretação Detalhada": [
                    "Clientes da Alemanha têm 1,46x mais chance de churn do que clientes da França. Forte associação com saída.",
                    "Clientes da Espanha têm 9,2% menos chance de churn comparado à França. Leve proteção contra saída.",
                    "A cada ano a mais de idade, a chance de churn aumenta em 152,3%, sugerindo forte saída entre clientes mais velhos.",
                    "Saldo positivo tem impacto leve no aumento da chance de churn.",
                    "Cada produto bancário adicional reduz a chance de churn em 12,3%. Clientes com mais produtos são mais fiéis.",
                    "Clientes ativos têm 42,4% menos chance de churn. Fator importante de retenção no modelo."
                ]
            }
        df_interpretativa = pd.DataFrame(data_interpretativa)
        st.table(df_interpretativa)
        st.markdown("### 🧩 Conclusões Estratégicas da Análise de Churn ")
        st.markdown("""
        - **Clientes ativos** têm menor probabilidade de churn. Invista em ações para ativar clientes inativos.
        - **Número de produtos bancários** está negativamente relacionado ao churn. Estratégias de cross-selling podem ser eficazes.
        - **Clientes da Alemanha** apresentam maior risco e devem ser analisados regionalmente.
        - **Idade** é o fator mais crítico: quanto mais velho o cliente, maior a chance de churn. Personalize ofertas para esse público.
        - **Saldo bancário** mostrou influência leve, sugerindo que retenção depende mais do engajamento do que do volume financeiro.
        - **A geografia** tem impacto considerável. A Espanha demonstra maior fidelidade.
        """)

        # Avaliação do desempenho com dados balanceados (SMOTE)
        y_prob = pipeline.predict_proba(X_resampled)[:, 1]
        avaliar_modelo_classificacao(y_true=y_resampled, y_pred=y_pred, y_prob=y_prob)
        st.markdown("""
        ### 📊 Interpretação das Métricas de Desempenho

        - **Acurácia (0.72):** 72% das previsões totais foram corretas.  
        - **Precisão (0.72):** 72% dos clientes previstos como churn realmente saíram.  
        - **Sensibilidade/Recall (0.73):** O modelo identificou corretamente 73% dos clientes que saíram.  
        - **Especificidade (0.72):** Acertou 72% dos clientes que permaneceram.  
        - **AUC (0.78):** Boa capacidade geral de distinguir entre quem sai e quem fica.

        ✅ O modelo apresenta **desempenho equilibrado** e é adequado para **ações estratégicas de retenção**.
        """)  

        
    else:
                # Exemplo: salvar em sessão, ou atualizar os gráficos/predições
        #st.warning("⚠️ As classes já estão relativamente balanceadas. SMOTE não é necessário.")

            
    
        # dataframe com a interpretação dos coeficientes
        st.header("📊 Tabela Interpretativa dos Coeficientes antes do SMOTE")

        # Dados da tabela interpretativa
        data_interpretativa1 = {
            "Variável": [
                "geo__Geography_Germany",
                "geo__Geography_Spain",
                "remainder__Age",
                "remainder__Balance",
                "remainder__NumOfProducts",
                "remainder__IsActiveMember"
            ],
            "Coeficiente": [
                "+0.798",
                "+0.084",
                "+0.072",
                "0",
                "–0.075",
                "–1.059"
            ],
            "Odds Ratio": [
                "2.222",
                "1.088",
                "1.075",
                "1.000",
                "0.928",
                "0.347"
            ],
            "Impacto sobre o Churn (%)": [
                "+122,2%",
                "+8,8%",
                "+7,5% por ano",
                "0%",
                "–7,2% por produto",
                "–65,3%"
            ],
            "Interpretação Detalhada": [
                "Clientes da Alemanha têm 2,2x mais chance de churn do que clientes da França (grupo base). Forte associação com saída.",
                "Clientes da Espanha têm 8,8% mais chance de churn comparado à França. Efeito pequeno, quase neutro.",
                "A cada ano a mais de idade, a chance de churn aumenta em 7,5%, sugerindo maior saída entre clientes mais velhos.",
                "O saldo na conta não teve efeito significativo sobre o churn neste modelo.",
                "Cada produto bancário adicional reduz a chance de churn em 7,2%. Clientes com mais produtos são mais fiéis.",
                "Clientes ativos têm 65,3% menos chance de churn. É o fator mais forte de retenção no modelo."
            ]
        }

    
        df_interpretativa1 = pd.DataFrame(data_interpretativa1)

        # Exibir a tabela no Streamlit
        st.table(df_interpretativa1)

     

    st.markdown("---")
    st.caption("🔍 Coeficientes positivos aumentam a chance de churn; negativos indicam maior retenção.")
    st.markdown("> **Interpretação:** Coeficientes positivos aumentam a chance de churn; negativos diminuem. A coluna `Odds Ratio` mostra quanto a chance é multiplicada para cada unidade da variável.")
    
    st.markdown("""
        ---

        ### 📚 Referências
        - Field, A. (2009). *Descobrindo a estatística usando o SPSS*. 2. ed. Porto Alegre: Artmed, 2009
        - Grus, J. (2021). Data science do zero: noções fundamentais com Python (2ª ed.). Alta Books.

        ---

        ### Autores
        - **PPCA**: Programa de Computação Aplicada - UNB  
        - **AEDI**: Análise Estatística de Dados e Informações  
        - **Prof.** João Gabriel de Moraes Souza  
        - **Aluna**: Silva Laryssa Branco da Silva  
        - **Data**: 2025/07/18

        ### 🔗 Links

        - Projeto no HungginFace: [https://huggingface.co/spaces/silviabranco/regressaologistica](https://huggingface.co/spaces/silviabranco/regressaologistica)  
        - Projeto Community Cloud: [https://regressaologisticaaeidunb.streamlit.app/](https://regressaologisticaaeidunb.streamlit.app/)    
        - GITHub: [https://github.com/silvialaryssa/regressaologistica.git](https://github.com/silvialaryssa/regressaologistica)
      
        """)
