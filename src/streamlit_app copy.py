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

st.title("🔍 Painel de Churn – Retenção de Clientes do Banco")



def avaliar_violacoes_e_sugestoes(shapiro_p, bp_pvalue, vif_table, figs_linearidade=None):
    """
    Avalia violações de pressupostos e sugere alternativas.
    Parâmetros:
        - shapiro_p: p-valor do teste de Shapiro-Wilk (normalidade dos resíduos)
        - bp_pvalue: p-valor do teste de Breusch-Pagan (homocedasticidade)
        - vif_table: DataFrame com VIFs das variáveis
        - figs_linearidade: Lista de gráficos de resíduos vs preditores
    Retorna:
        Texto com interpretações e sugestões.
    """
    sugestoes = []

    # 1. Normalidade dos resíduos
    if shapiro_p < 0.05:
        sugestoes.append("❌ **Normalidade dos resíduos violada** (p-valor do Shapiro-Wilk < 0.05).\
        \n➡️ Sugestão: Aplicar transformações (ex: log, Box-Cox) ou utilizar **Regressão Quantílica**.")

    # 2. Homocedasticidade
    if bp_pvalue < 0.05:
        sugestoes.append("❌ **Heterocedasticidade detectada** (p-valor do teste de Breusch-Pagan < 0.05).\
        \n➡️ Sugestão: Utilizar **Regressão Robusta** (ex: `HuberRegressor`, `RLM` do `statsmodels`).")

    # 3. Multicolinearidade
    variaveis_com_vif_alto = vif_table[vif_table["VIF"] > 5]["Variável"].tolist()
    if len(variaveis_com_vif_alto) > 0:
        sugestoes.append(f"❌ **Multicolinearidade identificada nas variáveis:** {', '.join(variaveis_com_vif_alto)}.\
        \n➡️ Sugestão: Utilizar **modelos com regularização** como **Ridge**, **Lasso** ou **ElasticNet**.")

    # 4. Linearidade (avaliação visual)
    if figs_linearidade:
        sugestoes.append("⚠️ **Verifique os gráficos de resíduos vs preditores.**\
        \nSe houver padrões não aleatórios, pode haver violação da suposição de **linearidade**.\
        \n➡️ Sugestão: Utilizar **Regressão Polinomial** ou **modelos baseados em árvore** (ex: `Random Forest`).")

    if not sugestoes:
        return "✅ Todos os pressupostos parecem estar atendidos. O modelo de Regressão Logística é apropriado."

    return "\n\n".join(sugestoes)




# -----------------------------
# Função de validação de pressupostos
# -----------------------------


def validar_pressupostos_completo(X, y, modelo_pipeline):
    """
    Valida pressupostos da regressão logística:
        1. Normalidade dos resíduos (Q-Q plot + Shapiro-Wilk)
        2. Homocedasticidade (resíduos vs valores previstos + Breusch-Pagan)
        3. Multicolinearidade (VIF)
        4. Linearidade (resíduos vs cada preditor já transformado)

    Parâmetros
    ----------
    X : pd.DataFrame ou np.ndarray
        Dados de teste (features) antes da transformação do pipeline.
    y : pd.Series ou np.ndarray
        Valores reais (target) correspondentes a X.
    modelo_pipeline : sklearn.pipeline.Pipeline
        Pipeline já treinado contendo:
            - step 'prep' : transformador (por ex. ColumnTransformer)
            - step 'logreg' : modelo LogisticRegression

    Retorno
    -------
    dict com:
        - fig_qqplot                : Matplotlib Figure
        - shapiro_p                 : float
        - fig_homocedasticidade     : Matplotlib Figure
        - bp_pvalue                 : float
        - vif_table                 : pd.DataFrame
        - figs_linearidade          : list[Matplotlib Figure]
    """

    # ----- 0. Preparação -----
    # X transformado pelo pré-processador
    X_transf = modelo_pipeline.named_steps['prep'].transform(X)
    # Probabilidade prevista (classe positiva)
    y_pred_prob = modelo_pipeline.named_steps['logreg'].predict_proba(X_transf)[:, 1]
    # Resíduos (observado − previsto)
    residuos = y - y_pred_prob

    # ----- 1. Normalidade dos resíduos -----
    fig_qqplot, ax_qq = plt.subplots()
    sm.qqplot(residuos, line='45', ax=ax_qq)
    ax_qq.set_title('Q-Q Plot dos Resíduos')

    # Teste de Shapiro-Wilk
    shapiro_stat, shapiro_p = shapiro(residuos)

    # ----- 2. Homocedasticidade -----
    fig_homo, ax_homo = plt.subplots()
    ax_homo.scatter(y_pred_prob, residuos, alpha=0.5)
    ax_homo.axhline(y=0, color='r', linestyle='--')
    ax_homo.set_xlabel("Valores previstos (probabilidades)")
    ax_homo.set_ylabel("Resíduos")
    ax_homo.set_title("Resíduos vs Valores Previstos")

    # Breusch-Pagan
    # Precisa de matriz explicativa com intercepto
    exog_bp = sm.add_constant(X_transf, prepend=False)
    _, bp_pvalue, _, _ = het_breuschpagan(residuos, exog_bp)

    # ----- 3. Multicolinearidade (VIF) -----
    feature_names = modelo_pipeline.named_steps['prep'].get_feature_names_out()
    vif_data = pd.DataFrame({
        "Variável": feature_names,
        "VIF": [variance_inflation_factor(X_transf, i)
                for i in range(X_transf.shape[1])]
    }).round(2)

    # ----- 4. Linearidade (resíduos vs preditores) -----
    figs_linearidade = []
    for idx, fname in enumerate(feature_names):
        fig_lin, ax_lin = plt.subplots()
        ax_lin.scatter(X_transf[:, idx], residuos, alpha=0.5)
        ax_lin.axhline(y=0, color='r', linestyle='--')
        ax_lin.set_title(f"Resíduos vs {fname}")
        ax_lin.set_xlabel(fname)
        ax_lin.set_ylabel("Resíduos")
        figs_linearidade.append(fig_lin)

    # ----- 5. Retorno -----
    resultados = {
        "fig_qqplot": fig_qqplot,
        "shapiro_p": shapiro_p,
        "fig_homocedasticidade": fig_homo,
        "bp_pvalue": bp_pvalue,
        "vif_table": vif_data,
        "figs_linearidade": figs_linearidade
    }

    return resultados




# ------------------------------
# VISUALIZAÇÃO GERAL DOS DADOS
# ------------------------------
if show_dados:
    st.header("📊 Visualização dos Dados Gerais")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("---")
    st.caption("Desenvolvido por Silvia Laryssa – Streamlit + Plotly 📊")

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

# ------------------------------
# MODELAGEM COM REGRESSÃO LOGÍSTICA
# ------------------------------
if show_model:
    st.header("🧠 Questão A e C - Modelagem com Regressão Logística (Retenção de Clientes)")

    st.markdown("Selecionando variáveis explicativas mais relevantes:")
    st.code("Age, Balance, IsActiveMember, NumOfProducts, Geography")

    X = df[['Age', 'Balance', 'IsActiveMember', 'NumOfProducts', 'Geography']]
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[('geo', OneHotEncoder(drop='first'), ['Geography'])],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('logreg', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    st.subheader("📋 Avaliação do Modelo")
    cm = confusion_matrix(y_test, y_pred)
    st.write("**Matriz de Confusão:**")
    st.dataframe(pd.DataFrame(cm, columns=["Previsto 0", "Previsto 1"], index=["Real 0", "Real 1"]))

    st.markdown("**Relatório de Classificação:**")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

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

    # ------------------------------
    # VALIDAÇÃO DOS PRESSUPOSTOS
    # ------------------------------
    st.header("🔎 Validação dos Pressupostos da Regressão Logística")

    resultados = validar_pressupostos_completo(X_test, y_test, pipeline)

    # 1. Normalidade dos Resíduos
    st.subheader("1. Normalidade dos Resíduos")
    st.pyplot(resultados["fig_qqplot"])
    st.markdown(f"**Teste de Shapiro-Wilk – p-valor:** `{resultados['shapiro_p']:.4f}`")
    if resultados["shapiro_p"] < 0.05:
        st.warning("A normalidade dos resíduos foi violada (p < 0.05).")

    # 2. Homocedasticidade
    st.subheader("2. Homocedasticidade")
    st.pyplot(resultados["fig_homocedasticidade"])
    st.markdown(f"**Teste de Breusch-Pagan – p-valor:** `{resultados['bp_pvalue']:.4f}`")
    if resultados["bp_pvalue"] < 0.05:
        st.warning("Heterocedasticidade detectada (p < 0.05).")

    # 3. Multicolinearidade (VIF)
    st.subheader("3. Multicolinearidade (VIF)")
    st.dataframe(resultados["vif_table"])
    if resultados["vif_table"]["VIF"].max() > 5:
        st.warning("Algumas variáveis apresentam VIF > 5, indicando multicolinearidade.")

    # 4. Linearidade
    st.subheader("4. Linearidade – Resíduos vs Preditores")
    st.markdown("Verifique visualmente se há padrões sistemáticos. Idealmente, os pontos devem estar distribuídos aleatoriamente.")
    for fig in resultados["figs_linearidade"]:
        st.pyplot(fig)

    # 5. Sugestões baseadas nas violações
    st.subheader("🧪 Diagnóstico e Sugestões com base nas Violações")
    sugestoes_texto = avaliar_violacoes_e_sugestoes(
        shapiro_p=resultados["shapiro_p"],
        bp_pvalue=resultados["bp_pvalue"],
        vif_table=resultados["vif_table"],
        figs_linearidade=resultados["figs_linearidade"]
    )
    st.markdown(sugestoes_texto)


    
    # dataframe com a interpretação dos coeficientes
    st.header("📊 Tabela Interpretativa dos Coeficientes")

    # Dados da tabela interpretativa
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

    st.markdown("### 🧩 Conclusões Estratégicas da Análise de Churn")

    st.markdown("""
    - **Clientes ativos** têm muito menos chance de churn. É essencial investir em estratégias para engajar clientes inativos (como campanhas de reativação e uso do app).
    - **Mais produtos bancários** reduzem o churn, sugerindo que ações de venda cruzada (cross-selling) aumentam a fidelização.
    - **Clientes da Alemanha** apresentam maior risco de saída, exigindo intervenções regionais e análise de causas locais (como concorrência e oferta de serviços).
    - **A idade** está positivamente associada ao churn. É necessário adaptar serviços e comunicação para faixas etárias mais velhas.
    - **Saldo bancário** não influencia significativamente o churn, indicando que a retenção depende mais do relacionamento do que do volume financeiro.
    - Esses achados apoiam **campanhas personalizadas**, **melhorias específicas por região** e **ajustes no portfólio de produtos**.
    - O modelo permite tomadas de decisão mais **assertivas e direcionadas**, contribuindo fortemente para a **estratégia de retenção**.
    """)


    df_interpretativa = pd.DataFrame(data_interpretativa)

    # Exibir a tabela no Streamlit
    st.table(df_interpretativa)

    st.markdown("---")
    st.caption("🔍 Coeficientes positivos aumentam a chance de churn; negativos indicam maior retenção.")

    
   

    st.markdown("> **Interpretação:** Coeficientes positivos aumentam a chance de churn; negativos diminuem. A coluna `Odds Ratio` mostra quanto a chance é multiplicada para cada unidade da variável.")
