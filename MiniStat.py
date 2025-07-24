
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io
import os

def generate_pdf(summary_text, image_path=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in summary_text.split('\n'):
        pdf.multi_cell(0, 10, line)
    if image_path and os.path.exists(image_path):
        pdf.image(image_path, x=10, y=None, w=180)
    pdf_string = pdf.output(dest='S').encode('latin1')
    buffer = io.BytesIO(pdf_string)
    return buffer

def plot_imr_chart(data):
    df = pd.DataFrame({'X': data.dropna()})
    df['MR'] = df['X'].diff().abs()

    mean_x = df['X'].mean()
    std_x = df['X'].std()
    UCL_I = mean_x + 3 * std_x
    LCL_I = mean_x - 3 * std_x
    ooc_i = df[(df['X'] > UCL_I) | (df['X'] < LCL_I)]

    mean_mr = df['MR'].mean()
    UCL_MR = mean_mr * 3.267
    ooc_mr = df[df['MR'] > UCL_MR]

    def detect_trend(series, window=6):
        last = series[-window:]
        if all(np.diff(last) > 0):
            return "📈 Opwaartse trend gedetecteerd"
        elif all(np.diff(last) < 0):
            return "📉 Neerwaartse trend gedetecteerd"
        else:
            return "Geen duidelijke trend in de laatste waarnemingen"

    trend_i = detect_trend(df['X'])
    trend_mr = detect_trend(df['MR'].dropna())

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(df['X'], marker='o', label='Waarnemingen')
    axes[0].axhline(mean_x, color='green', linestyle='--', label=f'Gemiddelde = {mean_x:.2f}')
    axes[0].axhline(UCL_I, color='red', linestyle='--', label='UCL')
    axes[0].axhline(LCL_I, color='red', linestyle='--', label='LCL')
    axes[0].scatter(ooc_i.index, ooc_i['X'], color='red', marker='X', s=100, label='OOC punten')
    axes[0].set_title("I-chart")
    axes[0].legend()

    axes[1].plot(df['MR'], marker='o', label='Moving Range')
    axes[1].axhline(mean_mr, color='green', linestyle='--', label=f'Gemiddelde MR = {mean_mr:.2f}')
    axes[1].axhline(UCL_MR, color='red', linestyle='--', label='UCL')
    axes[1].scatter(ooc_mr.index, ooc_mr['MR'], color='red', marker='X', s=100, label='OOC MR punten')
    axes[1].set_title("MR-chart")
    axes[1].legend()

    plt.figtext(0.5, 0.01, f"I-chart: {trend_i} | MR-chart: {trend_mr}", ha="center", fontsize=10)
    fig.tight_layout()
    st.pyplot(fig)
    return fig

st.set_page_config(layout="wide")
st.title("📊 MiniStat – Statistische Analysetool (Minitab-stijl)")

uploaded_file = st.file_uploader("📁 Upload je gegevensbestand", type=["csv", "xls", "xlsx"])
view_option = st.radio("👁️ Weergaveoptie", ["📋 Alleen datatabel", "📈 Alleen resultaten", "📊 Beide (resultaten + data)"])

chart_path = None

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("❌ Bestandstype niet ondersteund.")
        st.stop()

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(include='object').columns.tolist()

    st.sidebar.header("📌 Kies Analyse")
    analysis_type = st.sidebar.radio(
        "Selecteer analyse:",
        ("Beschrijvende statistiek", "One-sample T-test", "Two-sample T-test", "Lineaire regressie", "ANOVA", "I-MR Control Chart")
    )

    summary_report = ""

    with st.container():
        if view_option in ["📈 Alleen resultaten", "📊 Beide (resultaten + data)"]:
            st.subheader("🔎 Analyse resultaten")

            if analysis_type == "Beschrijvende statistiek":
                col = st.selectbox("Kies kolom", numeric_columns)
                desc = df[col].describe()
                st.write(desc)
                summary_report += f"Beschrijvende statistiek voor {col}:\n{desc.to_string()}\n"

            elif analysis_type == "One-sample T-test":
                col = st.selectbox("Kies kolom", numeric_columns)
                mu = st.number_input("Verwachte gemiddelde waarde", value=0.0)
                t_stat, p_val = stats.ttest_1samp(df[col].dropna(), mu)
                st.write(f"t = {t_stat:.3f}, p = {p_val:.4f}")
                summary_report += f"One-sample T-test voor {col} (mu={mu}): t = {t_stat:.3f}, p = {p_val:.4f}\n"

            elif analysis_type == "Two-sample T-test":
                col = st.selectbox("Kies numerieke kolom", numeric_columns)
                group_col = st.selectbox("Kies categorische kolom", categorical_columns)
                groups = df[group_col].dropna().unique()
                if len(groups) == 2:
                    g1 = df[df[group_col] == groups[0]][col].dropna()
                    g2 = df[df[group_col] == groups[1]][col].dropna()
                    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
                    st.write(f"{groups[0]} vs {groups[1]}: t = {t_stat:.3f}, p = {p_val:.4f}")
                    summary_report += f"Two-sample T-test tussen {groups[0]} en {groups[1]} op {col}: t = {t_stat:.3f}, p = {p_val:.4f}\n"
                else:
                    st.warning("⚠️ Kies een kolom met exact 2 groepen.")

            elif analysis_type == "Lineaire regressie":
                y_col = st.selectbox("Y (afhankelijk)", numeric_columns)
                x_col = st.selectbox("X (onafhankelijk)", [col for col in numeric_columns if col != y_col])
                X = sm.add_constant(df[x_col].dropna())
                y = df[y_col].dropna()
                model = sm.OLS(y.loc[X.index], X).fit()
                st.write(model.summary())
                summary_report += f"Regressie Y={y_col}, X={x_col}:\n{model.summary()}\n"
                fig, ax = plt.subplots()
                sns.regplot(x=df[x_col], y=df[y_col], ax=ax)
                ax.set_title("Regressieplot")
                st.pyplot(fig)
                chart_path = "regression_plot.png"
                fig.savefig(chart_path)

            elif analysis_type == "ANOVA":
                dep = st.selectbox("Afhankelijke variabele", numeric_columns)
                group = st.selectbox("Groepsvariabele", categorical_columns)
                model = sm.formula.ols(f"{dep} ~ C({group})", data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                st.write(anova_table)
                summary_report += f"ANOVA voor {dep} op {group}:\n{anova_table.to_string()}\n"

            elif analysis_type == "I-MR Control Chart":
                col = st.selectbox("Kolom voor controlekaart", numeric_columns)
                st.write("Controlekaart:")
                fig = plot_imr_chart(df[col])
                summary_report += f"I-MR controlekaart gegenereerd voor {col}.\n"
                chart_path = "imr_chart.png"
                fig.savefig(chart_path)

        if view_option in ["📋 Alleen datatabel", "📊 Beide (resultaten + data)"]:
            st.subheader("🧾 Dataweergave")
            st.dataframe(df)

        if summary_report and view_option != "📋 Alleen datatabel":
            pdf_data = generate_pdf(summary_report, image_path=chart_path)
            st.download_button(
                label="📄 Download rapport als PDF (inclusief grafiek indien van toepassing)",
                data=pdf_data,
                file_name="rapport_minitstat.pdf",
                mime="application/pdf"
            )
