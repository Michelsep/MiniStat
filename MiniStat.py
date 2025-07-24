import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from fpdf import FPDF
import base64
import os

st.set_page_config(page_title="MiniStat", layout="wide")

def generate_pdf_report(text, chart_path=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.cell(200, 10, txt=line, ln=True)
    if chart_path and os.path.exists(chart_path):
        pdf.image(chart_path, x=10, y=None, w=180)
    pdf_output = pdf.output(dest="S").encode("latin-1")
    return pdf_output

st.title("📊 MiniStat - Statistische Analyse Tool")

uploaded_file = st.file_uploader("Upload een CSV of Excel-bestand", type=["csv", "xls", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Fout bij het laden van het bestand: {e}")
        st.stop()

    st.subheader("Geüploade Data")
    st.dataframe(df)

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    all_columns = df.columns.tolist()

    analysis_type = st.selectbox("Kies analyse", [
        "Beschrijvende statistiek",
        "Regressieanalyse",
        "ANOVA",
        "T-toets (2-sample)",
        "I-MR Control Chart"
    ])

    summary_report = ""
    chart_path = ""

    if analysis_type == "Beschrijvende statistiek":
        cols = st.multiselect("Kies kolommen", numeric_columns)
        for col in cols:
            desc = df[col].describe()
            summary_report += f"Statistiek voor {col}:\n{desc}\n\n"

    elif analysis_type == "Regressieanalyse":
        y_col = st.selectbox("Kies afhankelijke variabele (Y)", numeric_columns)
        x_cols = st.multiselect("Kies onafhankelijke variabele(n) (X)", numeric_columns)

        if y_col and x_cols:
            formula = f"{y_col} ~ {' + '.join(x_cols)}"
            model = ols(formula, data=df).fit()
            summary_report += f"Regressiemodel: {formula}\n"
            summary_report += str(model.summary())

            if len(x_cols) == 1:
                x_vals = df[x_cols[0]]
                y_vals = df[y_col]
                fig, ax = plt.subplots()
                ax.scatter(x_vals, y_vals, label="Data")
                predicted = model.predict(df[x_cols])
                ax.plot(x_vals, predicted, color='red', label="Regressielijn")
                ax.set_xlabel(x_cols[0])
                ax.set_ylabel(y_col)
                r_squared = model.rsquared
                equation = f"{y_col} = {model.params[0]:.2f} + {model.params[1]:.2f}*{x_cols[0]} (R²={r_squared:.3f})"
                ax.legend()
                ax.set_title("Lineaire regressie")
                chart_path = "regressie.png"
                fig.savefig(chart_path)
                st.pyplot(fig)
                st.markdown(f"📉 Regressievergelijking: `{equation}`")
            else:
                st.info("📊 Regressiegrafiek alleen zichtbaar bij 1 X-variabele.")

    elif analysis_type == "ANOVA":
        cat_col = st.selectbox("Categorische kolom", all_columns)
        val_col = st.selectbox("Numerieke kolom", numeric_columns)
        if cat_col and val_col:
            model = ols(f"{val_col} ~ C({cat_col})", data=df).fit()
            anova_table = stats.f_oneway(*[group[val_col].values for name, group in df.groupby(cat_col)])
            summary_report += f"ANOVA resultaat voor {val_col} per {cat_col}:\nF-statistic={anova_table.statistic}, p-value={anova_table.pvalue}"

    elif analysis_type == "T-toets (2-sample)":
        cat_col = st.selectbox("Kolom met 2 groepen", all_columns)
        val_col = st.selectbox("Numerieke kolom", numeric_columns)
        if cat_col and val_col and df[cat_col].nunique() == 2:
            groups = df[cat_col].unique()
            t_stat, p_val = stats.ttest_ind(
                df[df[cat_col] == groups[0]][val_col],
                df[df[cat_col] == groups[1]][val_col]
            )
            summary_report += f"T-toets voor {val_col} tussen {groups[0]} en {groups[1]}:\nT={t_stat:.3f}, p={p_val:.4f}"
        else:
            st.warning("⚠️ Kies een kolom met exact 2 groepen.")

    elif analysis_type == "I-MR Control Chart":
        col = st.selectbox("Kolom voor controlekaart", numeric_columns)
        if col:
            data = df[col].reset_index(drop=True)
            mean = np.mean(data)
            std_dev = np.std(data, ddof=1)
            ucl = mean + 3 * std_dev
            lcl = mean - 3 * std_dev
            ooc_points = data[(data > ucl) | (data < lcl)].index.tolist()
            mr = data.diff().abs().dropna()
            mr_mean = mr.mean()
            mr_ucl = mr_mean * 3.267

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax1.plot(data, marker='o')
            ax1.axhline(mean, color='green', linestyle='--', label=f'Gemiddelde ({mean:.2f})')
            ax1.axhline(ucl, color='red', linestyle='--', label='UCL')
            ax1.axhline(lcl, color='red', linestyle='--', label='LCL')
            if ooc_points:
                ax1.plot(ooc_points, data[ooc_points], 'ro', label='Out-of-control')
                st.warning(f"⚠️ Out-of-control punten gedetecteerd bij index: {ooc_points}")
            ax1.set_title(f'I Chart voor {col}')
            ax1.legend()

            ax2.plot(mr, marker='o', color='purple')
            ax2.axhline(mr_mean, color='green', linestyle='--', label=f'MR Gemiddelde ({mr_mean:.2f})')
            ax2.axhline(mr_ucl, color='red', linestyle='--', label='MR UCL')
            ax2.set_title('MR Chart')
            ax2.legend()

            fig.tight_layout()
            chart_path = "imr_chart.png"
            fig.savefig(chart_path)
            st.pyplot(fig)

            summary_report += f"I-MR chart voor {col}\\nGemiddelde: {mean:.2f}, UCL: {ucl:.2f}, LCL: {lcl:.2f}\\n"
            summary_report += f"MR Gemiddelde: {mr_mean:.2f}, MR UCL: {mr_ucl:.2f}\\n"
            if ooc_points:
                summary_report += f"⚠️ Out-of-control punten: {ooc_points}\\n"
    if summary_report:
        st.subheader("📄 Analyse Resultaten")
        st.text_area("Resultaat", summary_report, height=300)

        pdf_bytes = generate_pdf_report(summary_report, chart_path)
        b64_pdf = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="rapport.pdf">📥 Download rapport als PDF</a>'
        st.markdown(href, unsafe_allow_html=True)
