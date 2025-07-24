
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from fpdf import FPDF
import seaborn as sns
import io

st.set_page_config(layout="wide")
st.title("📊 MiniStat: Statistische Analyse Tool")

uploaded_file = st.file_uploader("Upload een CSV of Excel-bestand", type=["csv", "xls", "xlsx"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("✅ Bestand succesvol geladen.")
    except Exception as e:
        st.error(f"Fout bij inlezen: {e}")
        st.stop()

    st.write("### Voorbeeld van de dataset")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_column = st.selectbox("Kies een kolom voor analyse", numeric_cols)

    summary_report = f"Statistisch rapport voor kolom: {selected_column}\n\n"

    # Beschrijvende statistiek
    desc = df[selected_column].describe()
    summary_report += "### Beschrijvende statistiek\n"
    summary_report += desc.to_string() + "\n\n"
    st.write("### Beschrijvende statistiek")
    st.write(desc)

    # Regressieanalyse
    if len(numeric_cols) >= 2:
        x_col = st.selectbox("Selecteer X (onafhankelijke variabele)", numeric_cols, index=0)
        y_col = st.selectbox("Selecteer Y (afhankelijke variabele)", numeric_cols, index=1)
        if x_col != y_col:
            X = sm.add_constant(df[x_col])
            model = sm.OLS(df[y_col], X).fit()
            summary_report += f"### Lineaire regressie: {y_col} ~ {x_col}\n"
            summary_report += str(model.summary()) + "\n\n"
            st.write("### Lineaire Regressie Resultaten")
            st.text(model.summary())

            # Regressieplot
            fig1, ax1 = plt.subplots()
            sns.regplot(x=x_col, y=y_col, data=df, ax=ax1)
            ax1.set_title(f"Regressieplot: {y_col} vs. {x_col}")
            fig1.tight_layout()
            reg_path = "regression_plot.png"
            fig1.savefig(reg_path)
            plt.close(fig1)

    # I-MR Chart
    df['MR'] = df[selected_column].diff().abs()
    mean_x = df[selected_column].mean()
    std_x = df[selected_column].std()
    UCL = mean_x + 3 * std_x
    LCL = mean_x - 3 * std_x
    ooc_points = df[(df[selected_column] > UCL) | (df[selected_column] < LCL)]

    def detect_trend(series, window=6):
        last = series[-window:]
        if all(np.diff(last) > 0):
            return "📈 Opwaartse trend gedetecteerd"
        elif all(np.diff(last) < 0):
            return "📉 Neerwaartse trend gedetecteerd"
        else:
            return "Geen duidelijke trend in de laatste waarnemingen"

    trend_text = detect_trend(df[selected_column])

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(df[selected_column], marker='o', label="Waarnemingen")
    ax2.axhline(mean_x, color='green', linestyle='--', label='Gemiddelde')
    ax2.axhline(UCL, color='red', linestyle='--', label='UCL')
    ax2.axhline(LCL, color='red', linestyle='--', label='LCL')
    ax2.scatter(ooc_points.index, ooc_points[selected_column], color='red', marker='X', s=100, label='OOC punten')
    ax2.set_title("I-Chart met OOC detectie")
    ax2.legend()
    plt.figtext(0.5, -0.05, trend_text, ha="center", fontsize=10)
    fig2.tight_layout()
    st.pyplot(fig2)
    imr_path = "imr_chart.png"
    fig2.savefig(imr_path)
    plt.close(fig2)

    # PDF-generatie
    def generate_pdf(text, image_path1=None, image_path2=None):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in text.split("\n"):
            pdf.multi_cell(0, 8, line)
        if image_path1:
            pdf.add_page()
            pdf.image(image_path1, x=10, w=180)
        if image_path2:
            pdf.add_page()
            pdf.image(image_path2, x=10, w=180)
        buffer = io.BytesIO()
        pdf.output(buffer)
        return buffer.getvalue()

    pdf_data = generate_pdf(summary_report, image_path1=reg_path if 'reg_path' in locals() else None,
                            image_path2=imr_path)

    st.download_button("📄 Download PDF-rapport", data=pdf_data, file_name="MiniStat_Rapport.pdf", mime="application/pdf")
