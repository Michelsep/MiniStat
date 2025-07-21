import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# --- Functie: PDF genereren ---
def generate_pdf(summary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in summary_text.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True)
    pdf.output("rapport_minitstat.pdf")
    st.success("📄 PDF gegenereerd als 'rapport_minitstat.pdf'.")

# --- Functie: I-MR controlekaart ---
def plot_imr_chart(data):
    df = pd.DataFrame({'X': data.dropna()})
    df['MR'] = df['X'].diff().abs()

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    mean_x = df['X'].mean()
    mr_bar = df['MR'].mean()

    # I-chart
    axes[0].plot(df['X'], marker='o')
    axes[0].axhline(mean_x, color='green', linestyle='--', label='Gemiddelde')
    UCL = mean_x + 3 * df['X'].std()
    LCL = mean_x - 3 * df['X'].std()
    axes[0].axhline(UCL, color='red', linestyle='--', label='UCL')
    axes[0].axhline(LCL, color='red', linestyle='--', label='LCL')
    axes[0].set_title("I-chart (Individuele Waarden)")
    axes[0].legend()

    # MR-chart
    axes[1].plot(df['MR'], marker='o')
    axes[1].axhline(mr_bar, color='green', linestyle='--', label='Gemiddelde MR')
    mr_UCL = mr_bar * 3.267  # Voor n=2
    axes[1].axhline(mr_UCL, color='red', linestyle='--', label='UCL')
    axes[1].set_title("MR-chart (Moving Range)")
    axes[1].legend()

    st.pyplot(fig)

# --- Streamlit Interface ---
st.title("📊 MiniStat – Statistische Analysetool")
st.write("Upload een CSV-bestand en kies een statistische analyse.")

uploaded_file = st.file_uploader("📁 Upload je CSV-bestand", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Bestand geladen")

    st.subheader("🔍 Dataset voorbeeld")
    st.dataframe(df.head())

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(include='object').columns.tolist()

    st.sidebar.header("📌 Kies Analyse")

    analysis_type = st.sidebar.radio(
        "Selecteer analyse:",
        ("Beschrijvende statistiek", "One-sample T-test", "Two-sample T-test", "Lineaire regressie", "ANOVA", "I-MR Control Chart")
    )

    summary_report = ""

    if analysis_type == "Beschrijvende statistiek":
        col = st.selectbox("Kies kolom", numeric_columns)
        desc = df[col].describe()
        st.write(desc)
        summary_report += f"Beschrijvende statistiek voor {col}:\n{desc}\n"

    elif analysis_type == "One-sample T-test":
        col = st.selectbox("Kies kolom", numeric_columns)
        mu = st.number_input("Verwachte gemiddelde waarde", value=0.0)
        t_stat, p_val = stats.ttest_1samp(df[col].dropna(), mu)
        st.write(f"t = {t_stat:.3f}, p = {p_val:.4f}")
        summary_report += f"One-sample T-test voor {col} (mu={mu}): t = {t_stat:.3f}, p = {p_val:.4f}\n"

    elif analysis_type == "Two-sample T-test":
        col = st.selectbox("Kies numerieke kolom", numeric_columns)
        group_col = st.selectbox("Kies categorische kolom (2 groepen)", categorical_columns)
        groups = df[group_col].dropna().unique()
        if len(groups) == 2:
            g1 = df[df[group_col] == groups[0]][col].dropna()
            g2 = df[df[group_col] == groups[1]][col].dropna()
            t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
            st.write(f"{groups[0]} vs {groups[1]}: t = {t_stat:.3f}, p = {p_val:.4f}")
            summary_report += f"Two-sample T-test voor {col} tussen {groups[0]} en {groups[1]}: t = {t_stat:.3f}, p = {p_val:.4f}\n"
        else:
            st.warning("⚠️ Selecteer een kolom met exact 2 unieke waarden.")

    elif analysis_type == "Lineaire regressie":
        y_col = st.selectbox("Afhankelijke variabele (Y)", numeric_columns)
        x_col = st.selectbox("Onafhankelijke variabele (X)", [col for col in numeric_columns if col != y_col])
        X = sm.add_constant(df[x_col].dropna())
        y = df[y_col].dropna()
        model = sm.OLS(y.loc[X.index], X).fit()
        st.write(model.summary())
        summary_report += f"Regressieanalyse Y={y_col}, X={x_col}:\n{model.summary()}\n"

        # Plot
        fig, ax = plt.subplots()
        sns.regplot(x=df[x_col], y=df[y_col], ax=ax)
        ax.set_title("Regressieplot")
        st.pyplot(fig)

    elif analysis_type == "ANOVA":
        dep = st.selectbox("Kies afhankelijke (numerieke) variabele", numeric_columns)
        group = st.selectbox("Kies categorische groepsvariabele", categorical_columns)
        model = sm.formula.ols(f"{dep} ~ C({group})", data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        st.write(anova_table)
        summary_report += f"ANOVA voor {dep} op basis van groepen in {group}:\n{anova_table}\n"

    elif analysis_type == "I-MR Control Chart":
        col = st.selectbox("Kies kolom voor I-MR chart", numeric_columns)
        st.write("⚙️ Controlekaart wordt gegenereerd...")
        plot_imr_chart(df[col])
        summary_report += f"I-MR Chart gegenereerd voor {col}\n"

    # PDF genereren
    st.subheader("📄 Rapportage")
    if st.button("Genereer PDF van resultaten"):
        if summary_report:
            generate_pdf(summary_report)
        else:
            st.warning("Voer eerst een analyse uit.")