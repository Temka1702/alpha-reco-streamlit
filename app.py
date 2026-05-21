import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss

from catboost import CatBoostClassifier


# -----------------------------
# Synthetic data generator
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_synthetic_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Синтетический датасет, структурно близкий к признакам из Приложения №1:
    income_avg, spend_avg, balance_avg, cash_share, online_share, travel_share,
    sessions_week, session_minutes_avg, invest_views_3m, banner_ctr,
    complaints_6m, days_since_last_ticket, has_deposit, has_credit_card,
    target: opened_deposit_30d
    """
    rng = np.random.default_rng(seed)

    # доход: логнормальное распределение
    income_avg = rng.lognormal(mean=np.log(70000), sigma=0.5, size=n)
    income_avg = np.clip(income_avg, 15000, 400000)

    # расход коррелирует с доходом
    spend_ratio = rng.normal(loc=0.75, scale=0.15, size=n)
    spend_ratio = np.clip(spend_ratio, 0.2, 1.2)
    spend_avg = income_avg * spend_ratio
    spend_avg = np.clip(spend_avg, 5000, 450000)

    # остаток
    balance_avg = income_avg - spend_avg + rng.normal(0, 8000, size=n)
    balance_avg = np.clip(balance_avg, -50000, 250000)

    # доли (0..1)
    cash_share = rng.beta(2, 6, size=n)
    online_share = rng.beta(3, 3, size=n)
    travel_share = rng.beta(1.2, 8, size=n)

    # цифровая активность
    sessions_week = rng.poisson(lam=4, size=n)  # 0.. ~
    sessions_week = np.clip(sessions_week, 0, 20)

    session_minutes_avg = rng.gamma(shape=2.0, scale=1.5, size=n)  # ~ 0..10
    session_minutes_avg = np.clip(session_minutes_avg, 0.2, 15)

    invest_views_3m = rng.poisson(lam=2.5, size=n)
    invest_views_3m = np.clip(invest_views_3m, 0, 30)

    banner_ctr = rng.beta(2, 15, size=n)  # обычно небольшой CTR

    # сервисный профиль
    complaints_6m = rng.poisson(lam=0.6, size=n)
    complaints_6m = np.clip(complaints_6m, 0, 10)

    days_since_last_ticket = rng.integers(low=0, high=365, size=n)

    # продукты
    has_deposit = rng.binomial(1, p=0.25, size=n)
    has_credit_card = rng.binomial(1, p=0.35, size=n)

    # Генерация таргета (вероятность открытия вклада)
    # Логика: растёт от остатка/дохода/интереса к инвестициям/цифровой активности
    # и падает при жалобах + если уже есть вклад
    z = (
        -2.0
        + 0.000010 * income_avg
        + 0.000020 * balance_avg
        + 0.060 * invest_views_3m
        + 0.040 * sessions_week
        + 0.8 * online_share
        - 0.55 * complaints_6m
        - 2.5 * has_deposit
    )

    p = sigmoid(z)
    opened_deposit_30d = rng.binomial(1, p=np.clip(p, 0.01, 0.95), size=n)

    df = pd.DataFrame({
        "income_avg": income_avg.round(2),
        "spend_avg": spend_avg.round(2),
        "balance_avg": balance_avg.round(2),
        "cash_share": cash_share.round(4),
        "online_share": online_share.round(4),
        "travel_share": travel_share.round(4),
        "sessions_week": sessions_week,
        "session_minutes_avg": session_minutes_avg.round(4),
        "invest_views_3m": invest_views_3m,
        "banner_ctr": banner_ctr.round(5),
        "complaints_6m": complaints_6m,
        "days_since_last_ticket": days_since_last_ticket,
        "has_deposit": has_deposit,
        "has_credit_card": has_credit_card,
        "opened_deposit_30d": opened_deposit_30d
    })

    return df


# -----------------------------
# Model training + explainability
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(uploaded_file, n_rows: int, seed: int) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return generate_synthetic_dataset(n=n_rows, seed=seed)


@st.cache_resource(show_spinner=False)
def train_catboost(df: pd.DataFrame, target_col: str, seed: int):
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False
    )

    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]
    metrics = {
        "ROC-AUC": float(roc_auc_score(y_test, proba_test)),
        "LogLoss": float(log_loss(y_test, proba_test)),
        "Test size": int(len(X_test)),
        "Positive rate (test)": float(y_test.mean())
    }

    return model, X_train, X_test, y_train, y_test, proba_test, metrics


def plot_feature_importance(model, feature_names, top_n=15):
    imp = model.get_feature_importance()  # PredictionValuesChange
    fi = pd.DataFrame({"feature": feature_names, "importance": imp})
    fi = fi.sort_values("importance", ascending=False)

    fi_top = fi.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(fi_top["feature"], fi_top["importance"])
    ax.set_title(f"Feature Importance (CatBoost), Top-{top_n}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig, fi


def compute_shap_bar(model, X_sample: pd.DataFrame, top_n=15):
    """
    SHAP из CatBoost: get_feature_importance(type="ShapValues")
    Возвращает mean(|shap|) по каждому признаку и bar chart.
    """
    shap_vals = model.get_feature_importance(type="ShapValues", data=X_sample)
    # shap_vals shape: (n_samples, n_features + 1), последний столбец - base value
    shap_feat = shap_vals[:, :-1]

    mean_abs_shap = np.abs(shap_feat).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    top = shap_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_title(f"SHAP (mean |value|) by feature, Top-{top_n}")
    ax.set_xlabel("mean(|SHAP|)")
    fig.tight_layout()

    return fig, shap_df


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Feature Importance + SHAP (CatBoost)", layout="wide")

st.title("Прототип: Feature Importance и SHAP для модели рекомендаций (CatBoost)")

with st.sidebar:
    st.header("Данные")
    uploaded = st.file_uploader("Загрузить CSV (если есть)", type=["csv"])
    st.caption("Если CSV не загружен — используем синтетические данные.")

    n_rows = st.slider("Размер синтетического датасета", 1000, 20000, 6000, step=1000)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)

    st.header("Параметры графиков")
    top_n = st.slider("Top-N признаков", 5, 30, 15, step=1)
    shap_sample = st.slider("Размер выборки для SHAP", 200, 5000, 1200, step=200)

st.divider()

df = load_data(uploaded, n_rows=n_rows, seed=seed)

target_col = "opened_deposit_30d"
if target_col not in df.columns:
    st.error(f"В датасете нет колонки таргета `{target_col}`. Добавь её или переименуй.")
    st.stop()

st.subheader("1) Данные")
col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(df.head(20), use_container_width=True)
with col2:
    st.metric("Строк", len(df))
    st.metric("Признаков", df.shape[1] - 1)
    st.metric("Доля таргета=1", round(df[target_col].mean(), 4))

st.divider()

st.subheader("2) Обучение модели CatBoost")
with st.spinner("Обучаю модель..."):
    model, X_train, X_test, y_train, y_test, proba_test, metrics = train_catboost(df, target_col, seed)

m1, m2, m3, m4 = st.columns(4)
m1.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")
m2.metric("LogLoss", f"{metrics['LogLoss']:.4f}")
m3.metric("Test size", f"{metrics['Test size']}")
m4.metric("Positive rate (test)", f"{metrics['Positive rate (test)']:.4f}")

tabs = st.tabs(["Feature importance", "SHAP (bar)", "Экспорт"])

# --- Feature importance tab ---
with tabs[0]:
    st.subheader("Feature Importance (CatBoost)")
    fig_fi, fi_df = plot_feature_importance(model, X_train.columns, top_n=top_n)
    st.pyplot(fig_fi, use_container_width=True)
    st.dataframe(fi_df.sort_values("importance", ascending=False).head(30), use_container_width=True)

# --- SHAP tab ---
with tabs[1]:
    st.subheader("SHAP: mean(|SHAP|) по признакам (CatBoost ShapValues)")
    X_sample = X_test.sample(n=min(shap_sample, len(X_test)), random_state=seed)
    with st.spinner("Считаю SHAP..."):
        fig_shap, shap_df = compute_shap_bar(model, X_sample, top_n=top_n)

    st.pyplot(fig_shap, use_container_width=True)
    st.dataframe(shap_df.head(30), use_container_width=True)

    st.caption("Это корректный SHAP-важностный рейтинг: среднее абсолютное значение SHAP по выборке.")

# --- Export tab ---
with tabs[2]:
    st.subheader("Экспорт графиков и таблиц (для вставки в ВКР)")

    fig_fi, fi_df = plot_feature_importance(model, X_train.columns, top_n=top_n)
    X_sample = X_test.sample(n=min(shap_sample, len(X_test)), random_state=seed)
    fig_shap, shap_df = compute_shap_bar(model, X_sample, top_n=top_n)

    st.download_button(
        "Скачать feature_importance.csv",
        data=fi_df.to_csv(index=False).encode("utf-8"),
        file_name="feature_importance_catboost.csv",
        mime="text/csv"
    )

    st.download_button(
        "Скачать shap_importance.csv",
        data=shap_df.to_csv(index=False).encode("utf-8"),
        file_name="shap_importance_mean_abs.csv",
        mime="text/csv"
    )

    st.download_button(
        "Скачать график Feature Importance (PNG)",
        data=fig_to_png_bytes(fig_fi),
        file_name="feature_importance_catboost.png",
        mime="image/png"
    )

    st.download_button(
        "Скачать график SHAP bar (PNG)",
        data=fig_to_png_bytes(fig_shap),
        file_name="shap_mean_abs_bar.png",
        mime="image/png"
    )

st.info(
    "Если хочешь именно SHAP-beeswarm как в статьях (summary_plot), можно добавить зависимость `shap`, "
    "но для Streamlit Cloud это иногда тяжелее. Текущий вариант (mean |SHAP|) обычно достаточно для ВКР."
)
