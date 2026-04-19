from __future__ import annotations

import io
import json
import math
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="A/B Testing Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)


def safe_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def get_llm_client():
    try:
        from openai import OpenAI
    except Exception:
        return None, None
    api_key = st.secrets.get("LLM_API_KEY", "") if hasattr(st, "secrets") else ""
    base_url = st.secrets.get("LLM_BASE_URL", "") if hasattr(st, "secrets") else ""
    model = st.secrets.get("LLM_MODEL", "") if hasattr(st, "secrets") else ""
    if not api_key:
        return None, None
    try:
        client = OpenAI(api_key=api_key, base_url=base_url or None)
        return client, model
    except Exception:
        return None, None


_DEFAULTS = {
    "df": None,
    "file_name": None,
    "variant_col": None,
    "metric_col": None,
    "metric_type": "Binary",
    "timestamp_col": None,
    "segment_col": None,
    "pre_metric_col": None,
    "control_label": None,
    "variant_labels": [],
    "traffic_split": None,
    "hypothesis": "",
    "alpha": 0.05,
    "power": 0.80,
    "mde": 0.02,
    "baseline_rate": 0.10,
    "baseline_mean": 100.0,
    "baseline_std": 30.0,
    "analysis_result": None,
    "validation_report": None,
    "bayes_result": None,
    "sequential_result": None,
    "cuped_result": None,
    "segment_result": None,
    "guardrails": [],
    "guardrail_result": None,
    "decision_result": None,
    "history": [],
    "anomaly_result": None,
}
for _key, _value in _DEFAULTS.items():
    st.session_state.setdefault(_key, _value)


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    seen: Dict[str, int] = {}
    new_cols: List[str] = []
    for c in cols:
        c_str = str(c)
        if c_str in seen:
            seen[c_str] += 1
            new_cols.append(f"{c_str}_{seen[c_str]}")
        else:
            seen[c_str] = 0
            new_cols.append(c_str)
    df.columns = new_cols
    return df


@st.cache_data(show_spinner=False)
def load_dataframe(uploaded_file) -> Optional[pd.DataFrame]:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file, sep=";")
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep="\t")
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        return None
    return dedupe_columns(df)


def sample_size_binary(p1: float, mde: float, alpha: float, power: float) -> int:
    p2 = p1 + mde
    if p2 <= 0 or p2 >= 1 or p1 <= 0 or p1 >= 1:
        return 0
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    numerator = (
        z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    denom = (p2 - p1) ** 2
    if denom == 0:
        return 0
    return int(math.ceil(numerator / denom))


def sample_size_continuous(sd: float, mde: float, alpha: float, power: float) -> int:
    if sd <= 0 or mde <= 0:
        return 0
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) * sd / mde) ** 2
    return int(math.ceil(n))


def run_z_test(c_success: int, c_n: int, v_success: int, v_n: int, alpha: float) -> Dict:
    if c_n == 0 or v_n == 0:
        return {"error": "Örneklem büyüklüğü sıfır."}
    p_c = c_success / c_n
    p_v = v_success / v_n
    p_pool = (c_success + v_success) / (c_n + v_n)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / c_n + 1 / v_n))
    if se == 0:
        return {"error": "Standart hata sıfır."}
    z = (p_v - p_c) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    se_diff = math.sqrt(p_c * (1 - p_c) / c_n + p_v * (1 - p_v) / v_n)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_v - p_c
    ci_low = diff - z_crit * se_diff
    ci_high = diff + z_crit * se_diff
    uplift_rel = (p_v - p_c) / p_c if p_c > 0 else np.nan
    return {
        "method": "Two-proportion z-test",
        "p_control": p_c,
        "p_variant": p_v,
        "diff": diff,
        "z": z,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "uplift_abs": diff,
        "uplift_rel": uplift_rel,
        "alpha": alpha,
        "n_control": c_n,
        "n_variant": v_n,
    }


def run_chi_square(c_success: int, c_n: int, v_success: int, v_n: int) -> Dict:
    observed = np.array(
        [[c_success, c_n - c_success], [v_success, v_n - v_success]]
    )
    chi2, p, dof, _ = stats.chi2_contingency(observed)
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof)}


def run_t_test(c: np.ndarray, v: np.ndarray, alpha: float, welch: bool = True) -> Dict:
    if len(c) < 2 or len(v) < 2:
        return {"error": "Her grupta en az 2 gözlem olmalı."}
    t_stat, p_value = stats.ttest_ind(v, c, equal_var=not welch)
    mean_c = float(np.mean(c))
    mean_v = float(np.mean(v))
    sd_c = float(np.std(c, ddof=1))
    sd_v = float(np.std(v, ddof=1))
    n_c, n_v = len(c), len(v)
    if welch:
        se_diff = math.sqrt(sd_c ** 2 / n_c + sd_v ** 2 / n_v)
        df = (sd_c ** 2 / n_c + sd_v ** 2 / n_v) ** 2 / (
            (sd_c ** 2 / n_c) ** 2 / (n_c - 1) + (sd_v ** 2 / n_v) ** 2 / (n_v - 1)
        )
    else:
        sp2 = ((n_c - 1) * sd_c ** 2 + (n_v - 1) * sd_v ** 2) / (n_c + n_v - 2)
        se_diff = math.sqrt(sp2 * (1 / n_c + 1 / n_v))
        df = n_c + n_v - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    diff = mean_v - mean_c
    ci_low = diff - t_crit * se_diff
    ci_high = diff + t_crit * se_diff
    pooled_sd = math.sqrt(((n_c - 1) * sd_c ** 2 + (n_v - 1) * sd_v ** 2) / (n_c + n_v - 2))
    cohen_d = diff / pooled_sd if pooled_sd > 0 else np.nan
    uplift_rel = diff / mean_c if mean_c != 0 else np.nan
    return {
        "method": "Welch's t-test" if welch else "Student's t-test",
        "mean_control": mean_c,
        "mean_variant": mean_v,
        "sd_control": sd_c,
        "sd_variant": sd_v,
        "diff": diff,
        "t": float(t_stat),
        "df": float(df),
        "p_value": float(p_value),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cohen_d": float(cohen_d),
        "uplift_abs": diff,
        "uplift_rel": uplift_rel,
        "alpha": alpha,
        "n_control": n_c,
        "n_variant": n_v,
    }


def run_mann_whitney(c: np.ndarray, v: np.ndarray) -> Dict:
    if len(c) < 2 or len(v) < 2:
        return {"error": "Her grupta en az 2 gözlem olmalı."}
    u, p = stats.mannwhitneyu(v, c, alternative="two-sided")
    return {"method": "Mann-Whitney U", "u": float(u), "p_value": float(p)}


def bayesian_binary(c_success: int, c_n: int, v_success: int, v_n: int, draws: int = 50000) -> Dict:
    rng = np.random.default_rng(42)
    alpha_prior, beta_prior = 1.0, 1.0
    post_c = rng.beta(alpha_prior + c_success, beta_prior + c_n - c_success, size=draws)
    post_v = rng.beta(alpha_prior + v_success, beta_prior + v_n - v_success, size=draws)
    prob_v_better = float(np.mean(post_v > post_c))
    diff = post_v - post_c
    expected_lift = float(np.mean(diff))
    ci = np.percentile(diff, [2.5, 97.5])
    expected_loss_v = float(np.mean(np.maximum(post_c - post_v, 0)))
    return {
        "method": "Bayesian Beta-Binomial",
        "prob_variant_better": prob_v_better,
        "expected_lift_abs": expected_lift,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "expected_loss_variant": expected_loss_v,
    }


def bayesian_continuous(c: np.ndarray, v: np.ndarray, draws: int = 20000) -> Dict:
    rng = np.random.default_rng(42)
    n_c, n_v = len(c), len(v)
    if n_c < 2 or n_v < 2:
        return {"error": "Her grupta en az 2 gözlem olmalı."}
    mean_c = float(np.mean(c))
    mean_v = float(np.mean(v))
    sd_c = float(np.std(c, ddof=1))
    sd_v = float(np.std(v, ddof=1))
    se_c = sd_c / math.sqrt(n_c)
    se_v = sd_v / math.sqrt(n_v)
    post_c = rng.normal(mean_c, se_c, size=draws)
    post_v = rng.normal(mean_v, se_v, size=draws)
    prob = float(np.mean(post_v > post_c))
    diff = post_v - post_c
    ci = np.percentile(diff, [2.5, 97.5])
    return {
        "method": "Bayesian Normal approximation",
        "prob_variant_better": prob,
        "expected_lift_abs": float(np.mean(diff)),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "expected_loss_variant": float(np.mean(np.maximum(post_c - post_v, 0))),
    }


def srm_check(observed_counts: Dict[str, int], expected_split: Dict[str, float]) -> Dict:
    total = sum(observed_counts.values())
    keys = list(observed_counts.keys())
    obs = np.array([observed_counts[k] for k in keys], dtype=float)
    exp = np.array([expected_split[k] * total for k in keys], dtype=float)
    if any(exp == 0) or total == 0:
        return {"error": "SRM için yeterli veri yok."}
    chi2 = float(np.sum((obs - exp) ** 2 / exp))
    dof = len(keys) - 1
    p_value = float(1 - stats.chi2.cdf(chi2, dof))
    return {
        "chi2": chi2,
        "dof": dof,
        "p_value": p_value,
        "srm_detected": p_value < 0.001,
        "observed": dict(zip(keys, obs.astype(int).tolist())),
        "expected": dict(zip(keys, exp.tolist())),
    }


def detect_outliers_iqr(x: np.ndarray) -> Tuple[int, Tuple[float, float]]:
    if len(x) < 4:
        return 0, (float("nan"), float("nan"))
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return int(np.sum((x < low) | (x > high))), (float(low), float(high))


def cuped_adjust(
    pre: np.ndarray, post: np.ndarray, variant: np.ndarray, control_label: str
) -> Dict:
    mask_c = variant == control_label
    if mask_c.sum() < 2 or (~mask_c).sum() < 2:
        return {"error": "CUPED için yeterli veri yok."}
    mu_pre = float(np.mean(pre))
    var_pre = float(np.var(pre, ddof=1))
    if var_pre == 0:
        return {"error": "Pre-period varyansı sıfır."}
    cov = float(np.cov(pre, post, ddof=1)[0, 1])
    theta = cov / var_pre
    adjusted = post - theta * (pre - mu_pre)
    var_reduction = 1 - (np.var(adjusted, ddof=1) / np.var(post, ddof=1))
    return {
        "theta": theta,
        "variance_reduction_pct": float(var_reduction * 100),
        "adjusted": adjusted,
    }


def sequential_obf(
    c_success: int, c_n: int, v_success: int, v_n: int, looks_total: int, look_idx: int, alpha: float
) -> Dict:
    if look_idx < 1 or look_idx > looks_total:
        return {"error": "Look index geçersiz."}
    t = look_idx / looks_total
    if t == 0:
        return {"error": "t=0"}
    alpha_t = 2 * (1 - stats.norm.cdf(stats.norm.ppf(1 - alpha / 2) / math.sqrt(t)))
    z = run_z_test(c_success, c_n, v_success, v_n, alpha)
    if "error" in z:
        return z
    reject = abs(z["z"]) > stats.norm.ppf(1 - alpha_t / 2)
    return {
        "t_fraction": t,
        "alpha_boundary": float(alpha_t),
        "z_stat": z["z"],
        "p_value": z["p_value"],
        "reject_h0": bool(reject),
        "recommendation": "Early stop: anlamlı" if reject else "Devam et",
    }


def multiple_testing_correction(p_values: List[float], method: str = "bonferroni") -> Dict:
    p_arr = np.array(p_values, dtype=float)
    m = len(p_arr)
    if m == 0:
        return {"error": "p-değer yok."}
    if method == "bonferroni":
        adj = np.minimum(p_arr * m, 1.0)
    elif method == "fdr_bh":
        order = np.argsort(p_arr)
        ranked = p_arr[order]
        adj_ranked = ranked * m / (np.arange(1, m + 1))
        adj_ranked = np.minimum.accumulate(adj_ranked[::-1])[::-1]
        adj = np.empty_like(adj_ranked)
        adj[order] = np.minimum(adj_ranked, 1.0)
    else:
        adj = p_arr
    return {"method": method, "adjusted": adj.tolist()}


def what_if_binary(baseline: float, uplift: float, n_per_arm: int, alpha: float) -> Dict:
    new_rate = baseline + uplift
    if new_rate <= 0 or new_rate >= 1:
        return {"error": "Yeni oran 0-1 aralığı dışında."}
    c_success = int(round(baseline * n_per_arm))
    v_success = int(round(new_rate * n_per_arm))
    return run_z_test(c_success, n_per_arm, v_success, n_per_arm, alpha)


def decision_engine(
    p_value: Optional[float],
    uplift_abs: Optional[float],
    uplift_rel: Optional[float],
    alpha: float,
    guardrail_violated: bool,
    n_required: int,
    n_actual: int,
    bayes_prob: Optional[float] = None,
) -> Dict:
    reasons: List[str] = []
    if n_actual < n_required * 0.7:
        reasons.append(
            f"Örneklem yetersiz (mevcut {n_actual:,}, gereken {n_required:,})."
        )
        return {"decision": "Continue Test", "icon": "Continue", "reasons": reasons}
    if guardrail_violated:
        reasons.append("Guardrail metriklerinde olumsuz sapma tespit edildi.")
        return {"decision": "Do Not Launch", "icon": "Do Not Launch", "reasons": reasons}
    if p_value is None:
        return {"decision": "Re-run Experiment", "icon": "Re-run", "reasons": ["İstatistik sonucu üretilemedi."]}
    if p_value < alpha and (uplift_abs is not None and uplift_abs > 0):
        reasons.append(f"p-değeri {p_value:.4f} < alpha {alpha:.3f}.")
        if bayes_prob is not None:
            reasons.append(f"Varyantın daha iyi olma olasılığı: {bayes_prob*100:.1f}%.")
        return {"decision": "Ship Winner", "icon": "Ship Winner", "reasons": reasons}
    if p_value < alpha and (uplift_abs is not None and uplift_abs < 0):
        reasons.append("Varyant anlamlı şekilde daha kötü.")
        return {"decision": "Do Not Launch", "icon": "Do Not Launch", "reasons": reasons}
    if p_value >= alpha and n_actual >= n_required:
        reasons.append("Örneklem yeterli ancak anlamlı fark yok.")
        return {"decision": "No Significant Difference", "icon": "No Difference", "reasons": reasons}
    reasons.append("Sonuç kararsız, testin devam etmesi önerilir.")
    return {"decision": "Continue Test", "icon": "Continue", "reasons": reasons}


def build_executive_summary(result: Dict, decision: Dict, hypothesis: str) -> str:
    lines: List[str] = []
    lines.append("EXECUTIVE SUMMARY")
    lines.append(f"Hipotez: {hypothesis or '-'}")
    lines.append(f"Karar: {decision.get('decision','-')}")
    if result:
        if "p_value" in result:
            lines.append(f"p-değer: {result['p_value']:.4f}")
        if "uplift_rel" in result and result.get("uplift_rel") is not None and not (
            isinstance(result["uplift_rel"], float) and math.isnan(result["uplift_rel"])
        ):
            lines.append(f"Relatif uplift: {result['uplift_rel']*100:.2f}%")
        if "uplift_abs" in result and result.get("uplift_abs") is not None:
            lines.append(f"Mutlak fark: {result['uplift_abs']:.4f}")
        if "ci_low" in result and "ci_high" in result:
            lines.append(f"Güven aralığı: [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
        if "n_control" in result:
            lines.append(f"Örneklem: kontrol={result['n_control']:,}, varyant={result['n_variant']:,}")
    for r in decision.get("reasons", []):
        lines.append(f"- {r}")
    return "\n".join(lines)


def llm_explain(summary_text: str) -> Optional[str]:
    client, model = get_llm_client()
    if client is None or not model:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sen bir A/B test analistisin. Verilen teknik özet üzerinden "
                        "yöneticiler için Türkçe, sade ve aksiyon odaklı bir açıklama üret. "
                        "Çıktın: kısa özet, sonuç yorumu, önerilen aksiyon ve riskler. "
                        "Emoji kullanma."
                    ),
                },
                {"role": "user", "content": summary_text},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"LLM açıklaması üretilemedi: {exc}"


def export_excel(
    hypothesis: str,
    result: Optional[Dict],
    decision: Optional[Dict],
    validation: Optional[Dict],
    bayes: Optional[Dict],
    segment: Optional[pd.DataFrame],
) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "field": ["hypothesis", "timestamp"],
                "value": [hypothesis, datetime.now().isoformat(timespec="seconds")],
            }
        ).to_excel(writer, sheet_name="Summary", index=False)
        if result:
            pd.DataFrame([{k: v for k, v in result.items() if not isinstance(v, (list, np.ndarray))}]).to_excel(
                writer, sheet_name="Analysis", index=False
            )
        if decision:
            pd.DataFrame(
                [
                    {
                        "decision": decision.get("decision"),
                        "reasons": " | ".join(decision.get("reasons", [])),
                    }
                ]
            ).to_excel(writer, sheet_name="Decision", index=False)
        if validation:
            pd.DataFrame([validation]).to_excel(writer, sheet_name="Validation", index=False)
        if bayes:
            pd.DataFrame([bayes]).to_excel(writer, sheet_name="Bayesian", index=False)
        if segment is not None and not segment.empty:
            segment.to_excel(writer, sheet_name="Segments", index=False)
    buf.seek(0)
    return buf.getvalue()


def export_json(payload: Dict) -> bytes:
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    return json.dumps(payload, indent=2, ensure_ascii=False, default=_default).encode("utf-8")


def sidebar_nav() -> str:
    st.sidebar.title("A/B Testing Agent")
    st.sidebar.caption("Uçtan uca deney yönetim aracı")
    return st.sidebar.radio(
        "Adımlar",
        [
            "1. Deney Tasarımı",
            "2. Veri Yükleme & Eşleme",
            "3. Veri Validasyonu",
            "4. İstatistiksel Analiz",
            "5. Guardrail İzleme",
            "6. Bayesian Analiz",
            "7. Sequential / Peeking",
            "8. CUPED & Düzeltmeler",
            "9. Segment Analizi",
            "10. What-If Simülasyonu",
            "11. Anomali İzleme",
            "12. Karar & Rapor",
            "13. Deney Bilgi Bankası",
        ],
        index=0,
    )


def page_experiment_setup() -> None:
    st.header("Deney Tasarımı")
    st.caption("Hipotez, metrik seçimi, trafik dağılımı ve otomatik sample size.")

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.hypothesis = st.text_area(
            "Hipotez",
            value=st.session_state.hypothesis,
            placeholder="Örn: Yeni checkout akışı dönüşüm oranını artıracak.",
            height=100,
        )
        st.session_state.metric_type = st.selectbox(
            "Primary metric tipi",
            ["Binary", "Continuous", "Count", "Rate", "Retention"],
            index=["Binary", "Continuous", "Count", "Rate", "Retention"].index(
                st.session_state.metric_type
            ),
        )
        st.session_state.alpha = st.number_input(
            "Alpha (Significance)", min_value=0.001, max_value=0.2, value=st.session_state.alpha, step=0.005
        )
        st.session_state.power = st.number_input(
            "Power", min_value=0.5, max_value=0.99, value=st.session_state.power, step=0.05
        )
    with col2:
        st.session_state.mde = st.number_input(
            "Minimum Detectable Effect (MDE, mutlak)",
            min_value=0.0001,
            value=float(st.session_state.mde),
            step=0.005,
            format="%.4f",
        )
        n_arms = st.number_input("Kol sayısı (kontrol dahil)", min_value=2, max_value=6, value=2, step=1)
        default_split = [1 / n_arms] * n_arms
        split_text = st.text_input(
            "Trafik dağılımı (virgüllü, toplam 1)",
            value=",".join(f"{x:.2f}" for x in default_split),
        )
        try:
            parsed = [float(x.strip()) for x in split_text.split(",") if x.strip()]
            if len(parsed) == n_arms and abs(sum(parsed) - 1.0) < 0.01:
                st.session_state.traffic_split = parsed
            else:
                st.warning("Dağılım toplamı 1 olmalı ve kol sayısına eşit olmalı.")
        except Exception:
            st.warning("Dağılım parse edilemedi.")

    st.subheader("Sample Size Hesaplayıcı")
    if st.session_state.metric_type == "Binary":
        st.session_state.baseline_rate = st.number_input(
            "Baseline conversion rate (kontrol)",
            min_value=0.0001,
            max_value=0.9999,
            value=float(st.session_state.baseline_rate),
            step=0.005,
            format="%.4f",
        )
        n = sample_size_binary(
            st.session_state.baseline_rate, st.session_state.mde, st.session_state.alpha, st.session_state.power
        )
    else:
        st.session_state.baseline_mean = st.number_input(
            "Baseline ortalama", value=float(st.session_state.baseline_mean)
        )
        st.session_state.baseline_std = st.number_input(
            "Baseline standart sapma", min_value=0.0001, value=float(st.session_state.baseline_std)
        )
        n = sample_size_continuous(
            st.session_state.baseline_std,
            st.session_state.mde,
            st.session_state.alpha,
            st.session_state.power,
        )

    st.metric("Kol başına önerilen örneklem", f"{n:,}")
    total = n * int(n_arms)
    st.caption(f"Toplam gereken örneklem yaklaşık {total:,}.")


def page_upload() -> None:
    st.header("Veri Yükleme & Eşleme")
    uploaded = st.file_uploader("CSV veya XLSX", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        df = load_dataframe(uploaded)
        if df is None:
            st.error("Dosya okunamadı.")
            return
        st.session_state.df = df
        st.session_state.file_name = uploaded.name
        st.success(f"Yüklendi: {uploaded.name} ({len(df):,} satır, {df.shape[1]} kolon)")

    df = st.session_state.df
    if df is None:
        st.info("Önce veri yükleyin.")
        return

    st.subheader("Önizleme")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Kolon Eşleme")
    cols = list(df.columns)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.variant_col = st.selectbox(
            "Variant kolonu", cols, index=cols.index(st.session_state.variant_col) if st.session_state.variant_col in cols else 0
        )
    with c2:
        st.session_state.metric_col = st.selectbox(
            "Primary metric kolonu",
            cols,
            index=cols.index(st.session_state.metric_col) if st.session_state.metric_col in cols else 0,
        )
    with c3:
        ts_options = ["(yok)"] + cols
        cur = st.session_state.timestamp_col or "(yok)"
        st.session_state.timestamp_col = st.selectbox(
            "Timestamp kolonu", ts_options, index=ts_options.index(cur) if cur in ts_options else 0
        )
        if st.session_state.timestamp_col == "(yok)":
            st.session_state.timestamp_col = None

    c4, c5 = st.columns(2)
    with c4:
        seg_options = ["(yok)"] + cols
        cur_seg = st.session_state.segment_col or "(yok)"
        st.session_state.segment_col = st.selectbox(
            "Segment kolonu (opsiyonel)",
            seg_options,
            index=seg_options.index(cur_seg) if cur_seg in seg_options else 0,
        )
        if st.session_state.segment_col == "(yok)":
            st.session_state.segment_col = None
    with c5:
        pre_options = ["(yok)"] + cols
        cur_pre = st.session_state.pre_metric_col or "(yok)"
        st.session_state.pre_metric_col = st.selectbox(
            "Pre-period metric (CUPED için)",
            pre_options,
            index=pre_options.index(cur_pre) if cur_pre in pre_options else 0,
        )
        if st.session_state.pre_metric_col == "(yok)":
            st.session_state.pre_metric_col = None

    if st.session_state.variant_col:
        variants = sorted([str(x) for x in df[st.session_state.variant_col].dropna().unique().tolist()])
        if variants:
            default_ctrl = st.session_state.control_label if st.session_state.control_label in variants else variants[0]
            st.session_state.control_label = st.selectbox(
                "Kontrol etiketi", variants, index=variants.index(default_ctrl)
            )
            st.session_state.variant_labels = [v for v in variants if v != st.session_state.control_label]
            st.caption(
                f"Kontrol: {st.session_state.control_label} | Varyantlar: {', '.join(st.session_state.variant_labels)}"
            )


def require_mapping() -> bool:
    if st.session_state.df is None:
        st.info("Önce veri yükleyin.")
        return False
    if not st.session_state.variant_col or not st.session_state.metric_col:
        st.info("Variant ve metric kolonlarını eşleyin.")
        return False
    return True


def get_two_arms() -> Optional[Tuple[str, str, pd.DataFrame]]:
    if not require_mapping():
        return None
    df = st.session_state.df
    if not st.session_state.variant_labels:
        st.info("Varyant bulunamadı.")
        return None
    ctrl = st.session_state.control_label
    variant = st.selectbox("Karşılaştırılacak varyant", st.session_state.variant_labels)
    sub = df[df[st.session_state.variant_col].astype(str).isin([str(ctrl), str(variant)])].copy()
    return ctrl, variant, sub


def page_validation() -> None:
    st.header("Veri Validasyonu")
    if not require_mapping():
        return
    df = st.session_state.df
    report: Dict = {}

    variant_col = st.session_state.variant_col
    metric_col = st.session_state.metric_col

    counts = df[variant_col].astype(str).value_counts().to_dict()
    total = sum(counts.values())
    n_arms = len(counts)
    if n_arms >= 2 and total > 0:
        if st.session_state.traffic_split and len(st.session_state.traffic_split) == n_arms:
            labels = list(counts.keys())
            expected = dict(zip(labels, st.session_state.traffic_split))
        else:
            expected = {k: 1 / n_arms for k in counts}
        srm = srm_check(counts, expected)
        report["srm"] = srm

    dup_count = int(df.duplicated().sum())
    report["duplicates"] = dup_count

    missing = df[[variant_col, metric_col]].isna().sum().to_dict()
    report["missing"] = {k: int(v) for k, v in missing.items()}

    if pd.api.types.is_numeric_dtype(df[metric_col]):
        outliers, bounds = detect_outliers_iqr(df[metric_col].dropna().to_numpy())
        report["outliers"] = {"count": outliers, "lower": bounds[0], "upper": bounds[1]}

    try:
        _ = pd.to_numeric(df[metric_col], errors="raise")
        report["metric_numeric"] = True
    except Exception:
        report["metric_numeric"] = False

    if st.session_state.timestamp_col:
        try:
            ts = pd.to_datetime(df[st.session_state.timestamp_col], errors="coerce")
            report["timestamp_parsable_pct"] = float(ts.notna().mean() * 100)
        except Exception:
            report["timestamp_parsable_pct"] = 0.0

    st.session_state.validation_report = report

    c1, c2, c3 = st.columns(3)
    with c1:
        if "srm" in report and isinstance(report["srm"], dict) and "srm_detected" in report["srm"]:
            st.metric("SRM p-değeri", f"{report['srm']['p_value']:.4f}")
            if report["srm"]["srm_detected"]:
                st.error("Sample Ratio Mismatch tespit edildi.")
            else:
                st.success("SRM bulunmadı.")
    with c2:
        st.metric("Duplicate kayıt", f"{dup_count:,}")
    with c3:
        st.metric("Eksik (metric)", f"{report['missing'].get(metric_col, 0):,}")

    st.subheader("Detay Rapor")
    st.json(report)

    st.subheader("Dağılım")
    dist = df[variant_col].astype(str).value_counts().reset_index()
    dist.columns = ["variant", "count"]
    fig = px.bar(dist, x="variant", y="count")
    st.plotly_chart(fig, use_container_width=True)


def page_statistical_analysis() -> None:
    st.header("İstatistiksel Analiz")
    tup = get_two_arms()
    if tup is None:
        return
    ctrl, variant, sub = tup
    metric_col = st.session_state.metric_col
    metric_type = st.session_state.metric_type

    method_options = []
    if metric_type == "Binary":
        method_options = ["z-test (two-proportion)", "Chi-square"]
    else:
        method_options = ["Welch's t-test", "Student's t-test", "Mann-Whitney U"]
    method = st.selectbox("Yöntem", method_options)

    c_arr = sub[sub[st.session_state.variant_col].astype(str) == str(ctrl)][metric_col]
    v_arr = sub[sub[st.session_state.variant_col].astype(str) == str(variant)][metric_col]

    result: Dict = {}
    if metric_type == "Binary":
        c_num = pd.to_numeric(c_arr, errors="coerce").dropna()
        v_num = pd.to_numeric(v_arr, errors="coerce").dropna()
        c_success = int(c_num.sum())
        c_n = int(len(c_num))
        v_success = int(v_num.sum())
        v_n = int(len(v_num))
        if method == "z-test (two-proportion)":
            result = run_z_test(c_success, c_n, v_success, v_n, st.session_state.alpha)
        else:
            chi = run_chi_square(c_success, c_n, v_success, v_n)
            z = run_z_test(c_success, c_n, v_success, v_n, st.session_state.alpha)
            result = {**z, **{"chi2": chi.get("chi2"), "chi_p": chi.get("p_value")}, "method": "Chi-square"}
    else:
        c_num = pd.to_numeric(c_arr, errors="coerce").dropna().to_numpy()
        v_num = pd.to_numeric(v_arr, errors="coerce").dropna().to_numpy()
        if method.startswith("Welch"):
            result = run_t_test(c_num, v_num, st.session_state.alpha, welch=True)
        elif method.startswith("Student"):
            result = run_t_test(c_num, v_num, st.session_state.alpha, welch=False)
        else:
            result = run_mann_whitney(c_num, v_num)

    st.session_state.analysis_result = result

    if "error" in result:
        st.error(result["error"])
        return

    cols = st.columns(4)
    if "p_value" in result:
        cols[0].metric("p-değer", f"{result['p_value']:.4f}")
    if "uplift_rel" in result and result.get("uplift_rel") is not None and not (
        isinstance(result["uplift_rel"], float) and math.isnan(result["uplift_rel"])
    ):
        cols[1].metric("Relatif uplift", f"{result['uplift_rel']*100:.2f}%")
    if "uplift_abs" in result:
        cols[2].metric("Mutlak fark", f"{result['uplift_abs']:.4f}")
    if "ci_low" in result and "ci_high" in result:
        cols[3].metric("Güven aralığı", f"[{result['ci_low']:.3f}, {result['ci_high']:.3f}]")

    st.subheader("Yöntem ve Detay")
    st.json({k: v for k, v in result.items() if not isinstance(v, (list, np.ndarray))})


def page_guardrail() -> None:
    st.header("Guardrail İzleme")
    if not require_mapping():
        return
    df = st.session_state.df
    cols = list(df.columns)
    selected = st.multiselect(
        "Guardrail metrikleri",
        [c for c in cols if c != st.session_state.metric_col],
        default=st.session_state.guardrails or [],
    )
    threshold = st.number_input("Negatif sapma eşiği (relatif)", value=0.02, step=0.005, format="%.3f")
    st.session_state.guardrails = selected

    if not selected:
        st.info("Guardrail seçilmedi.")
        return

    ctrl = st.session_state.control_label
    rows = []
    violated = False
    for col in selected:
        if not pd.api.types.is_numeric_dtype(df[col]):
            rows.append({"metric": col, "status": "Sayısal değil, atlandı."})
            continue
        c_mean = df[df[st.session_state.variant_col].astype(str) == str(ctrl)][col].mean()
        for v in st.session_state.variant_labels:
            v_mean = df[df[st.session_state.variant_col].astype(str) == str(v)][col].mean()
            if c_mean == 0 or pd.isna(c_mean):
                rel = np.nan
            else:
                rel = (v_mean - c_mean) / abs(c_mean)
            bad = (rel < -threshold) if not pd.isna(rel) else False
            if bad:
                violated = True
            rows.append(
                {
                    "metric": col,
                    "variant": v,
                    "control_mean": float(c_mean) if not pd.isna(c_mean) else None,
                    "variant_mean": float(v_mean) if not pd.isna(v_mean) else None,
                    "relative_change": None if pd.isna(rel) else float(rel),
                    "violated": bool(bad),
                }
            )
    table = pd.DataFrame(rows)
    st.dataframe(table, use_container_width=True)
    if violated:
        st.error("Primary metric iyileşmiş olabilir, ancak guardrail metriklerinde olumsuz etki görülüyor.")
    else:
        st.success("Guardrail metriklerinde olumsuz sapma yok.")
    st.session_state.guardrail_result = {"violated": violated, "rows": rows}


def page_bayesian() -> None:
    st.header("Bayesian Analiz")
    tup = get_two_arms()
    if tup is None:
        return
    ctrl, variant, sub = tup
    metric_col = st.session_state.metric_col
    metric_type = st.session_state.metric_type

    c_arr = sub[sub[st.session_state.variant_col].astype(str) == str(ctrl)][metric_col]
    v_arr = sub[sub[st.session_state.variant_col].astype(str) == str(variant)][metric_col]

    if metric_type == "Binary":
        c_num = pd.to_numeric(c_arr, errors="coerce").dropna()
        v_num = pd.to_numeric(v_arr, errors="coerce").dropna()
        res = bayesian_binary(int(c_num.sum()), int(len(c_num)), int(v_num.sum()), int(len(v_num)))
    else:
        c_num = pd.to_numeric(c_arr, errors="coerce").dropna().to_numpy()
        v_num = pd.to_numeric(v_arr, errors="coerce").dropna().to_numpy()
        res = bayesian_continuous(c_num, v_num)

    st.session_state.bayes_result = res
    if "error" in res:
        st.error(res["error"])
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("P(varyant > kontrol)", f"{res['prob_variant_better']*100:.2f}%")
    c2.metric("Beklenen lift", f"{res['expected_lift_abs']:.4f}")
    c3.metric("Expected loss (varyant)", f"{res['expected_loss_variant']:.4f}")
    st.json(res)


def page_sequential() -> None:
    st.header("Sequential Testing & Peeking-Safe Analiz")
    tup = get_two_arms()
    if tup is None:
        return
    ctrl, variant, sub = tup
    metric_col = st.session_state.metric_col

    if st.session_state.metric_type != "Binary":
        st.info("Bu basitleştirilmiş sequential analiz binary metrikler için kurgulanmıştır.")
        return

    looks_total = st.number_input("Toplam planlanan bakış sayısı", min_value=2, max_value=50, value=5, step=1)
    look_idx = st.number_input("Mevcut bakış indeksi", min_value=1, max_value=int(looks_total), value=1, step=1)

    c_num = pd.to_numeric(sub[sub[st.session_state.variant_col].astype(str) == str(ctrl)][metric_col], errors="coerce").dropna()
    v_num = pd.to_numeric(sub[sub[st.session_state.variant_col].astype(str) == str(variant)][metric_col], errors="coerce").dropna()
    res = sequential_obf(
        int(c_num.sum()), int(len(c_num)), int(v_num.sum()), int(len(v_num)),
        int(looks_total), int(look_idx), st.session_state.alpha
    )
    st.session_state.sequential_result = res
    if "error" in res:
        st.error(res["error"])
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("t (ilerleme)", f"{res['t_fraction']:.2f}")
    c2.metric("Düzeltilmiş alpha", f"{res['alpha_boundary']:.4f}")
    c3.metric("Öneri", res["recommendation"])
    st.json(res)


def page_cuped_correction() -> None:
    st.header("CUPED & Çoklu Test Düzeltmesi")
    tup = get_two_arms()
    if tup is None:
        return
    ctrl, variant, sub = tup
    metric_col = st.session_state.metric_col

    st.subheader("CUPED")
    if not st.session_state.pre_metric_col:
        st.info("CUPED için pre-period metric kolonu eşleyin.")
    else:
        pre = pd.to_numeric(sub[st.session_state.pre_metric_col], errors="coerce").to_numpy()
        post = pd.to_numeric(sub[metric_col], errors="coerce").to_numpy()
        varr = sub[st.session_state.variant_col].astype(str).to_numpy()
        mask = ~(pd.isna(pre) | pd.isna(post))
        cuped = cuped_adjust(pre[mask], post[mask], varr[mask], str(ctrl))
        st.session_state.cuped_result = cuped
        if "error" in cuped:
            st.warning(cuped["error"])
        else:
            c1, c2 = st.columns(2)
            c1.metric("Varyans azaltımı", f"{cuped['variance_reduction_pct']:.2f}%")
            c2.metric("Theta", f"{cuped['theta']:.4f}")
            adj = cuped["adjusted"]
            c_mask = varr[mask] == str(ctrl)
            v_mask = varr[mask] == str(variant)
            if c_mask.sum() > 1 and v_mask.sum() > 1:
                t_res = run_t_test(adj[c_mask], adj[v_mask], st.session_state.alpha, welch=True)
                st.subheader("CUPED-düzeltilmiş Welch t-test")
                st.json({k: v for k, v in t_res.items() if not isinstance(v, (list, np.ndarray))})

    st.subheader("Çoklu Test Düzeltmesi")
    raw = st.text_input("p-değerleri (virgüllü)", value="")
    method = st.selectbox("Yöntem", ["bonferroni", "fdr_bh"])
    if raw.strip():
        try:
            pvs = [float(x.strip()) for x in raw.split(",") if x.strip()]
            res = multiple_testing_correction(pvs, method=method)
            st.json(res)
        except Exception as exc:
            st.error(f"Parse hatası: {exc}")


def page_segment() -> None:
    st.header("Segment & Deep Dive Analiz")
    tup = get_two_arms()
    if tup is None:
        return
    ctrl, variant, sub = tup
    seg_col = st.session_state.segment_col
    if not seg_col:
        st.info("Segment kolonu seçilmedi.")
        return
    metric_col = st.session_state.metric_col
    rows = []
    for seg, g in sub.groupby(seg_col, dropna=False):
        c_arr = pd.to_numeric(g[g[st.session_state.variant_col].astype(str) == str(ctrl)][metric_col], errors="coerce").dropna()
        v_arr = pd.to_numeric(g[g[st.session_state.variant_col].astype(str) == str(variant)][metric_col], errors="coerce").dropna()
        if len(c_arr) < 5 or len(v_arr) < 5:
            rows.append({"segment": seg, "n_control": len(c_arr), "n_variant": len(v_arr), "status": "yetersiz"})
            continue
        if st.session_state.metric_type == "Binary":
            r = run_z_test(int(c_arr.sum()), int(len(c_arr)), int(v_arr.sum()), int(len(v_arr)), st.session_state.alpha)
        else:
            r = run_t_test(c_arr.to_numpy(), v_arr.to_numpy(), st.session_state.alpha, welch=True)
        if "error" in r:
            rows.append({"segment": seg, "status": r["error"]})
            continue
        rows.append(
            {
                "segment": seg,
                "n_control": r.get("n_control"),
                "n_variant": r.get("n_variant"),
                "p_value": r.get("p_value"),
                "uplift_rel": r.get("uplift_rel"),
                "category": "Confirmed" if r.get("p_value", 1) < st.session_state.alpha else "Exploratory",
            }
        )
    tbl = pd.DataFrame(rows)
    st.session_state.segment_result = tbl
    st.dataframe(tbl, use_container_width=True)

    pvals = [r.get("p_value") for r in rows if isinstance(r.get("p_value"), float)]
    if pvals:
        corr = multiple_testing_correction(pvals, method="fdr_bh")
        st.subheader("FDR (Benjamini-Hochberg) düzeltilmiş p-değerleri")
        st.write(corr["adjusted"])


def page_what_if() -> None:
    st.header("What-If Simülasyonu")
    st.caption("Senaryo bazlı sonuç tahminlemesi.")
    if st.session_state.metric_type != "Binary":
        st.info("Bu basit simülasyon binary metrikler içindir.")
    baseline = st.number_input("Baseline conversion", min_value=0.0001, max_value=0.9999, value=float(st.session_state.baseline_rate), step=0.005, format="%.4f")
    uplift = st.number_input("Uplift (mutlak)", value=0.01, step=0.005, format="%.4f")
    n = st.number_input("Kol başına örneklem", min_value=100, value=10000, step=100)
    res = what_if_binary(baseline, uplift, int(n), st.session_state.alpha)
    if "error" in res:
        st.error(res["error"])
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Beklenen p-değer", f"{res['p_value']:.4f}")
    c2.metric("Relatif uplift", f"{res['uplift_rel']*100:.2f}%")
    c3.metric("Güven aralığı", f"[{res['ci_low']:.4f}, {res['ci_high']:.4f}]")


def page_anomaly() -> None:
    st.header("Anomali-Aware İzleme")
    if not require_mapping():
        return
    ts_col = st.session_state.timestamp_col
    if not ts_col:
        st.info("Timestamp kolonu eşleyin.")
        return
    df = st.session_state.df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    metric_col = st.session_state.metric_col
    daily = (
        df.groupby([pd.Grouper(key=ts_col, freq="D"), st.session_state.variant_col])[metric_col]
        .mean()
        .reset_index()
    )
    if daily.empty:
        st.info("Veri yok.")
        return
    fig = px.line(daily, x=ts_col, y=metric_col, color=st.session_state.variant_col, markers=True)
    st.plotly_chart(fig, use_container_width=True)

    rows = []
    for v, g in daily.groupby(st.session_state.variant_col):
        vals = g[metric_col].to_numpy()
        if len(vals) < 5:
            continue
        mu = np.mean(vals)
        sd = np.std(vals, ddof=1)
        if sd == 0:
            continue
        z = (vals - mu) / sd
        for i, zi in enumerate(z):
            if abs(zi) > 3:
                rows.append({"variant": v, "date": g.iloc[i][ts_col], "value": float(vals[i]), "z": float(zi)})
    st.session_state.anomaly_result = rows
    if rows:
        st.warning(f"{len(rows)} anomali noktası tespit edildi.")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.success("Belirgin anomali yok.")


def page_decision() -> None:
    st.header("Karar Motoru & Raporlama")
    res = st.session_state.analysis_result
    guard = st.session_state.guardrail_result
    bayes = st.session_state.bayes_result
    if not res:
        st.info("Önce istatistiksel analizi çalıştırın.")
        return

    if st.session_state.metric_type == "Binary":
        n_req = sample_size_binary(
            st.session_state.baseline_rate,
            st.session_state.mde,
            st.session_state.alpha,
            st.session_state.power,
        )
    else:
        n_req = sample_size_continuous(
            st.session_state.baseline_std,
            st.session_state.mde,
            st.session_state.alpha,
            st.session_state.power,
        )

    n_actual = int(res.get("n_control", 0)) + int(res.get("n_variant", 0))
    n_req_total = n_req * 2
    dec = decision_engine(
        p_value=res.get("p_value"),
        uplift_abs=res.get("uplift_abs"),
        uplift_rel=res.get("uplift_rel"),
        alpha=st.session_state.alpha,
        guardrail_violated=bool(guard["violated"]) if guard else False,
        n_required=n_req_total,
        n_actual=n_actual,
        bayes_prob=(bayes or {}).get("prob_variant_better"),
    )
    st.session_state.decision_result = dec

    st.subheader("Karar")
    st.success(dec["decision"])
    for r in dec["reasons"]:
        st.write(f"- {r}")

    summary = build_executive_summary(res, dec, st.session_state.hypothesis)
    st.subheader("Executive Summary")
    st.code(summary)

    st.subheader("LLM Açıklaması (Streamlit Cloud Secrets)")
    if st.button("LLM ile doğal dilde açıklama üret"):
        with st.spinner("Üretiliyor..."):
            explanation = llm_explain(summary)
        if explanation is None:
            st.warning(
                "LLM istemcisi bulunamadı. Streamlit Cloud Settings > Secrets altında "
                "LLM_API_KEY, LLM_BASE_URL ve LLM_MODEL değerlerini ekleyin."
            )
        else:
            st.markdown(explanation)

    st.subheader("Export")
    seg_df = st.session_state.segment_result if isinstance(st.session_state.segment_result, pd.DataFrame) else None
    excel_bytes = export_excel(
        st.session_state.hypothesis,
        res,
        dec,
        st.session_state.validation_report,
        bayes,
        seg_df,
    )
    st.download_button("Excel indir", data=excel_bytes, file_name="ab_test_report.xlsx")

    payload = {
        "hypothesis": st.session_state.hypothesis,
        "metric_type": st.session_state.metric_type,
        "alpha": st.session_state.alpha,
        "power": st.session_state.power,
        "result": res,
        "decision": dec,
        "validation": st.session_state.validation_report,
        "bayesian": bayes,
        "guardrail": guard,
        "sequential": st.session_state.sequential_result,
        "cuped": {k: v for k, v in (st.session_state.cuped_result or {}).items() if k != "adjusted"} if st.session_state.cuped_result else None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    st.download_button("JSON indir", data=export_json(payload), file_name="ab_test_report.json")

    if st.button("Bilgi bankasına kaydet"):
        st.session_state.history.append(
            {
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "hypothesis": st.session_state.hypothesis,
                "metric_type": st.session_state.metric_type,
                "decision": dec["decision"],
                "p_value": res.get("p_value"),
                "uplift_rel": res.get("uplift_rel"),
                "file": st.session_state.file_name,
            }
        )
        st.success("Kaydedildi.")


def page_history() -> None:
    st.header("Deney Bilgi Bankası")
    hist = st.session_state.history
    if not hist:
        st.info("Henüz kaydedilmiş deney yok.")
        return
    st.dataframe(pd.DataFrame(hist), use_container_width=True)
    st.caption("Not: Bu bilgi bankası oturum bazlıdır. Kalıcı saklama için dış veritabanı entegre edin.")


PAGES = {
    "1. Deney Tasarımı": page_experiment_setup,
    "2. Veri Yükleme & Eşleme": page_upload,
    "3. Veri Validasyonu": page_validation,
    "4. İstatistiksel Analiz": page_statistical_analysis,
    "5. Guardrail İzleme": page_guardrail,
    "6. Bayesian Analiz": page_bayesian,
    "7. Sequential / Peeking": page_sequential,
    "8. CUPED & Düzeltmeler": page_cuped_correction,
    "9. Segment Analizi": page_segment,
    "10. What-If Simülasyonu": page_what_if,
    "11. Anomali İzleme": page_anomaly,
    "12. Karar & Rapor": page_decision,
    "13. Deney Bilgi Bankası": page_history,
}


def main() -> None:
    choice = sidebar_nav()
    st.title("A/B Testing Agent")
    st.caption(
        "Deney tasarımından karar önerisine, uçtan uca standartlaştırılmış A/B test akışı."
    )
    st.warning(
        "Bu uygulama bir analiz ve karar destek aracıdır. Üretilen sonuçlar iş kararlarını "
        "destekler niteliktedir; veri kalitesi, dış faktörler ve alan uzmanlığı ile birlikte "
        "değerlendirilmelidir."
    )
    PAGES[choice]()


main()
