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
    try:
        api_key = st.secrets.get("LLM_API_KEY", "") if hasattr(st, "secrets") else ""
        base_url = st.secrets.get("LLM_BASE_URL", "") if hasattr(st, "secrets") else ""
        model = st.secrets.get("LLM_MODEL", "") if hasattr(st, "secrets") else ""
    except Exception:
        api_key = ""
        base_url = ""
        model = ""
    if not api_key:
        return None, None
    try:
        client = OpenAI(api_key=api_key, base_url=base_url or None)
        return client, model
    except Exception:
        return None, None


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
                        "Teknik jargonu minimuma indir; emoji kullanma."
                    ),
                },
                {"role": "user", "content": summary_text},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"LLM açıklaması üretilemedi: {exc}"


_DEFAULTS = {
    "step": 1,
    "hypothesis": "",
    "metric_kind": "Dönüşüm / Tıklama / Evet-Hayır",
    "alpha": 0.05,
    "power": 0.80,
    "mde_hint": 0.02,
    "df": None,
    "file_name": None,
    "mapping": None,
    "auto_mapping": None,
    "full_result": None,
    "history": [],
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


def run_t_test(c: np.ndarray, v: np.ndarray, alpha: float) -> Dict:
    if len(c) < 2 or len(v) < 2:
        return {"error": "Her grupta en az 2 gözlem olmalı."}
    t_stat, p_value = stats.ttest_ind(v, c, equal_var=False)
    mean_c = float(np.mean(c))
    mean_v = float(np.mean(v))
    sd_c = float(np.std(c, ddof=1))
    sd_v = float(np.std(v, ddof=1))
    n_c, n_v = len(c), len(v)
    se_diff = math.sqrt(sd_c ** 2 / n_c + sd_v ** 2 / n_v)
    denom = (sd_c ** 2 / n_c) ** 2 / (n_c - 1) + (sd_v ** 2 / n_v) ** 2 / (n_v - 1)
    df = (sd_c ** 2 / n_c + sd_v ** 2 / n_v) ** 2 / denom if denom > 0 else n_c + n_v - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    diff = mean_v - mean_c
    ci_low = diff - t_crit * se_diff
    ci_high = diff + t_crit * se_diff
    pooled_sd = math.sqrt(((n_c - 1) * sd_c ** 2 + (n_v - 1) * sd_v ** 2) / (n_c + n_v - 2))
    cohen_d = diff / pooled_sd if pooled_sd > 0 else np.nan
    uplift_rel = diff / mean_c if mean_c != 0 else np.nan
    return {
        "method": "Welch's t-test",
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


def bayesian_binary(c_success: int, c_n: int, v_success: int, v_n: int, draws: int = 30000) -> Dict:
    rng = np.random.default_rng(42)
    post_c = rng.beta(1 + c_success, 1 + c_n - c_success, size=draws)
    post_v = rng.beta(1 + v_success, 1 + v_n - v_success, size=draws)
    diff = post_v - post_c
    return {
        "prob_variant_better": float(np.mean(post_v > post_c)),
        "expected_lift_abs": float(np.mean(diff)),
        "ci_low": float(np.percentile(diff, 2.5)),
        "ci_high": float(np.percentile(diff, 97.5)),
        "expected_loss_variant": float(np.mean(np.maximum(post_c - post_v, 0))),
    }


def bayesian_continuous(c: np.ndarray, v: np.ndarray, draws: int = 20000) -> Dict:
    if len(c) < 2 or len(v) < 2:
        return {"prob_variant_better": float("nan"), "expected_lift_abs": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "expected_loss_variant": float("nan")}
    rng = np.random.default_rng(42)
    post_c = rng.normal(np.mean(c), np.std(c, ddof=1) / math.sqrt(len(c)), size=draws)
    post_v = rng.normal(np.mean(v), np.std(v, ddof=1) / math.sqrt(len(v)), size=draws)
    diff = post_v - post_c
    return {
        "prob_variant_better": float(np.mean(post_v > post_c)),
        "expected_lift_abs": float(np.mean(diff)),
        "ci_low": float(np.percentile(diff, 2.5)),
        "ci_high": float(np.percentile(diff, 97.5)),
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
    }


def detect_outliers_iqr(x: np.ndarray) -> int:
    if len(x) < 4:
        return 0
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return int(np.sum((x < low) | (x > high)))


def cuped_adjust(pre: np.ndarray, post: np.ndarray) -> Dict:
    if len(pre) < 10 or np.var(pre, ddof=1) == 0:
        return {"error": "CUPED için yeterli veri yok."}
    mu_pre = float(np.mean(pre))
    var_pre = float(np.var(pre, ddof=1))
    cov = float(np.cov(pre, post, ddof=1)[0, 1])
    theta = cov / var_pre
    adjusted = post - theta * (pre - mu_pre)
    var_reduction = 1 - (np.var(adjusted, ddof=1) / np.var(post, ddof=1))
    return {
        "theta": theta,
        "variance_reduction_pct": float(var_reduction * 100),
        "adjusted": adjusted,
    }


def multiple_testing_correction(p_values: List[float], method: str = "fdr_bh") -> List[float]:
    p_arr = np.array(p_values, dtype=float)
    m = len(p_arr)
    if m == 0:
        return []
    if method == "bonferroni":
        return np.minimum(p_arr * m, 1.0).tolist()
    order = np.argsort(p_arr)
    ranked = p_arr[order]
    adj_ranked = ranked * m / (np.arange(1, m + 1))
    adj_ranked = np.minimum.accumulate(adj_ranked[::-1])[::-1]
    adj = np.empty_like(adj_ranked)
    adj[order] = np.minimum(adj_ranked, 1.0)
    return adj.tolist()


def decision_engine(
    p_value: Optional[float],
    uplift_abs: Optional[float],
    alpha: float,
    guardrail_violated: bool,
    n_required: int,
    n_actual: int,
) -> Dict:
    reasons: List[str] = []
    if n_actual < n_required * 0.7:
        reasons.append(
            f"Örneklem henüz yeterli değil (mevcut {n_actual:,}, hedef ≈ {n_required:,})."
        )
        return {"decision": "Continue Test", "reasons": reasons}
    if guardrail_violated:
        reasons.append("Yan metriklerde (guardrail) olumsuz sapma görülüyor.")
        return {"decision": "Do Not Launch", "reasons": reasons}
    if p_value is None:
        return {"decision": "Re-run Experiment", "reasons": ["İstatistik sonucu üretilemedi."]}
    if p_value < alpha and (uplift_abs is not None and uplift_abs > 0):
        reasons.append("Sonuç istatistiksel olarak anlamlı ve varyant kontrolden daha iyi.")
        return {"decision": "Ship Winner", "reasons": reasons}
    if p_value < alpha and (uplift_abs is not None and uplift_abs < 0):
        reasons.append("Varyant anlamlı şekilde kontrolden daha kötü performans gösterdi.")
        return {"decision": "Do Not Launch", "reasons": reasons}
    if p_value >= alpha and n_actual >= n_required:
        reasons.append("Örneklem yeterli; ancak anlamlı bir fark görülmüyor.")
        return {"decision": "No Significant Difference", "reasons": reasons}
    reasons.append("Sonuç kararsız, testin bir süre daha çalıştırılması önerilir.")
    return {"decision": "Continue Test", "reasons": reasons}


def confidence_label(p_value: float) -> str:
    if p_value < 0.001:
        return "Çok yüksek güven (%99.9+)"
    if p_value < 0.01:
        return "Yüksek güven (%99)"
    if p_value < 0.05:
        return "Yeterli güven (%95)"
    if p_value < 0.1:
        return "Düşük güven (%90)"
    return "Güven seviyesi yetersiz"


def guardrail_keywords() -> List[str]:
    return [
        "bounce", "error", "refund", "churn", "unsubscribe", "unsub",
        "complaint", "return", "iade", "cikis", "hata", "sikayet",
    ]


def autodetect_columns(df: pd.DataFrame, metric_kind: str) -> Dict[str, Optional[str]]:
    lower = {c.lower(): c for c in df.columns}

    def find(keywords: List[str]) -> Optional[str]:
        for k in keywords:
            for lc, orig in lower.items():
                if k in lc:
                    return orig
        return None

    binary_keys = ["convert", "click", "purchase", "signup", "success", "donusum", "tiklama", "satin"]
    continuous_keys = ["revenue", "value", "amount", "duration", "gelir", "ciro", "tutar", "sure"]

    variant = find(["variant", "group", "arm", "grup", "varyant", "treatment", "test_group"])
    timestamp = find(["timestamp", "date", "time", "tarih", "created_at", "occurred"])
    segment = find(["segment", "device", "country", "channel", "platform", "plan", "region"])
    pre_metric = find(["pre_", "previous_", "prev_", "before_", "onceki_", "onki_"])
    if metric_kind == "Dönüşüm / Tıklama / Evet-Hayır":
        metric = find(binary_keys)
        if metric is None:
            for c in df.columns:
                if c == variant or c == timestamp:
                    continue
                s = pd.to_numeric(df[c], errors="coerce").dropna()
                if len(s) > 0 and set(s.unique().tolist()).issubset({0, 1}):
                    metric = c
                    break
    else:
        metric = find(continuous_keys)
        if metric is None:
            for c in df.columns:
                if c in {variant, timestamp, segment, pre_metric}:
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    s = pd.to_numeric(df[c], errors="coerce").dropna()
                    if len(s) > 0 and not set(s.unique().tolist()).issubset({0, 1}):
                        metric = c
                        break
    return {"variant": variant, "metric": metric, "timestamp": timestamp, "segment": segment, "pre_metric": pre_metric}


def pick_control_label(variants: List[str]) -> str:
    priority = ["control", "kontrol", "a", "baseline"]
    lower = {v.lower(): v for v in variants}
    for p in priority:
        if p in lower:
            return lower[p]
    return sorted(variants)[0]


def run_full_analysis(
    df: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
    metric_kind: str,
    alpha: float,
    power: float,
    mde_hint: float,
) -> Dict:
    variant_col = mapping["variant"]
    metric_col = mapping["metric"]
    segment_col = mapping.get("segment")
    pre_metric_col = mapping.get("pre_metric")
    timestamp_col = mapping.get("timestamp")

    is_binary = metric_kind == "Dönüşüm / Tıklama / Evet-Hayır"

    variant_series = df[variant_col].astype(str)
    counts_dict = variant_series.value_counts().to_dict()
    all_variants = list(counts_dict.keys())
    if len(all_variants) < 2:
        return {"error": "Veri setinde en az 2 varyant bulunamadı."}
    control = pick_control_label(all_variants)
    variant_labels = [v for v in all_variants if v != control]

    expected = {k: 1 / len(all_variants) for k in all_variants}
    srm = srm_check(counts_dict, expected)

    guard_cols = [
        c for c in df.columns
        if c not in {variant_col, metric_col, timestamp_col}
        and any(kw in c.lower() for kw in guardrail_keywords())
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    analyses: List[Dict] = []
    for v in variant_labels:
        sub_c = df[variant_series == control][metric_col]
        sub_v = df[variant_series == v][metric_col]
        c_num = pd.to_numeric(sub_c, errors="coerce").dropna()
        v_num = pd.to_numeric(sub_v, errors="coerce").dropna()

        if is_binary:
            stat = run_z_test(int(c_num.sum()), int(len(c_num)), int(v_num.sum()), int(len(v_num)), alpha)
            bayes = bayesian_binary(int(c_num.sum()), int(len(c_num)), int(v_num.sum()), int(len(v_num)))
        else:
            stat = run_t_test(c_num.to_numpy(dtype=float), v_num.to_numpy(dtype=float), alpha)
            bayes = bayesian_continuous(c_num.to_numpy(dtype=float), v_num.to_numpy(dtype=float))

        if "error" in stat:
            analyses.append({"variant": v, "error": stat["error"]})
            continue

        if is_binary:
            base_rate = float(stat.get("p_control", 0.1)) or 0.1
            n_req = sample_size_binary(base_rate, max(mde_hint, 0.001), alpha, power)
        else:
            sd = (float(stat.get("sd_control", 0)) + float(stat.get("sd_variant", 0))) / 2
            baseline_mean = float(stat.get("mean_control", 1.0)) or 1.0
            n_req = sample_size_continuous(sd, max(mde_hint * abs(baseline_mean), 0.001), alpha, power)
        n_actual = int(stat.get("n_control", 0)) + int(stat.get("n_variant", 0))
        n_req_total = n_req * 2

        guard_rows = []
        guard_violated = False
        for gc in guard_cols:
            c_mean = float(df[variant_series == control][gc].mean())
            v_mean = float(df[variant_series == v][gc].mean())
            if math.isnan(c_mean) or c_mean == 0:
                rel = 0.0
            else:
                rel = (v_mean - c_mean) / abs(c_mean)
            bad = rel > 0.02
            if bad:
                guard_violated = True
            guard_rows.append({
                "metric": gc,
                "control_mean": c_mean,
                "variant_mean": v_mean,
                "relative_change": rel,
                "violated": bad,
            })

        dec = decision_engine(
            p_value=stat.get("p_value"),
            uplift_abs=stat.get("uplift_abs"),
            alpha=alpha,
            guardrail_violated=guard_violated,
            n_required=n_req_total,
            n_actual=n_actual,
        )

        segment_rows: List[Dict] = []
        if segment_col:
            sub_v_full = df[variant_series.isin([control, v])].copy()
            for seg, g in sub_v_full.groupby(segment_col, dropna=False):
                gc_arr = pd.to_numeric(g[g[variant_col].astype(str) == control][metric_col], errors="coerce").dropna()
                gv_arr = pd.to_numeric(g[g[variant_col].astype(str) == v][metric_col], errors="coerce").dropna()
                if len(gc_arr) < 10 or len(gv_arr) < 10:
                    continue
                if is_binary:
                    r = run_z_test(int(gc_arr.sum()), int(len(gc_arr)), int(gv_arr.sum()), int(len(gv_arr)), alpha)
                else:
                    r = run_t_test(gc_arr.to_numpy(dtype=float), gv_arr.to_numpy(dtype=float), alpha)
                if "error" in r:
                    continue
                segment_rows.append({
                    "segment": str(seg),
                    "n_control": r.get("n_control"),
                    "n_variant": r.get("n_variant"),
                    "uplift_rel": r.get("uplift_rel"),
                    "p_value": r.get("p_value"),
                })
            if segment_rows:
                adj = multiple_testing_correction([r["p_value"] for r in segment_rows], method="fdr_bh")
                for row, p_adj in zip(segment_rows, adj):
                    row["p_value_fdr"] = p_adj
                    row["category"] = "Confirmed" if p_adj < alpha else "Exploratory"

        cuped_info = None
        if pre_metric_col and not is_binary:
            mask = df[variant_series.isin([control, v])].index
            pre = pd.to_numeric(df.loc[mask, pre_metric_col], errors="coerce").to_numpy(dtype=float)
            post = pd.to_numeric(df.loc[mask, metric_col], errors="coerce").to_numpy(dtype=float)
            keep = ~(np.isnan(pre) | np.isnan(post))
            if keep.sum() > 10:
                cuped = cuped_adjust(pre[keep], post[keep])
                if "error" not in cuped:
                    cuped_info = {"variance_reduction_pct": cuped["variance_reduction_pct"], "theta": cuped["theta"]}

        analyses.append({
            "variant": v,
            "stat": stat,
            "bayes": bayes,
            "guardrail": {"violated": guard_violated, "rows": guard_rows},
            "n_required": n_req_total,
            "n_actual": n_actual,
            "decision": dec,
            "segments": segment_rows,
            "cuped": cuped_info,
        })

    quality = {
        "srm": srm,
        "duplicates": int(df.duplicated().sum()),
        "missing_variant": int(df[variant_col].isna().sum()),
        "missing_metric": int(df[metric_col].isna().sum()),
        "outliers": detect_outliers_iqr(pd.to_numeric(df[metric_col], errors="coerce").dropna().to_numpy()) if pd.api.types.is_numeric_dtype(df[metric_col]) else 0,
    }

    trend = None
    if timestamp_col:
        try:
            tmp = df[[timestamp_col, variant_col, metric_col]].copy()
            tmp[timestamp_col] = pd.to_datetime(tmp[timestamp_col], errors="coerce")
            tmp = tmp.dropna(subset=[timestamp_col])
            if not tmp.empty:
                tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
                trend = (
                    tmp.groupby([pd.Grouper(key=timestamp_col, freq="D"), variant_col])[metric_col]
                    .mean()
                    .reset_index()
                )
        except Exception:
            trend = None

    return {
        "metric_kind": metric_kind,
        "is_binary": is_binary,
        "alpha": alpha,
        "power": power,
        "control": control,
        "variants": variant_labels,
        "variant_col": variant_col,
        "metric_col": metric_col,
        "analyses": analyses,
        "quality": quality,
        "trend": trend,
    }


def export_excel(hypothesis: str, result: Dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "field": ["hypothesis", "metric_kind", "alpha", "power", "timestamp"],
                "value": [
                    hypothesis,
                    result.get("metric_kind"),
                    result.get("alpha"),
                    result.get("power"),
                    datetime.now().isoformat(timespec="seconds"),
                ],
            }
        ).to_excel(writer, sheet_name="Summary", index=False)
        rows: List[Dict] = []
        for a in result.get("analyses", []):
            if "error" in a:
                continue
            st_ = a["stat"]
            rows.append(
                {
                    "variant": a["variant"],
                    "decision": a["decision"]["decision"],
                    "p_value": st_.get("p_value"),
                    "uplift_rel": st_.get("uplift_rel"),
                    "uplift_abs": st_.get("uplift_abs"),
                    "ci_low": st_.get("ci_low"),
                    "ci_high": st_.get("ci_high"),
                    "n_control": st_.get("n_control"),
                    "n_variant": st_.get("n_variant"),
                    "prob_variant_better": a["bayes"].get("prob_variant_better"),
                }
            )
        if rows:
            pd.DataFrame(rows).to_excel(writer, sheet_name="Results", index=False)
        for a in result.get("analyses", []):
            if "error" in a:
                continue
            g_rows = a["guardrail"]["rows"]
            if g_rows:
                pd.DataFrame(g_rows).to_excel(writer, sheet_name=f"Guardrail_{a['variant'][:20]}", index=False)
            if a.get("segments"):
                pd.DataFrame(a["segments"]).to_excel(writer, sheet_name=f"Segments_{a['variant'][:20]}", index=False)
        q = result.get("quality", {})
        if q:
            pd.DataFrame([
                {"check": "SRM p-value", "value": q.get("srm", {}).get("p_value")},
                {"check": "Duplicates", "value": q.get("duplicates")},
                {"check": "Missing variant", "value": q.get("missing_variant")},
                {"check": "Missing metric", "value": q.get("missing_metric")},
                {"check": "Outliers", "value": q.get("outliers")},
            ]).to_excel(writer, sheet_name="DataQuality", index=False)
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
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="records")
        return str(o)
    return json.dumps(payload, indent=2, ensure_ascii=False, default=_default).encode("utf-8")


def build_summary_text(hypothesis: str, result: Dict) -> str:
    lines = [
        "A/B TEST ÖZETİ",
        f"Hipotez: {hypothesis or '-'}",
        f"Metrik tipi: {result.get('metric_kind')}",
        f"Kontrol: {result.get('control')}",
    ]
    for a in result.get("analyses", []):
        if "error" in a:
            lines.append(f"- {a['variant']}: {a['error']}")
            continue
        s = a["stat"]
        lines.append("")
        lines.append(f"Varyant: {a['variant']}")
        lines.append(f"Karar: {a['decision']['decision']}")
        if "p_value" in s:
            lines.append(f"p-değer: {s['p_value']:.4f}")
        if s.get("uplift_rel") is not None and not (isinstance(s["uplift_rel"], float) and math.isnan(s["uplift_rel"])):
            lines.append(f"Relatif fark: {s['uplift_rel']*100:+.2f}%")
        lines.append(f"Örneklem: kontrol={s.get('n_control', 0):,}, varyant={s.get('n_variant', 0):,}")
        for r in a["decision"]["reasons"]:
            lines.append(f"- {r}")
    return "\n".join(lines)


def go_to(step: int) -> None:
    st.session_state.step = step
    safe_rerun()


def reset_all() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    safe_rerun()


def sidebar() -> None:
    st.sidebar.title("A/B Testing Agent")
    st.sidebar.caption("3 adımda kolay A/B test")
    current = st.session_state.step
    steps = [(1, "Deneyim"), (2, "Veri"), (3, "Sonuç")]
    for i, label in steps:
        if i == current:
            st.sidebar.markdown(f"**{i}. {label} — şu an**")
        elif i < current:
            st.sidebar.markdown(f"{i}. {label} (tamamlandı)")
        else:
            st.sidebar.markdown(f"{i}. {label}")
    st.sidebar.divider()
    if st.sidebar.button("Yeni deney başlat", use_container_width=True):
        reset_all()
    with st.sidebar.expander("Yardım"):
        st.markdown(
            "1. **Deneyim**: Neyi test ettiğinizi ve başarıyı nasıl ölçtüğünüzü söyleyin.\n"
            "2. **Veri**: CSV/XLSX dosyanızı yükleyin; kolon eşlemesini sistem otomatik yapar.\n"
            "3. **Sonuç**: Karar, etki ve güven seviyesi tek kartta. Detaylar expander'larda."
        )


def page1_setup() -> None:
    st.header("1. Deneyim hakkında")
    st.caption("Neyi test ettiğinizi ve nasıl ölçtüğünüzü bize söyleyin. Teknik bilgi gerekmez.")

    st.session_state.hypothesis = st.text_area(
        "Ne test ediyorsunuz?",
        value=st.session_state.hypothesis,
        placeholder="Örn: Yeni ana sayfa tasarımı dönüşüm oranını artıracak.",
        height=90,
    )

    st.session_state.metric_kind = st.radio(
        "Başarıyı nasıl ölçüyorsunuz?",
        [
            "Dönüşüm / Tıklama / Evet-Hayır",
            "Gelir / Süre / Sayısal değer",
        ],
        index=0 if st.session_state.metric_kind.startswith("Dönüşüm") else 1,
        help="Dönüşüm: kullanıcı bir eylemi yaptı mı (evet/hayır). Gelir/Süre: sayısal bir değer.",
    )

    default_mde_pct = int(round(st.session_state.mde_hint * 100))
    if st.session_state.metric_kind.startswith("Dönüşüm"):
        mde_pct = st.slider(
            "En az ne kadarlık bir iyileşme anlamlı sayılır?",
            min_value=1, max_value=20, value=max(1, default_mde_pct),
            help="Ör: Mevcut dönüşüm %10 ise ve 2 puanlık bir artışı anlamlı buluyorsanız 2 seçin.",
        )
        st.session_state.mde_hint = mde_pct / 100.0
        st.caption(f"Seçilen duyarlılık: +{mde_pct} puan mutlak değişim.")
    else:
        mde_pct = st.slider(
            "En az yüzde kaç iyileşme anlamlı sayılır?",
            min_value=1, max_value=30, value=max(1, default_mde_pct),
            help="Örn: Kontrol ortalaması 100 TL ise %5'lik bir artış 5 TL'dir.",
        )
        st.session_state.mde_hint = mde_pct / 100.0
        st.caption(f"Seçilen duyarlılık: %{mde_pct} relatif değişim.")

    with st.expander("Gelişmiş ayarlar (opsiyonel)"):
        st.session_state.alpha = st.select_slider(
            "Güven seviyesi",
            options=[0.01, 0.05, 0.10],
            value=st.session_state.alpha,
            format_func=lambda a: f"%{int((1-a)*100)} güven",
        )
        st.session_state.power = st.select_slider(
            "İstatistiksel güç (power)",
            options=[0.70, 0.80, 0.90, 0.95],
            value=st.session_state.power,
            format_func=lambda p: f"{int(p*100)}%",
        )
        st.caption("Varsayılan %95 güven ve %80 güç, çoğu test için uygundur.")

    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("Devam et", type="primary", use_container_width=True):
            go_to(2)


def page2_data() -> None:
    st.header("2. Veri")
    st.caption("CSV veya XLSX dosyanızı yükleyin. Kolonları sizin için otomatik eşleyeceğiz.")

    uploaded = st.file_uploader("Dosya seç", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        df = load_dataframe(uploaded)
        if df is None:
            st.error("Dosya okunamadı.")
            return
        st.session_state.df = df
        st.session_state.file_name = uploaded.name
        st.session_state.auto_mapping = autodetect_columns(df, st.session_state.metric_kind)
        st.session_state.mapping = dict(st.session_state.auto_mapping)

    df = st.session_state.df
    if df is None:
        st.info("Yukarıdan bir dosya yükleyin.")
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("Geri", use_container_width=True):
                go_to(1)
        return

    st.success(f"Yüklendi: {st.session_state.file_name} — {len(df):,} satır, {df.shape[1]} kolon")

    st.subheader("Önizleme")
    st.dataframe(df.head(20), use_container_width=True)

    auto = st.session_state.auto_mapping or {}
    mapping = st.session_state.mapping or {}

    st.subheader("Otomatik algılanan kolonlar")
    auto_summary = {
        "Varyant (grup)": mapping.get("variant") or "(bulunamadı)",
        "Başarı metriği": mapping.get("metric") or "(bulunamadı)",
        "Zaman": mapping.get("timestamp") or "(yok)",
        "Segment": mapping.get("segment") or "(yok)",
        "Önceki dönem metriği": mapping.get("pre_metric") or "(yok)",
    }
    for k, v in auto_summary.items():
        st.markdown(f"- **{k}**: {v}")

    with st.expander("Manuel ayarla"):
        cols = list(df.columns)
        none_opts = ["(yok)"] + cols
        def _idx(val: Optional[str], opts: List[str]) -> int:
            if val and val in opts:
                return opts.index(val)
            return 0
        mapping["variant"] = st.selectbox("Varyant kolonu", cols, index=_idx(mapping.get("variant"), cols))
        mapping["metric"] = st.selectbox("Metric kolonu", cols, index=_idx(mapping.get("metric"), cols))
        ts_sel = st.selectbox("Zaman kolonu", none_opts, index=_idx(mapping.get("timestamp"), none_opts))
        mapping["timestamp"] = None if ts_sel == "(yok)" else ts_sel
        seg_sel = st.selectbox("Segment kolonu", none_opts, index=_idx(mapping.get("segment"), none_opts))
        mapping["segment"] = None if seg_sel == "(yok)" else seg_sel
        pre_sel = st.selectbox("Önceki dönem metriği (CUPED için)", none_opts, index=_idx(mapping.get("pre_metric"), none_opts))
        mapping["pre_metric"] = None if pre_sel == "(yok)" else pre_sel
        st.session_state.mapping = mapping

    ready = bool(mapping.get("variant") and mapping.get("metric"))
    if not ready:
        st.warning("Devam etmek için en az varyant ve metric kolonları seçilmeli.")

    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        if st.button("Geri", use_container_width=True):
            go_to(1)
    with c2:
        if st.button("Analizi başlat", type="primary", disabled=not ready, use_container_width=True):
            with st.spinner("Analiz çalışıyor..."):
                try:
                    result = run_full_analysis(
                        df, st.session_state.mapping,
                        st.session_state.metric_kind,
                        st.session_state.alpha,
                        st.session_state.power,
                        st.session_state.mde_hint,
                    )
                except Exception as exc:
                    st.error(f"Analiz hatası: {exc}")
                    return
            if "error" in result:
                st.error(result["error"])
                return
            st.session_state.full_result = result
            go_to(3)


def decision_card(decision_text: str) -> None:
    mapping = {
        "Ship Winner": ("Varyant kazandı. Yayına alınabilir.", "success"),
        "Do Not Launch": ("Varyant yayına alınmamalı.", "error"),
        "Continue Test": ("Kesin karar için teste devam edilmeli.", "info"),
        "No Significant Difference": ("Kontrol ile varyant arasında anlamlı fark yok.", "warning"),
        "Re-run Experiment": ("Testi yeniden kurgulayın.", "warning"),
    }
    text, kind = mapping.get(decision_text, ("Sonuç belirsiz", "info"))
    title = f"Sonuç: {text}"
    if kind == "success":
        st.success(title)
    elif kind == "error":
        st.error(title)
    elif kind == "warning":
        st.warning(title)
    else:
        st.info(title)


def plot_trend(trend: pd.DataFrame, variant_col: str, metric_col: str) -> None:
    if trend is None or trend.empty:
        return
    fig = px.line(trend, x=trend.columns[0], y=metric_col, color=variant_col, markers=True)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(fig, use_container_width=True)


def plot_comparison(stat: Dict, is_binary: bool, control: str, variant: str) -> None:
    if is_binary:
        labels = [control, variant]
        values = [stat.get("p_control", 0) * 100, stat.get("p_variant", 0) * 100]
        fig = go.Figure(go.Bar(x=labels, y=values, text=[f"%{v:.2f}" for v in values], textposition="outside"))
        fig.update_layout(yaxis_title="Oran (%)", margin=dict(l=10, r=10, t=10, b=10), height=300)
    else:
        labels = [control, variant]
        values = [stat.get("mean_control", 0), stat.get("mean_variant", 0)]
        fig = go.Figure(go.Bar(x=labels, y=values, text=[f"{v:.2f}" for v in values], textposition="outside"))
        fig.update_layout(yaxis_title="Ortalama", margin=dict(l=10, r=10, t=10, b=10), height=300)
    st.plotly_chart(fig, use_container_width=True)


def page3_results() -> None:
    st.header("3. Sonuç")
    result = st.session_state.full_result
    if not result:
        st.info("Önce analiz çalıştırın.")
        if st.button("Geri"):
            go_to(2)
        return

    st.caption(f"Hipotez: {st.session_state.hypothesis or '-'}")

    tabs = st.tabs([f"Varyant: {a['variant']}" for a in result["analyses"]])
    for tab, a in zip(tabs, result["analyses"]):
        with tab:
            if "error" in a:
                st.error(a["error"])
                continue

            stat = a["stat"]
            decision_card(a["decision"]["decision"])

            cols = st.columns(4)
            if result["is_binary"]:
                cols[0].metric(f"{result['control']} dönüşüm", f"%{stat['p_control']*100:.2f}")
                cols[1].metric(f"{a['variant']} dönüşüm", f"%{stat['p_variant']*100:.2f}")
                if not (isinstance(stat.get("uplift_rel"), float) and math.isnan(stat["uplift_rel"])):
                    cols[2].metric("Relatif fark", f"{stat['uplift_rel']*100:+.2f}%")
                cols[3].metric("Güven", confidence_label(stat["p_value"]))
            else:
                cols[0].metric(f"{result['control']} ortalama", f"{stat['mean_control']:.2f}")
                cols[1].metric(f"{a['variant']} ortalama", f"{stat['mean_variant']:.2f}")
                if not (isinstance(stat.get("uplift_rel"), float) and math.isnan(stat["uplift_rel"])):
                    cols[2].metric("Relatif fark", f"{stat['uplift_rel']*100:+.2f}%")
                cols[3].metric("Güven", confidence_label(stat["p_value"]))

            sample_pct = min(100, int(a["n_actual"] / max(a["n_required"], 1) * 100))
            st.progress(sample_pct / 100, text=f"Örneklem doluluğu: {a['n_actual']:,} / {a['n_required']:,} (%{sample_pct})")

            plot_comparison(stat, result["is_binary"], result["control"], a["variant"])

            st.markdown("**Ne yapmalıyım?**")
            for r in a["decision"]["reasons"]:
                st.markdown(f"- {r}")

            with st.expander("Veri kalitesi kontrolleri"):
                q = result["quality"]
                srm = q.get("srm", {})
                c1, c2, c3, c4 = st.columns(4)
                if "srm_detected" in srm:
                    c1.metric("SRM", "Sorun var" if srm["srm_detected"] else "Temiz", help=f"p={srm.get('p_value',0):.4f}")
                c2.metric("Duplicate", f"{q.get('duplicates', 0):,}")
                c3.metric("Eksik metric", f"{q.get('missing_metric', 0):,}")
                c4.metric("Aykırı değer", f"{q.get('outliers', 0):,}")

            with st.expander("Yan metrikler (guardrail)"):
                rows = a["guardrail"]["rows"]
                if not rows:
                    st.info("Veride otomatik yakalanan guardrail metriği yok.")
                else:
                    df_g = pd.DataFrame(rows)
                    df_g["relative_change"] = df_g["relative_change"].map(lambda x: f"{x*100:+.2f}%")
                    st.dataframe(df_g, use_container_width=True)
                    if a["guardrail"]["violated"]:
                        st.warning("En az bir guardrail metriğinde olumsuz yön tespit edildi.")
                    else:
                        st.success("Guardrail metriklerinde olumsuz sinyal yok.")

            with st.expander("Segment bazında performans"):
                if not a.get("segments"):
                    st.info("Segment kolonu seçilmedi veya yeterli veri yok.")
                else:
                    df_s = pd.DataFrame(a["segments"]).copy()
                    if "uplift_rel" in df_s:
                        df_s["uplift_rel"] = df_s["uplift_rel"].map(
                            lambda x: f"{x*100:+.2f}%" if isinstance(x, (int, float)) and not math.isnan(x) else "-"
                        )
                    st.dataframe(df_s, use_container_width=True)

            with st.expander("Varyans azaltma (CUPED)"):
                if not a.get("cuped"):
                    st.info("Önceki dönem metriği sağlanmadığı için CUPED çalıştırılmadı.")
                else:
                    st.metric("Varyans azaltımı", f"%{a['cuped']['variance_reduction_pct']:.1f}")
                    st.caption("CUPED ile test daha az örneklemle aynı hassasiyeti yakalar.")

            with st.expander("Bayesian olasılık"):
                b = a["bayes"]
                bc1, bc2 = st.columns(2)
                bc1.metric("Varyant daha iyi olma olasılığı", f"%{b['prob_variant_better']*100:.1f}")
                bc2.metric("Beklenen kayıp", f"{b['expected_loss_variant']:.4f}")

            with st.expander("Teknik istatistik detayları"):
                techs = {k: v for k, v in stat.items() if not isinstance(v, (list, np.ndarray))}
                st.json(techs)

    if result.get("trend") is not None:
        with st.expander("Zaman içinde metrik değişimi"):
            plot_trend(result["trend"], result["variant_col"], result["metric_col"])

    st.divider()
    st.subheader("Rapor")
    summary = build_summary_text(st.session_state.hypothesis, result)
    st.code(summary)

    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        st.download_button(
            "Excel indir",
            data=export_excel(st.session_state.hypothesis, result),
            file_name="ab_test_report.xlsx",
            use_container_width=True,
        )
    with cc2:
        payload = {
            "hypothesis": st.session_state.hypothesis,
            "metric_kind": result.get("metric_kind"),
            "alpha": result.get("alpha"),
            "power": result.get("power"),
            "analyses": [
                {**{k: v for k, v in a.items() if k != "stat"}, "stat": {k: v for k, v in a.get("stat", {}).items() if not isinstance(v, (list, np.ndarray))}}
                for a in result.get("analyses", [])
            ],
            "quality": result.get("quality"),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        st.download_button(
            "JSON indir",
            data=export_json(payload),
            file_name="ab_test_report.json",
            use_container_width=True,
        )
    with cc3:
        if st.button("Bilgi bankasına kaydet", use_container_width=True):
            for a in result.get("analyses", []):
                if "error" in a:
                    continue
                st.session_state.history.append({
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                    "hypothesis": st.session_state.hypothesis,
                    "metric_kind": result.get("metric_kind"),
                    "variant": a["variant"],
                    "decision": a["decision"]["decision"],
                    "uplift_rel": a["stat"].get("uplift_rel"),
                    "p_value": a["stat"].get("p_value"),
                    "file": st.session_state.file_name,
                })
            st.success("Kaydedildi.")
    with cc4:
        if st.button("Yöneticiye özet (LLM)", use_container_width=True):
            with st.spinner("Hazırlanıyor..."):
                text = llm_explain(summary)
            if text is None:
                st.info(
                    "LLM yapılandırılmamış. Streamlit Cloud → Settings → Secrets altında "
                    "LLM_API_KEY, LLM_BASE_URL ve LLM_MODEL tanımlayın."
                )
            else:
                st.markdown(text)

    if st.session_state.history:
        with st.expander(f"Geçmiş deneyler ({len(st.session_state.history)})"):
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)

    st.divider()
    cb1, cb2 = st.columns([1, 4])
    with cb1:
        if st.button("Yeni deney", use_container_width=True):
            reset_all()


def main() -> None:
    sidebar()
    st.title("A/B Testing Agent")
    st.caption("Teknik detaylara boğulmadan, üç adımda A/B test analizi ve karar.")
    step = st.session_state.step
    if step == 1:
        page1_setup()
    elif step == 2:
        page2_data()
    else:
        page3_results()


main()
