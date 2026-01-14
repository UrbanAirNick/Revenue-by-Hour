import io
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import streamlit as st

# --- Simple Password Gate ---
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "")

if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.title("Revenue Timestamp Tool 🔒")
    pw = st.text_input("Enter password", type="password")

    if st.button("Login"):
        if pw == APP_PASSWORD and pw != "":
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

st.set_page_config(page_title="Revenue Timestamp & Hourly Rollup", layout="wide")

st.title("Revenue Timestamp & Hourly Rollup")
st.caption(
    "Upload the **Revenue Summary Rollup Report.csv** and **Cash Drawer Logs Report.csv**. "
    "Click **Run** to generate matches, unmatched logs, a revenue rollup with approximate times, "
    "and a revenue-by-hour breakdown."
)

with st.sidebar:
    st.header("Inputs")
    rev_file = st.file_uploader("Revenue Summary Rollup Report (.csv)", type=["csv"])
    cash_file = st.file_uploader("Cash Drawer Logs Report (.csv)", type=["csv"])
    run_btn = st.button("Run", type="primary", disabled=not (rev_file and cash_file))

    st.divider()
    st.subheader("Matching & Timing Rules")
    st.write("- Cash drawer ↔ revenue cash transactions are matched by **Confirmation Code** (normalizing `J-` prefix).")
    st.write("- Matched cash timestamps become **anchor times**.")
    st.write("- Other payments on the same day receive **interpolated times** between the nearest anchors, based on ordering.")


def _read_csv(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    # Read bytes and let pandas infer delimiter/encoding; fall back safely.
    data = uploaded_file.getvalue()
    try:
        return pd.read_csv(io.BytesIO(data))
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(data), encoding="latin-1")


def build_outputs(rev: pd.DataFrame, cash: pd.DataFrame):
    # --- Parse/clean ---
    required_rev_cols = {"Transaction Date", "Provider", "Confirmation Code", "Payment ID", "Total Price"}
    required_cash_cols = {"Access Date Time", "Confirmation Code", "Cash Paid"}

    missing_rev = required_rev_cols - set(rev.columns)
    missing_cash = required_cash_cols - set(cash.columns)
    if missing_rev:
        raise ValueError(f"Revenue file is missing required columns: {sorted(missing_rev)}")
    if missing_cash:
        raise ValueError(f"Cash Drawer Logs file is missing required columns: {sorted(missing_cash)}")

    rev = rev.copy()
    cash = cash.copy()

    # Datetimes
    rev["Transaction Date"] = pd.to_datetime(rev["Transaction Date"], errors="coerce")
    cash["Access Date Time"] = pd.to_datetime(cash["Access Date Time"], errors="coerce")

    # Normalize confirmation codes
    cash["ConfNorm"] = cash["Confirmation Code"].astype(str).str.replace("J-", "", regex=False).str.strip()
    rev["ConfNorm"] = rev["Confirmation Code"].astype(str).str.replace("J-", "", regex=False).str.strip()

    # Restrict revenue to cash provider for matching
    rev_provider = rev["Provider"].astype(str).str.lower().str.strip()
    rev_cash = rev[rev_provider.eq("cash")].copy()

    # --- Match: cash drawer logs ↔ revenue cash transactions via ConfNorm ---
    merged = cash.merge(
        rev_cash,
        how="left",
        on="ConfNorm",
        suffixes=("_log", "_rev"),
    )
    merged["match_flag"] = merged["Payment ID"].notna()

    matches = merged[merged["match_flag"]].copy()
    unmatched = merged[~merged["match_flag"]].copy()

    # Tidy match output columns (best-effort; include if present)
    match_cols = [
        "Access Date Time", "Printer", "Requester", "Approver", "Access Reason",
        "Account ID_log", "Account Holder Name", "Transaction Amount", "Cash Paid",
        "Confirmation Code_log",
        "Transaction Date", "Total Price", "Transaction Type", "Provider", "Source",
        "Invoice Number", "Payment ID", "Confirmation Code_rev",
    ]
    present_match_cols = [c for c in match_cols if c in matches.columns]
    matches_out = matches[present_match_cols].copy()
    if "Account ID_log" in matches_out.columns:
        matches_out.rename(columns={"Account ID_log": "Account ID"}, inplace=True)

    # Unmatched output
    unmatched_cols = ["Access Date Time", "Confirmation Code_log", "Account Holder Name", "Cash Paid", "Transaction Amount"]
    present_unmatched_cols = [c for c in unmatched_cols if c in unmatched.columns]
    unmatched_out = unmatched[present_unmatched_cols].copy()
    if "Confirmation Code_log" in unmatched_out.columns:
        unmatched_out.rename(columns={"Confirmation Code_log": "Confirmation Code"}, inplace=True)

    # --- Use matched cash timestamps as anchors to approximate transaction time for ALL revenue rows ---
    anchors = matches[["Payment ID", "Access Date Time"]].dropna().copy()
    # Normalize Payment ID types
    anchors["Payment ID"] = pd.to_numeric(anchors["Payment ID"], errors="coerce")
    anchors = anchors.dropna(subset=["Payment ID"]).copy()
    anchors["Payment ID"] = anchors["Payment ID"].astype("int64")

    rev_all = rev.copy()
    rev_all["Payment ID_num"] = pd.to_numeric(rev_all["Payment ID"], errors="coerce")
    rev_all["Payment ID_int"] = rev_all["Payment ID_num"].dropna().astype("int64")

    rev_all = rev_all.merge(
        anchors.rename(columns={"Access Date Time": "Anchor Time"}),
        how="left",
        left_on="Payment ID_int",
        right_on="Payment ID",
        suffixes=("", "_anchor"),
    )

    # Ordering for interpolation: Invoice Number then Payment ID
    rev_all["Invoice Number_num"] = pd.to_numeric(rev_all.get("Invoice Number"), errors="coerce")
    rev_all["sort_key"] = rev_all["Invoice Number_num"].fillna(rev_all["Payment ID_num"]).fillna(0).astype(float)

    # Initialize approx time with anchor where available
    rev_all["Approx Transaction Time"] = pd.NaT
    rev_all.loc[rev_all["Anchor Time"].notna(), "Approx Transaction Time"] = rev_all.loc[
        rev_all["Anchor Time"].notna(), "Anchor Time"
    ]

    def interpolate_day(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("sort_key").copy()
        t = df["Approx Transaction Time"]
        x = df["sort_key"].astype(float).to_numpy()
        mask = ~t.isna()

        if mask.sum() >= 2:
            y = t.astype("int64").to_numpy()  # ns since epoch; NaT -> min int, but masked out
            y_interp = np.interp(x, x[mask.to_numpy()], y[mask.to_numpy()])
            df["Approx Transaction Time"] = pd.to_datetime(y_interp)
        elif mask.sum() == 1:
            df["Approx Transaction Time"] = t[mask].iloc[0]
        # else: keep NaT
        return df

    # Group by transaction date (date part)
    rev_all = rev_all.dropna(subset=["Transaction Date"]).copy()
    rev_all = rev_all.groupby(rev_all["Transaction Date"].dt.date, group_keys=False).apply(interpolate_day)

    # Revenue by hour
    rev_all["Total Price_num"] = pd.to_numeric(rev_all["Total Price"], errors="coerce").fillna(0.0)
    rev_all["Hour"] = pd.to_datetime(rev_all["Approx Transaction Time"]).dt.floor("h")
    hourly = (
        rev_all.groupby("Hour", as_index=False)["Total Price_num"].sum()
        .rename(columns={"Total Price_num": "Revenue"})
        .sort_values("Hour")
    )

    return matches_out, unmatched_out, rev_all, hourly


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_zip(files: dict) -> bytes:
    # files: {filename: bytes}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    buf.seek(0)
    return buf.read()


if run_btn:
    try:
        rev_df = _read_csv(rev_file)
        cash_df = _read_csv(cash_file)

        with st.spinner("Processing…"):
            matches_out, unmatched_out, rev_all, hourly = build_outputs(rev_df, cash_df)

        st.success("Complete!")

        # --- KPIs ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Revenue rows", f"{len(rev_all):,}")
        c2.metric("Matched cash anchors", f"{len(matches_out):,}")
        c3.metric("Unmatched drawer log entries", f"{len(unmatched_out):,}")
        c4.metric("Hours in rollup", f"{len(hourly):,}")

        st.divider()

        # --- Hourly table + chart ---
        st.subheader("Revenue by Hour")
        st.dataframe(hourly, use_container_width=True, hide_index=True)
        st.line_chart(hourly.set_index("Hour")["Revenue"])

        st.divider()

        # --- Downloadables ---
        st.subheader("Downloads")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        files = {
            f"{timestamp}_cashdrawer_to_revenue_matches.csv": df_to_csv_bytes(matches_out),
            f"{timestamp}_cashdrawer_unmatched.csv": df_to_csv_bytes(unmatched_out),
            f"{timestamp}_revenue_rollup_with_approx_times.csv": df_to_csv_bytes(rev_all),
            f"{timestamp}_revenue_by_hour.csv": df_to_csv_bytes(hourly),
        }
        zip_bytes = build_zip(files)

        st.download_button(
            label="Download all outputs (ZIP)",
            data=zip_bytes,
            file_name=f"{timestamp}_revenue_timestamp_outputs.zip",
            mime="application/zip",
        )

        # Individual downloads
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.download_button("Matches CSV", data=files[f"{timestamp}_cashdrawer_to_revenue_matches.csv"],
                               file_name=f"{timestamp}_cashdrawer_to_revenue_matches.csv", mime="text/csv")
        with d2:
            st.download_button("Unmatched CSV", data=files[f"{timestamp}_cashdrawer_unmatched.csv"],
                               file_name=f"{timestamp}_cashdrawer_unmatched.csv", mime="text/csv")
        with d3:
            st.download_button("Revenue w/ Times CSV", data=files[f"{timestamp}_revenue_rollup_with_approx_times.csv"],
                               file_name=f"{timestamp}_revenue_rollup_with_approx_times.csv", mime="text/csv")
        with d4:
            st.download_button("Revenue by Hour CSV", data=files[f"{timestamp}_revenue_by_hour.csv"],
                               file_name=f"{timestamp}_revenue_by_hour.csv", mime="text/csv")

        st.divider()

        # --- Inspect outputs ---
        with st.expander("Preview: Matches"):
            st.dataframe(matches_out, use_container_width=True, hide_index=True)
        with st.expander("Preview: Unmatched"):
            st.dataframe(unmatched_out, use_container_width=True, hide_index=True)
        with st.expander("Preview: Revenue Rollup with Approx Times"):
            preview_cols = [c for c in ["Transaction Date", "Approx Transaction Time", "Provider", "Total Price", "Source", "Invoice Number", "Payment ID", "Confirmation Code"] if c in rev_all.columns]
            st.dataframe(rev_all[preview_cols].sort_values("Approx Transaction Time"), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)

else:
    st.info("Upload both CSVs, then click **Run**.")
