import io
import zipfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Revenue Timestamp Tool",
    page_icon="⏱️",
    layout="wide",
)

# -----------------------------
# Password protection
# -----------------------------
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "")

if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.title("Revenue Timestamp Tool 🔒")
    password = st.text_input("Enter password", type="password")

    if st.button("Login", type="primary"):
        if APP_PASSWORD and password == APP_PASSWORD:
            st.session_state.auth_ok = True
            st.rerun()
        elif not APP_PASSWORD:
            st.error(
                "No APP_PASSWORD is configured. Add it in Streamlit Cloud under "
                "App settings → Secrets."
            )
        else:
            st.error("Incorrect password.")

    st.stop()


st.title("Revenue Timestamp & Hourly Rollup")
st.caption(
    "Upload the Revenue Summary Rollup Report and Cash Drawer Logs Report. "
    "The app uses exact cash drawer timestamps as anchors, estimates times for "
    "other transactions, and produces a revenue-by-hour breakdown."
)

with st.sidebar:
    st.header("Upload reports")
    revenue_file = st.file_uploader(
        "Revenue Summary Rollup Report (.csv)",
        type=["csv"],
        key="revenue_file",
    )
    drawer_file = st.file_uploader(
        "Cash Drawer Logs Report (.csv)",
        type=["csv"],
        key="drawer_file",
    )

    st.divider()
    st.subheader("Timing settings")
    fallback_minutes_per_transaction = st.number_input(
        "Fallback minutes per transaction",
        min_value=0.1,
        max_value=30.0,
        value=1.0,
        step=0.1,
        help=(
            "Used only when a day has a single anchor or when an observed "
            "anchor interval cannot provide a usable transaction pace."
        ),
    )

    run_button = st.button(
        "Run report",
        type="primary",
        disabled=not (revenue_file and drawer_file),
        use_container_width=True,
    )


def read_csv(uploaded_file):
    data = uploaded_file.getvalue()

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding)
        except UnicodeDecodeError:
            continue

    return pd.read_csv(io.BytesIO(data))


def normalize_code(series):
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"^J-", "", regex=True)
    )


def normalize_name(series):
    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def numeric_series(df, column):
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")

    return pd.to_numeric(df[column], errors="coerce")


def validate_columns(df, required, report_name):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            f"{report_name} is missing required column(s): {', '.join(missing)}"
        )


def build_anchor_matches(revenue, drawer):
    """
    Match each eligible drawer event to at most one revenue cash transaction.

    Primary match:
      same date + normalized confirmation code + account ID + amount

    Conservative fallback:
      same date + account ID + amount, but only when that candidate is unique

    This prevents a later cash payment from attaching its timestamp to an older
    card deposit that happens to share a booking confirmation code.
    """

    rev = revenue.copy()
    log = drawer.copy()

    validate_columns(
        rev,
        [
            "Transaction Date",
            "Provider",
            "Confirmation Code",
            "Payment ID",
            "Total Price",
        ],
        "Revenue Summary Rollup Report",
    )
    validate_columns(
        log,
        [
            "Access Date Time",
            "Confirmation Code",
            "Cash Paid",
        ],
        "Cash Drawer Logs Report",
    )

    rev["_revenue_row_id"] = np.arange(len(rev))
    log["_drawer_row_id"] = np.arange(len(log))

    rev["Transaction Date"] = pd.to_datetime(
        rev["Transaction Date"], errors="coerce"
    )
    log["Access Date Time"] = pd.to_datetime(
        log["Access Date Time"], errors="coerce"
    )

    rev["_date"] = rev["Transaction Date"].dt.normalize()
    log["_date"] = log["Access Date Time"].dt.normalize()

    rev["_conf"] = normalize_code(rev["Confirmation Code"])
    log["_conf"] = normalize_code(log["Confirmation Code"])

    rev["_account"] = numeric_series(rev, "Account ID")
    log["_account"] = numeric_series(log, "Account ID")

    rev["_amount"] = numeric_series(rev, "Total Price").round(2)
    log["_amount"] = numeric_series(log, "Cash Paid").round(2)

    # Use only purchase/payment drawer events as anchors.
    if "Access Reason" in log.columns:
        eligible_reason = (
            log["Access Reason"].fillna("").astype(str).str.lower().str.strip()
            .isin(["payment", "purchase"])
        )
    else:
        eligible_reason = pd.Series(True, index=log.index)

    provider_is_cash = (
        rev["Provider"].fillna("").astype(str).str.lower().str.strip().eq("cash")
    )

    # Prefer purchases; allow refunds only when the sign also matches.
    if "Transaction Type" in rev.columns:
        rev_type = (
            rev["Transaction Type"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
        )
        eligible_rev_type = rev_type.isin(["purchase", "refund"])
    else:
        eligible_rev_type = pd.Series(True, index=rev.index)

    rev_candidates = rev[provider_is_cash & eligible_rev_type].copy()
    log_candidates = log[eligible_reason].copy()

    rev_candidates = rev_candidates.dropna(
        subset=["_date", "_account", "_amount"]
    )
    log_candidates = log_candidates.dropna(
        subset=["_date", "_account", "_amount", "Access Date Time"]
    )

    used_revenue_rows = set()
    matched_records = []
    unmatched_records = []

    # Sort drawer records by timestamp for deterministic one-to-one matching.
    for _, log_row in log_candidates.sort_values(
        ["_date", "Access Date Time", "_drawer_row_id"]
    ).iterrows():

        exact = rev_candidates[
            (rev_candidates["_date"] == log_row["_date"])
            & (rev_candidates["_account"] == log_row["_account"])
            & (rev_candidates["_amount"] == log_row["_amount"])
            & (rev_candidates["_conf"] == log_row["_conf"])
            & (~rev_candidates["_revenue_row_id"].isin(used_revenue_rows))
        ]

        match_method = "Date + confirmation + account + amount"

        if len(exact) == 1:
            chosen = exact.iloc[0]
        else:
            # Only use fallback when there is exactly one same-day account/amount
            # candidate. If multiple candidates exist, do not guess.
            fallback = rev_candidates[
                (rev_candidates["_date"] == log_row["_date"])
                & (rev_candidates["_account"] == log_row["_account"])
                & (rev_candidates["_amount"] == log_row["_amount"])
                & (~rev_candidates["_revenue_row_id"].isin(used_revenue_rows))
            ]

            if len(fallback) == 1:
                chosen = fallback.iloc[0]
                match_method = "Unique date + account + amount fallback"
            else:
                reason = (
                    "Ambiguous match"
                    if len(exact) > 1 or len(fallback) > 1
                    else "No matching cash revenue transaction"
                )
                unmatched = log_row.to_dict()
                unmatched["Unmatched Reason"] = reason
                unmatched_records.append(unmatched)
                continue

        used_revenue_rows.add(int(chosen["_revenue_row_id"]))

        record = {}
        for col in log.columns:
            if not col.startswith("_"):
                record[f"Drawer {col}"] = log_row.get(col)

        for col in rev.columns:
            if not col.startswith("_"):
                record[f"Revenue {col}"] = chosen.get(col)

        record["Match Method"] = match_method
        record["_revenue_row_id"] = int(chosen["_revenue_row_id"])
        record["Anchor Time"] = log_row["Access Date Time"]
        matched_records.append(record)

    matches = pd.DataFrame(matched_records)
    unmatched = pd.DataFrame(unmatched_records)

    return rev, matches, unmatched


def transaction_sort_key(df):
    payment = numeric_series(df, "Payment ID")
    invoice = numeric_series(df, "Invoice Number")

    # Payment ID is the primary sequence. Invoice Number is a fallback.
    key = payment.copy()
    key = key.where(key.notna(), invoice)

    # Final deterministic fallback to original row position.
    fallback = pd.Series(
        np.arange(len(df), dtype="float64"),
        index=df.index,
    )
    key = key.where(key.notna(), fallback)

    return key.astype(float)


def robust_minutes_per_position(anchor_positions, anchor_times, fallback):
    """
    Estimate typical minutes per transaction position from consecutive anchors.
    Median is used to reduce the impact of unusual gaps.
    """

    rates = []

    for i in range(1, len(anchor_positions)):
        position_gap = anchor_positions[i] - anchor_positions[i - 1]
        time_gap_minutes = (
            anchor_times[i] - anchor_times[i - 1]
        ).total_seconds() / 60.0

        if position_gap > 0 and time_gap_minutes > 0:
            rates.append(time_gap_minutes / position_gap)

    if not rates:
        return float(fallback)

    # Keep extreme intervals from producing unreasonable extrapolation.
    return float(np.clip(np.median(rates), 0.1, 15.0))


def estimate_day_times(day_df, fallback_minutes):
    day_df = day_df.sort_values(
        ["_sort_key", "_revenue_row_id"]
    ).copy().reset_index(drop=True)

    day_df["_position"] = np.arange(len(day_df), dtype=float)
    day_df["Approx Transaction Time"] = pd.NaT
    day_df["Time Estimate Method"] = ""
    day_df["Time Confidence"] = ""

    anchor_mask = day_df["Anchor Time"].notna()
    anchor_indices = day_df.index[anchor_mask].tolist()

    if not anchor_indices:
        day_df["Time Estimate Method"] = "No cash anchors available"
        day_df["Time Confidence"] = "Unassigned"
        return day_df

    # Preserve exact anchor timestamps.
    day_df.loc[anchor_mask, "Approx Transaction Time"] = day_df.loc[
        anchor_mask, "Anchor Time"
    ]
    day_df.loc[anchor_mask, "Time Estimate Method"] = "Exact cash anchor"
    day_df.loc[anchor_mask, "Time Confidence"] = "High"

    if len(anchor_indices) == 1:
        anchor_index = anchor_indices[0]
        anchor_time = day_df.loc[anchor_index, "Anchor Time"]

        for idx in day_df.index:
            if idx == anchor_index:
                continue

            offset = (idx - anchor_index) * fallback_minutes
            day_df.loc[idx, "Approx Transaction Time"] = (
                anchor_time + timedelta(minutes=float(offset))
            )

            if idx < anchor_index:
                day_df.loc[
                    idx, "Time Estimate Method"
                ] = "Extrapolated before only anchor"
            else:
                day_df.loc[
                    idx, "Time Estimate Method"
                ] = "Extrapolated after only anchor"

            day_df.loc[idx, "Time Confidence"] = "Low"

        return day_df

    # Adjacent-anchor interpolation.
    for left_idx, right_idx in zip(anchor_indices[:-1], anchor_indices[1:]):
        left_time = day_df.loc[left_idx, "Anchor Time"]
        right_time = day_df.loc[right_idx, "Anchor Time"]
        position_gap = right_idx - left_idx

        if position_gap <= 0:
            continue

        time_gap_seconds = (right_time - left_time).total_seconds()

        for idx in range(left_idx + 1, right_idx):
            fraction = (idx - left_idx) / position_gap

            # If cash anchors are out of chronological sequence, maintain a
            # non-decreasing timeline by holding intermediate rows at left_time.
            if time_gap_seconds >= 0:
                estimated = left_time + timedelta(
                    seconds=time_gap_seconds * fraction
                )
                confidence = "Medium"
            else:
                estimated = left_time
                confidence = "Low"

            day_df.loc[idx, "Approx Transaction Time"] = estimated
            day_df.loc[
                idx, "Time Estimate Method"
            ] = "Interpolated between adjacent anchors"
            day_df.loc[idx, "Time Confidence"] = confidence

    anchor_positions = [float(i) for i in anchor_indices]
    anchor_times = [
        pd.Timestamp(day_df.loc[i, "Anchor Time"]) for i in anchor_indices
    ]

    typical_rate = robust_minutes_per_position(
        anchor_positions,
        anchor_times,
        fallback_minutes,
    )

    # Extrapolate before the first anchor.
    first_idx = anchor_indices[0]
    first_time = pd.Timestamp(day_df.loc[first_idx, "Anchor Time"])

    for idx in range(first_idx - 1, -1, -1):
        positions_before = first_idx - idx
        estimated = first_time - timedelta(
            minutes=typical_rate * positions_before
        )
        day_df.loc[idx, "Approx Transaction Time"] = estimated
        day_df.loc[
            idx, "Time Estimate Method"
        ] = "Extrapolated before first anchor"
        day_df.loc[idx, "Time Confidence"] = "Low"

    # Extrapolate after the final anchor.
    last_idx = anchor_indices[-1]
    last_time = pd.Timestamp(day_df.loc[last_idx, "Anchor Time"])

    for idx in range(last_idx + 1, len(day_df)):
        positions_after = idx - last_idx
        estimated = last_time + timedelta(
            minutes=typical_rate * positions_after
        )
        day_df.loc[idx, "Approx Transaction Time"] = estimated
        day_df.loc[
            idx, "Time Estimate Method"
        ] = "Extrapolated after last anchor"
        day_df.loc[idx, "Time Confidence"] = "Low"

    # Keep estimates inside the transaction date.
    report_date = pd.Timestamp(day_df["Transaction Date"].iloc[0]).normalize()
    lower_bound = report_date
    upper_bound = report_date + timedelta(days=1) - timedelta(microseconds=1)

    non_anchor = ~anchor_mask

    too_early = non_anchor & (
        day_df["Approx Transaction Time"] < lower_bound
    )
    too_late = non_anchor & (
        day_df["Approx Transaction Time"] > upper_bound
    )

    day_df.loc[too_early, "Approx Transaction Time"] = lower_bound
    day_df.loc[too_early, "Time Estimate Method"] += " (clamped to day start)"

    day_df.loc[too_late, "Approx Transaction Time"] = upper_bound
    day_df.loc[too_late, "Time Estimate Method"] += " (clamped to day end)"

    return day_df


def build_outputs(revenue, drawer, fallback_minutes):
    rev, matches, unmatched = build_anchor_matches(revenue, drawer)

    anchor_map = {}
    if not matches.empty:
        anchor_map = dict(
            zip(matches["_revenue_row_id"], matches["Anchor Time"])
        )

    rev["Anchor Time"] = rev["_revenue_row_id"].map(anchor_map)
    rev["_sort_key"] = transaction_sort_key(rev)

    dated = rev[rev["Transaction Date"].notna()].copy()
    undated = rev[rev["Transaction Date"].isna()].copy()

    estimated_groups = []

    for _, group in dated.groupby(
        dated["Transaction Date"].dt.normalize(),
        sort=True,
    ):
        estimated_groups.append(
            estimate_day_times(group, fallback_minutes)
        )

    if estimated_groups:
        estimated = pd.concat(
            estimated_groups,
            ignore_index=True,
            sort=False,
        )
    else:
        estimated = dated.copy()

    if not undated.empty:
        undated["Approx Transaction Time"] = pd.NaT
        undated["Time Estimate Method"] = "Invalid transaction date"
        undated["Time Confidence"] = "Unassigned"
        estimated = pd.concat(
            [estimated, undated],
            ignore_index=True,
            sort=False,
        )

    estimated = estimated.sort_values("_revenue_row_id").reset_index(drop=True)

    estimated["_revenue_amount"] = numeric_series(
        estimated,
        "Total Price",
    ).fillna(0.0)

    estimated["Hour"] = pd.to_datetime(
        estimated["Approx Transaction Time"],
        errors="coerce",
    ).dt.floor("h")

    hourly = (
        estimated.dropna(subset=["Hour"])
        .groupby("Hour", as_index=False)
        .agg(
            Revenue=("_revenue_amount", "sum"),
            Transactions=("_revenue_row_id", "count"),
            Exact_Anchors=(
                "Time Estimate Method",
                lambda s: int((s == "Exact cash anchor").sum()),
            ),
        )
        .sort_values("Hour")
    )

    # Clean internal fields from user-facing outputs.
    revenue_output = estimated.drop(
        columns=[
            c
            for c in estimated.columns
            if c.startswith("_")
        ],
        errors="ignore",
    )

    match_output = matches.drop(
        columns=["_revenue_row_id"],
        errors="ignore",
    )

    unmatched_output = unmatched.drop(
        columns=[
            c
            for c in unmatched.columns
            if c.startswith("_")
        ],
        errors="ignore",
    )

    return revenue_output, match_output, unmatched_output, hourly


def dataframe_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def build_download_zip(files):
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for filename, content in files.items():
            archive.writestr(filename, content)

    buffer.seek(0)
    return buffer.read()


if run_button:
    try:
        revenue_df = read_csv(revenue_file)
        drawer_df = read_csv(drawer_file)

        with st.spinner("Matching cash anchors and estimating times..."):
            (
                revenue_output,
                match_output,
                unmatched_output,
                hourly_output,
            ) = build_outputs(
                revenue_df,
                drawer_df,
                fallback_minutes_per_transaction,
            )

        st.success("Report complete.")

        total_revenue = pd.to_numeric(
            revenue_output.get("Total Price"),
            errors="coerce",
        ).fillna(0).sum()

        assigned_times = revenue_output[
            "Approx Transaction Time"
        ].notna().sum()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Revenue transactions",
            f"{len(revenue_output):,}",
        )
        col2.metric(
            "Exact cash anchors",
            f"{len(match_output):,}",
        )
        col3.metric(
            "Transactions with times",
            f"{assigned_times:,}",
        )
        col4.metric(
            "Total revenue",
            f"${total_revenue:,.2f}",
        )

        st.divider()
        st.subheader("Revenue by Hour")

        display_hourly = hourly_output.copy()

        if not display_hourly.empty:
            display_hourly["Hour"] = pd.to_datetime(
                display_hourly["Hour"]
            ).dt.strftime("%Y-%m-%d %I:00 %p")

            st.dataframe(
                display_hourly,
                use_container_width=True,
                hide_index=True,
            )

            chart_data = hourly_output.set_index("Hour")["Revenue"]
            st.bar_chart(chart_data)
        else:
            st.warning(
                "No hourly output could be produced because no transaction "
                "times were assigned."
            )

        st.divider()
        st.subheader("Estimate quality")

        quality = (
            revenue_output.groupby(
                ["Time Confidence", "Time Estimate Method"],
                dropna=False,
            )
            .size()
            .reset_index(name="Transactions")
            .sort_values(
                ["Time Confidence", "Transactions"],
                ascending=[True, False],
            )
        )

        st.dataframe(
            quality,
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        st.subheader("Downloads")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        files = {
            f"{timestamp}_cashdrawer_to_revenue_matches.csv":
                dataframe_csv_bytes(match_output),
            f"{timestamp}_cashdrawer_unmatched.csv":
                dataframe_csv_bytes(unmatched_output),
            f"{timestamp}_revenue_rollup_with_estimated_times.csv":
                dataframe_csv_bytes(revenue_output),
            f"{timestamp}_revenue_by_hour.csv":
                dataframe_csv_bytes(hourly_output),
        }

        st.download_button(
            "Download all outputs (ZIP)",
            data=build_download_zip(files),
            file_name=f"{timestamp}_revenue_timestamp_outputs.zip",
            mime="application/zip",
            type="primary",
        )

        dl1, dl2, dl3, dl4 = st.columns(4)

        with dl1:
            st.download_button(
                "Matched anchors",
                files[
                    f"{timestamp}_cashdrawer_to_revenue_matches.csv"
                ],
                file_name=(
                    f"{timestamp}_cashdrawer_to_revenue_matches.csv"
                ),
                mime="text/csv",
            )

        with dl2:
            st.download_button(
                "Unmatched drawer logs",
                files[f"{timestamp}_cashdrawer_unmatched.csv"],
                file_name=f"{timestamp}_cashdrawer_unmatched.csv",
                mime="text/csv",
            )

        with dl3:
            st.download_button(
                "Revenue with times",
                files[
                    f"{timestamp}_revenue_rollup_with_estimated_times.csv"
                ],
                file_name=(
                    f"{timestamp}_revenue_rollup_with_estimated_times.csv"
                ),
                mime="text/csv",
            )

        with dl4:
            st.download_button(
                "Revenue by hour",
                files[f"{timestamp}_revenue_by_hour.csv"],
                file_name=f"{timestamp}_revenue_by_hour.csv",
                mime="text/csv",
            )

        st.divider()

        with st.expander("Preview matched cash anchors"):
            st.dataframe(
                match_output,
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Preview unmatched drawer entries"):
            st.dataframe(
                unmatched_output,
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Preview revenue transactions with estimated times"):
            preview_columns = [
                column
                for column in [
                    "Transaction Date",
                    "Approx Transaction Time",
                    "Time Estimate Method",
                    "Time Confidence",
                    "Provider",
                    "Source",
                    "Account Holder",
                    "Total Price",
                    "Invoice Number",
                    "Payment ID",
                    "Confirmation Code",
                ]
                if column in revenue_output.columns
            ]

            st.dataframe(
                revenue_output[preview_columns],
                use_container_width=True,
                hide_index=True,
            )

    except Exception as exc:
        st.error(f"Unable to process the reports: {exc}")
        st.exception(exc)

else:
    st.info("Upload both CSV reports, then click **Run report**.")
