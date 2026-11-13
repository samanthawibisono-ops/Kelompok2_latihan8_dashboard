"""Final customer churn dashboard with EDA, filters, and model inference."""

import base64
import binascii
import io
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.model_utils import load_metadata, load_model, predict, validate_input


# Load the three saved models once; changing the selector only changes which
# precomputed predictions the charts read.
MODEL_DIR = Path(__file__).resolve().parent / "top3_models_cv_roc_auc"
REGISTRY = json.loads((MODEL_DIR / "top_models_registry.json").read_text(encoding="utf-8"))
MODEL_CONFIGS = {item["key"]: item for item in REGISTRY["models"]}
NO_MODEL = "none"
MODEL_OPTIONS = [{"label": "Tanpa model — Data aktual", "value": NO_MODEL}] + [
    {"label": f"Model {chr(64 + item['rank'])} — {item['name']}", "value": item["key"]}
    for item in REGISTRY["models"]
]
MODEL_LABELS = {option["value"]: option["label"] for option in MODEL_OPTIONS}
DEFAULT_MODEL = NO_MODEL
META = load_metadata(MODEL_DIR / "model_metadata.json")
MODELS = {
    key: load_model(MODEL_DIR / item["file"])
    for key, item in MODEL_CONFIGS.items()
}
raw = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])
raw["total_charges"] = raw["total_charges"].fillna(
    raw["monthly_charges"] * raw["tenure_months"]
)
raw["status"] = raw["churn"].map({0: "Bertahan", 1: "Churn"})
raw["periode"] = raw["signup_date"].dt.to_period("Q").astype(str)
DF_BY_MODEL = {}
for model_key, model in MODELS.items():
    DF_BY_MODEL[model_key] = predict(model, raw, META)

CATS = META["categories"]
RISK_ORDER = ["Rendah", "Sedang", "Tinggi"]
RISK_COLOR = {"Rendah": "#55A868", "Sedang": "#DD8452", "Tinggi": "#C44E52"}
STATUS_COLOR = {"Bertahan": "#4C72B0", "Churn": "#C44E52"}
TENURE_MIN = int(raw["tenure_months"].min())
TENURE_MAX = int(raw["tenure_months"].max())
MAX_UPLOAD_BYTES = 10_000_000
MAX_UPLOAD_ROWS = 5_000


def parse_uploaded_csv(contents, filename):
    if not filename or not filename.lower().endswith(".csv"):
        raise ValueError("Pilih file dengan ekstensi .csv.")
    if not contents or "," not in contents:
        raise ValueError("Isi file tidak dapat dibaca.")
    encoded = contents.split(",", 1)[1]
    if len(encoded) * 3 // 4 > MAX_UPLOAD_BYTES:
        raise ValueError("Ukuran CSV maksimal 10 MB.")
    try:
        decoded = base64.b64decode(encoded, validate=True)
        frame = pd.read_csv(io.BytesIO(decoded), encoding="utf-8-sig")
    except (binascii.Error, ValueError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError("CSV harus berformat UTF-8 dan memiliki kolom yang valid.") from exc
    if frame.empty:
        raise ValueError("CSV tidak berisi baris pelanggan.")
    if len(frame) > MAX_UPLOAD_ROWS:
        raise ValueError(f"CSV maksimal {MAX_UPLOAD_ROWS:,} baris pelanggan.")
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        raise ValueError("Nama kolom CSV tidak boleh duplikat.")

    required = set(META["feature_order"]) - {"total_charges"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Kolom wajib belum ada: {', '.join(missing)}.")

    for field in META["numeric_features"]:
        if field == "total_charges" and field not in frame:
            frame[field] = pd.NA
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame["total_charges"] = frame["total_charges"].fillna(
        frame["monthly_charges"] * frame["tenure_months"]
    )
    invalid_numeric = [
        field for field in META["numeric_features"] if frame[field].isna().any()
    ]
    if invalid_numeric:
        raise ValueError(f"Kolom numerik berisi nilai kosong/tidak valid: {', '.join(invalid_numeric)}.")

    for field in META["categorical_features"]:
        frame[field] = frame[field].astype("string").str.strip()
        if frame[field].isna().any() or frame[field].eq("").any():
            raise ValueError(f"Kolom {field} berisi nilai kosong.")
    errors = validate_input(frame, META)
    if errors:
        raise ValueError(" ".join(errors))

    generated_ids = pd.Series(
        [f"UPLOAD-{index + 1:05d}" for index in range(len(frame))], index=frame.index
    )
    if "customer_id" not in frame:
        frame["customer_id"] = generated_ids
    else:
        frame["customer_id"] = frame["customer_id"].astype("string").str.strip()
        frame["customer_id"] = frame["customer_id"].where(
            frame["customer_id"].notna() & frame["customer_id"].ne(""), generated_ids
        )

    if "signup_date" in frame:
        dates = pd.to_datetime(frame["signup_date"], errors="coerce")
        provided = frame["signup_date"].notna() & frame["signup_date"].astype(str).str.strip().ne("")
        if (provided & dates.isna()).any():
            raise ValueError("Kolom signup_date berisi tanggal yang tidak valid.")
        frame["signup_date"] = dates
    else:
        frame["signup_date"] = pd.NaT
    if "churn" in frame:
        churn = pd.to_numeric(frame["churn"], errors="coerce")
        if churn.isna().any() or not churn.isin([0, 1]).all():
            raise ValueError("Kolom churn harus berisi 0 atau 1 di setiap baris.")
        frame["churn"] = churn.astype(int)
    else:
        frame["churn"] = float("nan")
    frame["status"] = frame["churn"].map({0: "Bertahan", 1: "Churn"}).fillna("Belum diketahui")
    frame["periode"] = (
        frame["signup_date"].dt.to_period("Q").astype("string").fillna("")
    )
    columns = ["customer_id", "signup_date", *META["feature_order"], "churn", "status", "periode"]
    return frame[columns]

COLOR = {
    "bg": "#F5F7FB",
    "card": "white",
    "ink": "#1F2A44",
    "muted": "#6B7280",
    "accent": "#2F5DA8",
}
CARD = {
    "background": COLOR["card"],
    "borderRadius": "12px",
    "padding": "16px",
    "boxShadow": "0 1px 3px rgba(0,0,0,.08)",
}


def label(text):
    return html.Label(text, style={"fontWeight": 600, "fontSize": "13px"})


def rupiah(value):
    return f"Rp{value:,.0f}".replace(",", ".")


def kpi(judul, nilai, sub=""):
    return html.Div(
        style={**CARD, "flex": 1},
        children=[
            html.Div(judul, style={"color": COLOR["muted"], "fontSize": "13px"}),
            html.Div(
                nilai,
                style={"fontSize": "28px", "fontWeight": 700, "color": COLOR["ink"]},
            ),
            html.Div(sub, style={"color": COLOR["muted"], "fontSize": "12px"}),
        ],
    )


def num_input(component_id, text, value, minimum, maximum, step=1):
    return html.Div(
        [
            label(text),
            dcc.Input(
                id=component_id,
                type="number",
                value=value,
                min=minimum,
                max=maximum,
                step=step,
                style={"width": "100%"},
            ),
        ]
    )


def dd_input(component_id, text, options):
    return html.Div(
        [
            label(text),
            dcc.Dropdown(
                id=component_id, options=options, value=options[0], clearable=False
            ),
        ]
    )


def chart_card(component_id):
    return html.Div(dcc.Graph(id=component_id), style=CARD)


sidebar = html.Div(
    style={**CARD, "width": "250px", "flexShrink": 0},
    children=[
        html.H4("Model & Filter", style={"marginTop": 0}),
        label("Model prediksi"),
        dcc.Dropdown(
            id="f-model", options=MODEL_OPTIONS, value=DEFAULT_MODEL, clearable=False
        ),
        html.Small(id="model-info", style={"color": COLOR["muted"]}),
        html.Hr(),
        label("Data CSV"),
        dcc.Upload(
            id="upload-csv",
            children=html.Div("Klik atau tarik file CSV"),
            accept=".csv",
            multiple=False,
            style={
                "border": "1px dashed #9AA7BD",
                "borderRadius": "8px",
                "padding": "16px 8px",
                "textAlign": "center",
                "cursor": "pointer",
            },
        ),
        html.Small("Format: customer_churn.csv atau new_customers.csv · maks. 5.000 baris"),
        html.Div(
            id="upload-status",
            children=html.Small(f"Data bawaan: {len(raw):,} pelanggan."),
            style={"marginTop": "8px"},
        ),
        html.Button("Gunakan data bawaan", id="reset-upload", n_clicks=0),
        html.Hr(),
        label("Region"),
        dcc.Dropdown(id="f-region", options=CATS["region"], multi=True, placeholder="Semua"),
        html.Br(),
        label("Segmen"),
        dcc.Dropdown(id="f-segment", options=CATS["segment"], multi=True, placeholder="Semua"),
        html.Br(),
        label("Jenis kontrak"),
        dcc.Checklist(
            id="f-kontrak", options=CATS["contract_type"], value=CATS["contract_type"]
        ),
        html.Br(),
        html.Div(
            id="risk-filter-container",
            style={"display": "none"},
            children=[
                label("Kategori risiko (prediksi)"),
                dcc.Checklist(id="f-risk", options=RISK_ORDER, value=RISK_ORDER, inline=True),
                html.Br(),
            ],
        ),
        label("Tenure (bulan)"),
        dcc.RangeSlider(
            id="f-tenure",
            min=TENURE_MIN,
            max=TENURE_MAX,
            value=[TENURE_MIN, TENURE_MAX],
            step=1,
            marks={n: str(n) for n in (TENURE_MIN, 24, 48, TENURE_MAX)},
        ),
    ],
)

TABLE_FIELDS = [
    ("ID", "customer_id"),
    ("Region", "region"),
    ("Segmen", "segment"),
    ("Kontrak", "contract_type"),
    ("Tenure", "tenure_months"),
    ("Keluhan", "num_complaints"),
    ("Prob. churn", "churn_probability"),
    ("Risiko", "risk_category"),
]
RAW_TABLE_FIELDS = [
    ("ID", "customer_id"),
    ("Region", "region"),
    ("Segmen", "segment"),
    ("Kontrak", "contract_type"),
    ("Tenure", "tenure_months"),
    ("Tagihan bulanan", "monthly_charges"),
    ("Keluhan", "num_complaints"),
    ("Status aktual", "status"),
]
UNLABELED_TABLE_FIELDS = [
    ("ID", "customer_id"),
    ("Region", "region"),
    ("Segmen", "segment"),
    ("Kontrak", "contract_type"),
    ("Tenure", "tenure_months"),
    ("Tagihan bulanan", "monthly_charges"),
    ("Keluhan", "num_complaints"),
    ("Panggilan CS", "support_calls"),
]


def table_columns(fields):
    return [
        {
            "name": name,
            "id": field,
            **(
                {"type": "numeric", "format": {"specifier": ".1%"}}
                if field == "churn_probability"
                else {}
            ),
        }
        for name, field in fields
    ]


TABLE_COLUMNS = table_columns(TABLE_FIELDS)
RAW_TABLE_COLUMNS = table_columns(RAW_TABLE_FIELDS)
UNLABELED_TABLE_COLUMNS = table_columns(UNLABELED_TABLE_FIELDS)

tab_overview = html.Div(
    [
        html.Div(id="kpi-row", style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
            children=[
                chart_card("g-donut"),
                chart_card("g-region-risk"),
                chart_card("g-trend"),
                chart_card("g-contract"),
                chart_card("g-risk"),
                chart_card("g-actual-pred"),
            ],
        ),
        html.Div(
            style={**CARD, "marginTop": "12px"},
            children=[
                html.H4(id="tbl-title", style={"marginTop": 0}),
                dash_table.DataTable(
                    id="tbl-risk",
                    page_size=10,
                    sort_action="native",
                    filter_action="native",
                    columns=RAW_TABLE_COLUMNS,
                    style_cell={"padding": "6px", "fontSize": "13px", "textAlign": "left"},
                    style_header={"fontWeight": "bold", "background": "#E9EDF5"},
                    style_data_conditional=[
                        {
                            "if": {
                                "filter_query": "{risk_category} = 'Tinggi'",
                                "column_id": "risk_category",
                            },
                            "color": "white",
                            "backgroundColor": RISK_COLOR["Tinggi"],
                        },
                        {
                            "if": {
                                "filter_query": "{risk_category} = 'Sedang'",
                                "column_id": "risk_category",
                            },
                            "backgroundColor": "#FBE3D3",
                        },
                    ],
                ),
            ],
        ),
    ]
)

tab_predict = html.Div(
    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
    children=[
        html.Div(
            style=CARD,
            children=[
                html.H4("Simulasi prediksi pelanggan", style={"marginTop": 0}),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px"},
                    children=[
                        dd_input("p-region", "Region", CATS["region"]),
                        dd_input("p-segment", "Segmen", CATS["segment"]),
                        dd_input("p-contract", "Kontrak", CATS["contract_type"]),
                        dd_input("p-payment", "Pembayaran", CATS["payment_method"]),
                        dd_input("p-service", "Layanan", CATS["internet_service"]),
                        num_input("p-tenure", "Tenure (bulan)", 6, 1, 72),
                        num_input("p-monthly", "Tagihan bulanan (Rp)", 450000, 100000, 2000000, 10000),
                        num_input("p-complaints", "Jumlah keluhan", 2, 0, 10),
                        num_input("p-calls", "Panggilan CS", 3, 0, 15),
                    ],
                ),
                html.Br(),
                html.Button(
                    "Jalankan prediksi",
                    id="btn-predict",
                    n_clicks=0,
                    disabled=True,
                    style={
                        "background": COLOR["accent"],
                        "color": "white",
                        "border": 0,
                        "padding": "10px 18px",
                        "borderRadius": "8px",
                        "cursor": "pointer",
                    },
                ),
            ],
        ),
        html.Div(
            style=CARD,
            children=[
                html.H4("Hasil prediksi", style={"marginTop": 0}),
                html.Div(
                    id="pred-output",
                    children=html.P("Pilih Model A, B, atau C untuk memprediksi pelanggan."),
                ),
                dcc.Graph(id="g-gauge", style={"height": "260px"}),
            ],
        ),
    ],
)

app = Dash(__name__, title="Final Dashboard — Customer Churn")
server = app.server  # Entry point for deployment.
app.layout = html.Div(
    style={
        "background": COLOR["bg"],
        "minHeight": "100vh",
        "padding": "16px",
        "fontFamily": "Inter, Arial, sans-serif",
        "color": COLOR["ink"],
    },
    children=[
        dcc.Store(id="dataset-store", storage_type="memory"),
        html.Div(
            style={"marginBottom": "12px"},
            children=[
                html.H2("Customer Churn Intelligence Dashboard", style={"margin": 0}),
                html.Div(
                    "Pertanyaan bisnis: pelanggan mana yang berisiko churn dan apa yang harus "
                    "dilakukan? · Gunakan data bawaan atau unggah CSV pelanggan",
                    style={"color": COLOR["muted"]},
                ),
            ],
        ),
        html.Div(
            style={"display": "flex", "gap": "12px", "alignItems": "flex-start"},
            children=[
                sidebar,
                html.Div(
                    style={"flex": 1, "minWidth": 0},
                    children=[
                        dcc.Tabs(
                            id="tabs",
                            value="overview",
                            children=[
                                dcc.Tab(
                                    label="Overview Data",
                                    value="overview",
                                    children=[tab_overview],
                                ),
                                dcc.Tab(
                                    label="Simulasi Prediksi",
                                    value="predict",
                                    children=[tab_predict],
                                ),
                            ],
                        )
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("dataset-store", "data"),
    Output("upload-status", "children"),
    Input("upload-csv", "contents"),
    Input("reset-upload", "n_clicks"),
    State("upload-csv", "filename"),
    prevent_initial_call=True,
)
def update_dataset(contents, _reset_clicks, filename):
    if ctx.triggered_id == "reset-upload":
        return None, html.Small(f"Data bawaan: {len(raw):,} pelanggan.")
    if not contents:
        raise PreventUpdate
    try:
        frame = parse_uploaded_csv(contents, filename)
    except ValueError as exc:
        return no_update, html.Small(f"Upload gagal: {exc}", style={"color": "crimson"})
    label_type = "dengan churn aktual" if frame["churn"].notna().any() else "tanpa churn aktual"
    status = html.Small(f"{filename}: {len(frame):,} pelanggan ({label_type}).")
    records = json.loads(frame.to_json(orient="records", date_format="iso"))
    return {"filename": filename, "records": records}, status


@app.callback(
    Output("f-tenure", "min"),
    Output("f-tenure", "max"),
    Output("f-tenure", "value"),
    Output("f-tenure", "marks"),
    Output("f-region", "value"),
    Output("f-segment", "value"),
    Output("f-kontrak", "value"),
    Output("f-risk", "value"),
    Input("dataset-store", "data"),
)
def update_tenure_range(upload_data):
    frame = raw if upload_data is None else pd.DataFrame.from_records(upload_data["records"])
    minimum = int(frame["tenure_months"].min())
    maximum = max(int(frame["tenure_months"].max()), minimum + 1)
    middle = (minimum + maximum) // 2
    marks = {value: str(value) for value in (minimum, middle, maximum)}
    return minimum, maximum, [minimum, maximum], marks, None, None, CATS["contract_type"], RISK_ORDER


@app.callback(
    Output("model-info", "children"),
    Output("risk-filter-container", "style"),
    Output("btn-predict", "disabled"),
    Input("f-model", "value"),
)
def show_model_info(model_key):
    if model_key == NO_MODEL:
        return "Analisis data pelanggan tanpa prediksi model.", {"display": "none"}, True
    if model_key not in MODEL_CONFIGS:
        raise PreventUpdate
    config = MODEL_CONFIGS[model_key]
    info = (
        f"CV ROC AUC {config['cv_roc_auc']['mean']:.4f} · "
        f"Recall uji {config['holdout_metrics']['recall']:.1%}"
    )
    return info, {}, False


def dataset_for_model(model_key, upload_data):
    if upload_data is None:
        return raw if model_key == NO_MODEL else DF_BY_MODEL[model_key]
    frame = pd.DataFrame.from_records(upload_data["records"])
    return frame if model_key == NO_MODEL else predict(MODELS[model_key], frame, META)


def apply_filters(model_key, region, segment, kontrak, risk, tenure, upload_data=None):
    filtered = dataset_for_model(model_key, upload_data)
    if region:
        filtered = filtered[filtered["region"].isin(region)]
    if segment:
        filtered = filtered[filtered["segment"].isin(segment)]
    filtered = filtered[filtered["contract_type"].isin(kontrak or [])]
    if model_key != NO_MODEL:
        filtered = filtered[filtered["risk_category"].isin(risk or [])]
    return filtered[filtered["tenure_months"].between(*tenure)]


FILTERS = [
    Input("f-model", "value"),
    Input("f-region", "value"),
    Input("f-segment", "value"),
    Input("f-kontrak", "value"),
    Input("f-risk", "value"),
    Input("f-tenure", "value"),
    Input("dataset-store", "data"),
]


def unlabeled_overview(filtered):
    kpis = [
        kpi("Pelanggan", f"{len(filtered):,}", "sesuai filter"),
        kpi("Rata-rata tenure", f"{filtered['tenure_months'].mean():.1f} bulan"),
        kpi("Rata-rata tagihan bulanan", rupiah(filtered["monthly_charges"].mean())),
        kpi("Rata-rata keluhan", f"{filtered['num_complaints'].mean():.1f}"),
    ]
    segment = px.pie(
        filtered, names="segment", hole=0.55, title="Komposisi pelanggan per segmen"
    )
    regional_counts = filtered.groupby("region", as_index=False).size().sort_values("size")
    region_chart = px.bar(
        regional_counts,
        x="size",
        y="region",
        orientation="h",
        text_auto=True,
        template="plotly_white",
        title="Jumlah pelanggan per region",
        labels={"size": "Pelanggan", "region": ""},
    )
    if filtered["periode"].ne("").any():
        counts = filtered[filtered["periode"].ne("")].groupby("periode", as_index=False).size()
        trend = px.line(
            counts,
            x="periode",
            y="size",
            markers=True,
            template="plotly_white",
            title="Pelanggan per kuartal pendaftaran",
            labels={"periode": "Kuartal pendaftaran", "size": "Pelanggan"},
        )
    else:
        service_counts = filtered.groupby("internet_service", as_index=False).size()
        trend = px.bar(
            service_counts,
            x="internet_service",
            y="size",
            template="plotly_white",
            title="Jumlah pelanggan per layanan internet",
            labels={"internet_service": "Layanan internet", "size": "Pelanggan"},
        )
    contract_data = filtered.groupby(["contract_type", "segment"]).size().reset_index(name="n")
    contract = px.bar(
        contract_data,
        x="contract_type",
        y="n",
        color="segment",
        template="plotly_white",
        title="Pelanggan per kontrak dan segmen",
        labels={"contract_type": "Jenis kontrak", "n": "Pelanggan", "segment": "Segmen"},
    )
    tenure = px.histogram(
        filtered,
        x="tenure_months",
        color="segment",
        nbins=30,
        template="plotly_white",
        title="Distribusi tenure pelanggan",
        labels={"tenure_months": "Tenure (bulan)", "segment": "Segmen"},
    )
    charges = px.scatter(
        filtered,
        x="tenure_months",
        y="monthly_charges",
        color="segment",
        hover_name="customer_id",
        opacity=0.6,
        template="plotly_white",
        title="Tenure vs tagihan bulanan",
        labels={"tenure_months": "Tenure (bulan)", "monthly_charges": "Tagihan bulanan (Rp)"},
    )
    charges.update_yaxes(tickprefix="Rp", tickformat=",.0f")
    table_fields = [field for _, field in UNLABELED_TABLE_FIELDS]
    return (
        kpis,
        segment,
        region_chart,
        trend,
        contract,
        tenure,
        charges,
        UNLABELED_TABLE_COLUMNS,
        filtered[table_fields].to_dict("records"),
        f"Data pelanggan tanpa label churn — {len(filtered):,} baris",
    )


def actual_overview(filtered):
    if not filtered["churn"].notna().any():
        return unlabeled_overview(filtered)
    churn_count = int(filtered["churn"].sum())
    kpis = [
        kpi("Pelanggan", f"{len(filtered):,}", "sesuai filter"),
        kpi("Churn rate aktual", f"{filtered['churn'].mean():.1%}"),
        kpi("Pelanggan churn", f"{churn_count:,}"),
        kpi("Rata-rata tagihan bulanan", rupiah(filtered["monthly_charges"].mean())),
    ]

    donut = px.pie(
        filtered,
        names="status",
        hole=0.55,
        color="status",
        color_discrete_map=STATUS_COLOR,
        title="Komposisi churn aktual",
    )
    donut.update_traces(textinfo="percent+label")

    regional_churn = (
        filtered.groupby("region", as_index=False)["churn"]
        .mean()
        .sort_values("churn")
    )
    region_chart = px.bar(
        regional_churn,
        x="churn",
        y="region",
        orientation="h",
        text_auto=".1%",
        template="plotly_white",
        color_discrete_sequence=[STATUS_COLOR["Churn"]],
        title="Churn rate aktual per region",
        labels={"churn": "Churn rate aktual", "region": ""},
    )
    region_chart.update_xaxes(tickformat=".0%")

    if filtered["periode"].ne("").any():
        quarterly_churn = (
            filtered[filtered["periode"].ne("")]
            .groupby("periode", as_index=False)["churn"]
            .mean()
        )
        trend = px.line(
            quarterly_churn,
            x="periode",
            y="churn",
            markers=True,
            template="plotly_white",
            title="Churn rate aktual per kuartal pendaftaran",
            labels={"periode": "Kuartal pendaftaran", "churn": "Churn rate"},
        )
    else:
        segment_churn = filtered.groupby("segment", as_index=False)["churn"].mean()
        trend = px.bar(
            segment_churn,
            x="segment",
            y="churn",
            template="plotly_white",
            title="Churn rate aktual per segmen",
            labels={"segment": "Segmen", "churn": "Churn rate"},
        )
    trend.update_yaxes(tickformat=".0%")

    contract_data = (
        filtered.groupby(["contract_type", "status"])
        .size()
        .reset_index(name="n")
    )
    contract = px.bar(
        contract_data,
        x="contract_type",
        y="n",
        color="status",
        color_discrete_map=STATUS_COLOR,
        category_orders={"contract_type": ["Bulanan", "Tahunan", "Dua Tahun"]},
        template="plotly_white",
        title="Status churn aktual per jenis kontrak",
        labels={"contract_type": "", "n": "Pelanggan", "status": "Status"},
    )

    tenure = px.histogram(
        filtered,
        x="tenure_months",
        color="status",
        nbins=30,
        color_discrete_map=STATUS_COLOR,
        template="plotly_white",
        title="Distribusi tenure menurut status aktual",
        labels={"tenure_months": "Tenure (bulan)", "status": "Status"},
    )

    charges = px.scatter(
        filtered,
        x="tenure_months",
        y="monthly_charges",
        color="status",
        hover_name="customer_id",
        opacity=0.6,
        color_discrete_map=STATUS_COLOR,
        template="plotly_white",
        title="Tenure vs tagihan bulanan",
        labels={
            "tenure_months": "Tenure (bulan)",
            "monthly_charges": "Tagihan bulanan (Rp)",
            "status": "Status",
        },
    )
    charges.update_yaxes(tickprefix="Rp", tickformat=",.0f")

    table_fields = [field for _, field in RAW_TABLE_FIELDS]
    return (
        kpis,
        donut,
        region_chart,
        trend,
        contract,
        tenure,
        charges,
        RAW_TABLE_COLUMNS,
        filtered[table_fields].to_dict("records"),
        f"Data pelanggan — {len(filtered):,} baris",
    )


@app.callback(
    Output("kpi-row", "children"),
    Output("g-donut", "figure"),
    Output("g-region-risk", "figure"),
    Output("g-trend", "figure"),
    Output("g-contract", "figure"),
    Output("g-risk", "figure"),
    Output("g-actual-pred", "figure"),
    Output("tbl-risk", "columns"),
    Output("tbl-risk", "data"),
    Output("tbl-title", "children"),
    *FILTERS,
)
def update_overview(model_key, region, segment, kontrak, risk, tenure, upload_data=None):
    if model_key != NO_MODEL and model_key not in MODELS:
        raise PreventUpdate
    filtered = apply_filters(model_key, region, segment, kontrak, risk, tenure, upload_data)
    if filtered.empty:
        empty = px.scatter(title="Tidak ada data — longgarkan filter", template="plotly_white")
        columns = RAW_TABLE_COLUMNS if model_key == NO_MODEL else TABLE_COLUMNS
        return [kpi("Pelanggan", "0")], empty, empty, empty, empty, empty, empty, columns, [], "Tidak ada data"
    if model_key == NO_MODEL:
        return actual_overview(filtered)

    has_actual = filtered["churn"].notna().any()
    high_risk = filtered["risk_category"].eq("Tinggi")
    source_data = raw if upload_data is None else pd.DataFrame.from_records(upload_data["records"])
    actual_rate = f"{filtered['churn'].mean():.1%}" if has_actual else "—"
    actual_subtitle = (
        f"total: {source_data['churn'].mean():.1%}"
        if has_actual else "Label churn tidak tersedia"
    )
    kpis = [
        kpi("Pelanggan", f"{len(filtered):,}", "sesuai filter"),
        kpi("Churn rate aktual", actual_rate, actual_subtitle),
        kpi("Rata-rata prob. churn", f"{filtered['churn_probability'].mean():.1%}", "prediksi model"),
        kpi(
            "Risiko tinggi",
            f"{high_risk.sum():,}",
            rupiah(filtered.loc[high_risk, "monthly_charges"].sum()) + "/bln berisiko",
        ),
    ]

    donut = px.pie(
        filtered,
        names="risk_category",
        hole=0.55,
        color="risk_category",
        color_discrete_map=RISK_COLOR,
        category_orders={"risk_category": RISK_ORDER},
        title="Komposisi kategori risiko",
    )
    donut.update_traces(textinfo="percent+label", sort=False)
    donut.add_annotation(text=f"<b>{len(filtered):,}</b><br>pelanggan", showarrow=False, font_size=14)
    donut.update_layout(showlegend=False)

    regional_risk = (
        filtered.assign(tinggi=high_risk)
        .groupby("region", as_index=False)["tinggi"]
        .mean()
        .sort_values("tinggi")
    )
    region_risk = px.bar(
        regional_risk,
        x="tinggi",
        y="region",
        orientation="h",
        text_auto=".1%",
        template="plotly_white",
        color_discrete_sequence=[RISK_COLOR["Tinggi"]],
        title="Porsi pelanggan risiko tinggi per region",
        labels={"tinggi": "Porsi risiko tinggi", "region": ""},
    )
    region_risk.update_xaxes(tickformat=".0%")

    has_period = filtered["periode"].ne("").any()
    group_field = "periode" if has_period else "segment"
    trend_source = filtered[filtered["periode"].ne("")] if has_period else filtered
    aggregations = {"prediksi": ("churn_probability", "mean")}
    if has_actual:
        aggregations["aktual"] = ("churn", "mean")
    trend_data = trend_source.groupby(group_field).agg(**aggregations).reset_index()
    trend_long = trend_data.melt(group_field, var_name="seri", value_name="rate")
    trend_title = (
        "Churn aktual vs rata-rata prediksi" if has_actual else "Rata-rata prediksi churn"
    )
    trend_title += " per kuartal pendaftaran" if has_period else " per segmen"
    trend_labels = {group_field: "Kuartal pendaftaran" if has_period else "Segmen", "rate": "Rate", "seri": ""}
    if has_period:
        trend = px.line(
            trend_long, x=group_field, y="rate", color="seri", markers=True,
            template="plotly_white", title=trend_title, labels=trend_labels,
        )
    else:
        trend = px.bar(
            trend_long, x=group_field, y="rate", color="seri", barmode="group",
            template="plotly_white", title=trend_title, labels=trend_labels,
        )
    trend.update_yaxes(tickformat=".0%")

    contract_data = (
        filtered.groupby(["contract_type", "risk_category"])
        .size()
        .reset_index(name="n")
    )
    contract = px.bar(
        contract_data,
        x="contract_type",
        y="n",
        color="risk_category",
        color_discrete_map=RISK_COLOR,
        category_orders={
            "risk_category": RISK_ORDER,
            "contract_type": ["Bulanan", "Tahunan", "Dua Tahun"],
        },
        template="plotly_white",
        title="Komposisi risiko per jenis kontrak",
        labels={"contract_type": "", "n": "Pelanggan", "risk_category": "Risiko"},
    )

    risk_distribution = px.histogram(
        filtered,
        x="churn_probability",
        nbins=30,
        color="risk_category",
        color_discrete_map=RISK_COLOR,
        category_orders={"risk_category": RISK_ORDER},
        template="plotly_white",
        title="Distribusi probabilitas churn (prediksi)",
        labels={"churn_probability": "Probabilitas churn", "risk_category": "Risiko"},
    )
    risk_distribution.add_vline(
        x=META["decision_threshold"], line_dash="dash", annotation_text="threshold"
    )

    regional_prediction = (
        filtered.groupby("region")
        .agg(prediksi=("churn_probability", "mean"), n=("customer_id", "count"))
        .reset_index()
    )
    if has_actual:
        actual_by_region = filtered.groupby("region", as_index=False)["churn"].mean()
        regional_prediction = regional_prediction.merge(
            actual_by_region.rename(columns={"churn": "aktual"}), on="region"
        )
        actual_pred = px.scatter(
            regional_prediction,
            x="aktual",
            y="prediksi",
            size="n",
            text="region",
            template="plotly_white",
            title="Region: churn aktual vs prediksi (ukuran = jumlah pelanggan)",
            labels={"aktual": "Churn rate aktual", "prediksi": "Rata-rata prediksi"},
        )
        padding = 0.03
        actual_pred.update_xaxes(
            tickformat=".0%",
            range=[regional_prediction["aktual"].min() - padding, regional_prediction["aktual"].max() + padding],
        )
    else:
        actual_pred = px.scatter(
            regional_prediction,
            x="n",
            y="prediksi",
            size="n",
            text="region",
            template="plotly_white",
            title="Jumlah pelanggan vs rata-rata prediksi per region",
            labels={"n": "Pelanggan", "prediksi": "Rata-rata prediksi"},
        )
    actual_pred.update_traces(textposition="top center", cliponaxis=False)
    actual_pred.update_yaxes(
        tickformat=".0%",
        range=[regional_prediction["prediksi"].min() - 0.03, regional_prediction["prediksi"].max() + 0.03],
    )

    top = filtered.sort_values("churn_probability", ascending=False)
    if upload_data is None:
        top = top.head(200)
        title = (
            f"Prioritas retensi ({MODEL_LABELS[model_key]}) — "
            f"{len(top)} pelanggan dengan probabilitas tertinggi"
        )
    else:
        title = f"Prediksi CSV ({MODEL_LABELS[model_key]}) — {len(top)} pelanggan"
    return (
        kpis,
        donut,
        region_risk,
        trend,
        contract,
        risk_distribution,
        actual_pred,
        TABLE_COLUMNS,
        top[[field for _, field in TABLE_FIELDS]].to_dict("records"),
        title,
    )


@app.callback(
    Output("pred-output", "children"),
    Output("g-gauge", "figure"),
    Input("btn-predict", "n_clicks"),
    Input("f-model", "value"),
    State("p-region", "value"),
    State("p-segment", "value"),
    State("p-contract", "value"),
    State("p-payment", "value"),
    State("p-service", "value"),
    State("p-tenure", "value"),
    State("p-monthly", "value"),
    State("p-complaints", "value"),
    State("p-calls", "value"),
    prevent_initial_call=True,
)
def run_prediction(_clicks, model_key, region, segment, contract, payment, service, tenure, monthly, complaints, calls):
    if model_key == NO_MODEL:
        return html.P("Pilih Model A, B, atau C untuk memprediksi pelanggan."), go.Figure()
    if model_key not in MODELS:
        raise PreventUpdate
    if not _clicks:
        return html.P("Isi form lalu klik tombol."), go.Figure()
    if any(value is None for value in (tenure, monthly, complaints, calls)):
        raise PreventUpdate

    row = pd.DataFrame(
        [{
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": monthly * tenure,
            "num_complaints": complaints,
            "support_calls": calls,
            "region": region,
            "segment": segment,
            "contract_type": contract,
            "payment_method": payment,
            "internet_service": service,
        }]
    )
    errors = validate_input(row, META)
    if errors:
        return html.Div([html.P(error, style={"color": "crimson"}) for error in errors]), px.scatter()

    result = predict(MODELS[model_key], row, META).iloc[0]
    recommendations = {
        "Tinggi": "Prioritaskan retensi: hubungi dalam 48 jam, tawarkan kontrak tahunan.",
        "Sedang": "Pantau: kirim survei kepuasan dan penawaran loyalitas.",
        "Rendah": "Pertahankan layanan standar.",
    }
    output = html.Div(
        [
            html.Div(MODEL_LABELS[model_key], style={"color": COLOR["muted"]}),
            html.Div(
                f"{result.churn_probability:.1%}",
                style={
                    "fontSize": "40px",
                    "fontWeight": 700,
                    "color": RISK_COLOR[result.risk_category],
                },
            ),
            html.Div(
                f"Prediksi: {'Churn' if result.predicted_churn else 'Bertahan'} "
                f"· Risiko {result.risk_category}"
            ),
            html.P(recommendations[result.risk_category], style={"color": COLOR["muted"]}),
        ]
    )
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=result.churn_probability * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": RISK_COLOR[result.risk_category]},
                "steps": [
                    {"range": [0, 30], "color": "#E6F4EA"},
                    {"range": [30, 60], "color": "#FDF0E6"},
                    {"range": [60, 100], "color": "#FBE3E3"},
                ],
            },
        )
    )
    gauge.update_layout(margin=dict(t=20, b=10, l=20, r=20), height=240)
    return output, gauge


if __name__ == "__main__":
    app.run(debug=False, port=8053)
