"""
XPS（X線光電子分光）データ自動解析 Web アプリ
==============================================
Streamlit + Plotly + lmfit を用いて、ブラウザ上で XPS スペクトルの
読み込み・背景補正・ピーク検出・フィッティング・可視化を行います。

配色・参照データ:
  - 元素・軌道の結合エネルギー範囲と色は xps_database.json で一元管理
  - 新しい元素を追加する場合は JSON を編集するだけで UI / プロットに反映
  - 色は Okabe-Ito パレット（色覚バリアフリー）を使用
"""

from __future__ import annotations

import io
import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import olefile
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lmfit.models import PseudoVoigtModel, ConstantModel
from scipy.signal import find_peaks, savgol_filter

# OLE2 ストリーム名に Unicode を許可（Avantage プロパティストリーム用）
olefile.KEEP_UNICODE_NAMES = True


# ---------------------------------------------------------------------------
# 外部データベース（xps_database.json）の読み込み
# ---------------------------------------------------------------------------

# app.py と同じディレクトリにある JSON を参照（デプロイ時も相対パスで解決）
DATABASE_PATH = Path(__file__).resolve().parent / "xps_database.json"


def _read_xps_database_file(path: Path) -> dict[str, Any]:
    """
    JSON ファイルを読み込み、最低限のスキーマ検証を行う（キャッシュなし）。

    Parameters
    ----------
    path : Path
        xps_database.json のパス。

    Returns
    -------
    dict
        palette / role_colors / orbitals などを含む辞書。
    """
    if not path.exists():
        raise FileNotFoundError(
            f"XPS データベースが見つかりません: {path}\n"
            "プロジェクト直下に xps_database.json を配置してください。"
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "orbitals" not in data or not isinstance(data["orbitals"], list):
        raise ValueError("xps_database.json に orbitals 配列が必要です。")
    for i, orb in enumerate(data["orbitals"]):
        for key in ("name", "binding_energy_ev", "color", "search_window_ev"):
            if key not in orb:
                raise ValueError(f"orbitals[{i}] に必須キー '{key}' がありません。")
        window = orb["search_window_ev"]
        if "min" not in window or "max" not in window:
            raise ValueError(
                f"orbitals[{i}] ({orb.get('name')}) の search_window_ev に min/max が必要です。"
            )
    return data


def load_xps_database(path: Path | None = None) -> dict[str, Any]:
    """
    XPS 参照データベース（JSON）を読み込む。

    Streamlit は操作のたびにスクリプトを再実行するため、
    ここでの都度読込により JSON 編集が次の操作で反映されます。

    Parameters
    ----------
    path : Path or None
        JSON パス。省略時は DATABASE_PATH。

    Returns
    -------
    dict
        検証済みデータベース辞書。
    """
    return _read_xps_database_file(path or DATABASE_PATH)


def build_lookup_tables(db: dict[str, Any]) -> tuple[dict[str, str], dict[str, float], dict[str, dict]]:
    """
    JSON データベースからアプリ内で使う辞書を構築する。

    Parameters
    ----------
    db : dict
        load_xps_database の戻り値。

    Returns
    -------
    orbital_colors : dict[str, str]
        軌道名 → HEX カラー。
    reference_be : dict[str, float]
        軌道名 → 代表結合エネルギー (eV)。
    orbital_records : dict[str, dict]
        軌道名 → 元の軌道レコード（探索ウィンドウ等を含む）。
    """
    orbital_colors: dict[str, str] = {}
    reference_be: dict[str, float] = {}
    orbital_records: dict[str, dict] = {}

    unknown = db.get("unknown", {"name": "Unknown", "color": "#999999"})
    unknown_name = unknown.get("name", "Unknown")
    orbital_colors[unknown_name] = unknown.get("color", "#999999")

    for orb in db["orbitals"]:
        name = orb["name"]
        orbital_colors[name] = orb["color"]
        reference_be[name] = float(orb["binding_energy_ev"])
        orbital_records[name] = orb

    return orbital_colors, reference_be, orbital_records


def refresh_database_globals() -> dict[str, Any]:
    """
    xps_database.json を読み直し、モジュールグローバルな参照テーブルを更新する。

    Streamlit は操作のたびにスクリプトを上から再実行するため、
    この関数をモジュール末尾（または main 先頭）で呼べば JSON 編集が反映されます。

    Returns
    -------
    dict
        読み込んだデータベース。
    """
    global XPS_DB, OKABE_ITO, ROLE_COLORS, _FALLBACK_PALETTE
    global ORBITAL_COLORS, REFERENCE_BINDING_ENERGIES, ORBITAL_RECORDS
    global ASSIGN_TOLERANCE_EV, UNKNOWN_LABEL

    db = load_xps_database(DATABASE_PATH)

    XPS_DB = db
    OKABE_ITO = dict(db.get("palette", {}))
    ROLE_COLORS = dict(db.get("role_colors", {}))
    _FALLBACK_PALETTE = list(
        db.get(
            "fallback_palette",
            ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"],
        )
    )
    ORBITAL_COLORS, REFERENCE_BINDING_ENERGIES, ORBITAL_RECORDS = build_lookup_tables(db)
    ASSIGN_TOLERANCE_EV = float(db.get("default_assign_tolerance_ev", 5.0))
    UNKNOWN_LABEL = db.get("unknown", {}).get("name", "Unknown")
    return db


# 起動時（および Streamlit の各 rerun 時）にデータベースを読み込む
# 元素追加は xps_database.json の編集のみで完結する
XPS_DB = refresh_database_globals()


# ---------------------------------------------------------------------------
# ユーティリティ関数
# ---------------------------------------------------------------------------

def get_orbital_color(orbital_name: str, index: int = 0) -> str:
    """
    軌道名に対応する固定色を返す（xps_database.json の color を参照）。

    Parameters
    ----------
    orbital_name : str
        「C 1s」などの元素・軌道名。未登録の場合は予備パレットから選ぶ。
    index : int
        未登録軌道用のフォールバック色インデックス。

    Returns
    -------
    str
        HEX カラーコード。
    """
    if orbital_name in ORBITAL_COLORS:
        return ORBITAL_COLORS[orbital_name]
    return _FALLBACK_PALETTE[index % len(_FALLBACK_PALETTE)]


def is_excel_filename(filename: str) -> bool:
    """
    ファイル名が Excel（.xlsx / .xls）かどうかを判定する。

    Parameters
    ----------
    filename : str
        アップロードファイル名。

    Returns
    -------
    bool
        Excel ファイルなら True。
    """
    name = filename.lower()
    return name.endswith(".xlsx") or name.endswith(".xls")


def is_avantage_filename(filename: str) -> bool:
    """
    Thermo Fisher Avantage 形式（.vgx / .vgd）かどうかを判定する。

    Parameters
    ----------
    filename : str
        アップロードファイル名。

    Returns
    -------
    bool
        Avantage ファイルなら True。
    """
    name = filename.lower()
    return name.endswith(".vgx") or name.endswith(".vgd")


def excel_engine_for(filename: str) -> str:
    """
    拡張子に応じた pandas Excel エンジン名を返す。

    - .xlsx → openpyxl
    - .xls  → xlrd（xlrd 2.x は旧形式 .xls 専用）
    """
    name = filename.lower()
    if name.endswith(".xlsx"):
        return "openpyxl"
    if name.endswith(".xls"):
        return "xlrd"
    raise ValueError(f"未対応の Excel 拡張子です: {filename}")


def list_excel_sheet_names(uploaded_file) -> list[str]:
    """
    Excel ファイル内のシート名一覧を取得する。

    Parameters
    ----------
    uploaded_file : UploadedFile
        Streamlit のアップロードオブジェクト（getvalue でバイト列を取得）。

    Returns
    -------
    list[str]
        シート名のリスト。
    """
    engine = excel_engine_for(uploaded_file.name)
    bio = io.BytesIO(uploaded_file.getvalue())
    with pd.ExcelFile(bio, engine=engine) as xf:
        return list(xf.sheet_names)


def load_excel_sheet(uploaded_file, sheet_name: str) -> pd.DataFrame:
    """
    Excel の指定シートを DataFrame として読み込む（列はそのまま保持）。

    Parameters
    ----------
    uploaded_file : UploadedFile
        アップロードされた Excel ファイル。
    sheet_name : str
        読み込むシート名。

    Returns
    -------
    pd.DataFrame
        シートの生データ（ヘッダー行ありとして読み込み）。
    """
    engine = excel_engine_for(uploaded_file.name)
    bio = io.BytesIO(uploaded_file.getvalue())
    df = pd.read_excel(bio, sheet_name=sheet_name, engine=engine)
    # 完全に空の列・行を除去して扱いやすくする
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if df.empty:
        raise ValueError(f"シート「{sheet_name}」に有効なデータがありません。")
    # 列名を文字列に統一（selectbox 表示のため）
    df.columns = [str(c) for c in df.columns]
    return df.reset_index(drop=True)


def guess_column_name(columns: list[str], keywords: list[str], fallback_index: int = 0) -> str:
    """
    列名リストからキーワードに部分一致する列を推定する。

    Parameters
    ----------
    columns : list[str]
        候補となる列名。
    keywords : list[str]
        優先して探すキーワード（小文字比較）。例: ["binding", "energy", "結合"]
    fallback_index : int
        一致が無い場合に使う列インデックス。

    Returns
    -------
    str
        推定された列名。
    """
    lowered = [(c, c.lower()) for c in columns]
    for kw in keywords:
        for original, low in lowered:
            if kw in low:
                return original
    if not columns:
        raise ValueError("選択可能な列がありません。")
    idx = min(fallback_index, len(columns) - 1)
    return columns[idx]


def build_xps_dataframe_from_columns(
    raw_df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    """
    ユーザーが選んだ X/Y 列から解析用 DataFrame を整形する。

    処理内容:
      1. 指定列を数値型に変換（変換できない値は NaN）
      2. 欠損値（NaN）を除去
      3. 列名を ['Binding Energy', 'Intensity'] に正規化
      4. 結合エネルギー降順でソート（XPS 表示の慣例に合わせる）

    Parameters
    ----------
    raw_df : pd.DataFrame
        Excel などから読み込んだ生テーブル。
    x_col : str
        結合エネルギー（X軸）列名。
    y_col : str
        強度（Y軸）列名。

    Returns
    -------
    pd.DataFrame
        既存パイプラインに渡せる形式の DataFrame。
    """
    if x_col == y_col:
        raise ValueError("X軸とY軸に同じ列は指定できません。別々の列を選んでください。")
    if x_col not in raw_df.columns or y_col not in raw_df.columns:
        raise ValueError("指定された列がデータに存在しません。")

    x = pd.to_numeric(raw_df[x_col], errors="coerce")
    y = pd.to_numeric(raw_df[y_col], errors="coerce")
    result = pd.DataFrame({"Binding Energy": x, "Intensity": y})
    result = result.dropna().sort_values("Binding Energy", ascending=False).reset_index(drop=True)

    if len(result) < 5:
        raise ValueError(
            "数値として有効なデータ点が少なすぎます（5点未満）。"
            "列の選択やシート内容を確認してください。"
        )
    return result


def load_xps_file(uploaded_file) -> pd.DataFrame:
    """
    アップロードされた CSV / TXT を読み込み、Binding Energy と Intensity 列を返す。

    想定フォーマット:
      - 1列目: Binding Energy (eV)
      - 2列目: Intensity (counts など)
      - ヘッダー有無は自動判定（数値で始まらなければヘッダーありとみなす）
      - 区切りはカンマ・タブ・空白のいずれかに対応

    ※ Excel（.xlsx / .xls）はこの関数ではなく、シート・列選択 UI 経由で読み込みます。

    Parameters
    ----------
    uploaded_file : UploadedFile
        Streamlit のファイルアップローダから得たオブジェクト。

    Returns
    -------
    pd.DataFrame
        列名を ['Binding Energy', 'Intensity'] に正規化した DataFrame。
    """
    # getvalue() はポインタ位置に依存せず、何度でも同じバイト列を取得できる
    raw_bytes = uploaded_file.getvalue()
    text = raw_bytes.decode("utf-8", errors="ignore")

    # 区切り文字の推定（カンマ → タブ → 空白の順で試行）
    sample = text.splitlines()[:5]
    sample_joined = "\n".join(sample)
    if "," in sample_joined:
        sep = ","
    elif "\t" in sample_joined:
        sep = "\t"
    else:
        sep = r"\s+"

    # 先頭行が数値かどうかでヘッダー有無を判定
    first_token = sample[0].replace(",", " ").split()[0] if sample else ""
    has_header = False
    try:
        float(first_token)
    except ValueError:
        has_header = True

    df = pd.read_csv(
        io.StringIO(text),
        sep=sep,
        engine="python",
        header=0 if has_header else None,
    )

    # 数値列が2列以上あることを確認し、先頭2列を使用
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        # すべて object 型の場合は数値変換を試みる
        df = df.apply(pd.to_numeric, errors="coerce")
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if len(numeric_cols) < 2:
        raise ValueError(
            "データから数値列を2列以上読み取れませんでした。"
            "Binding Energy と Intensity の2列構成か確認してください。"
        )

    result = df[numeric_cols[:2]].copy()
    result.columns = ["Binding Energy", "Intensity"]
    result = result.dropna().sort_values("Binding Energy", ascending=False).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Thermo Fisher Avantage（.vgd / .vgx）OLE2 パーサ
# ---------------------------------------------------------------------------
#
# Avantage の測定データは Microsoft OLE2 コンパウンドファイルとして保存され、
# 主に次のストリームを持ちます:
#   - VGData      : 強度（little-endian float64）
#   - VGSpaceAxes : エネルギー軸（開始・ステップ・点数など）
#   - VGDataAxes  : 多次元（複数スペクトル）情報
#   - プロパティ  : SourceEnergy など測定パラメータ
# Binding Energy は通常 BE = SourceEnergy - KineticEnergy で算出します。
# ---------------------------------------------------------------------------

# OLE プロパティストリーム名（Avantage 固有）
_AVANTAGE_PROP_STREAM = "\x05Q5nw4m3lIjudbfwyAayojlptCa"

# VGSpaceAxes の軸タイプ番号 → 名前
_SPACE_AXIS_TYPES = {
    0: "UNDEFINED",
    1: "ENERGY",
    2: "ANGLE",
    3: "X",
    4: "Y",
    5: "LEVEL",
    6: "ETCHLEVEL",
    10: "POSITION",
}

# Al Kα / Mg Kα の代表値（SourceEnergy 推定用）
_DEFAULT_SOURCE_ENERGY_EV = 1486.68


@dataclass
class AvantageSpectrum:
    """Avantage ファイルから抽出した 1 本の XPS スペクトル。"""

    name: str
    binding_energy: np.ndarray
    intensity: np.ndarray
    kinetic_energy: np.ndarray = field(default_factory=lambda: np.array([]))
    source_energy: float = _DEFAULT_SOURCE_ENERGY_EV
    pass_energy: Optional[float] = None
    num_points: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """解析パイプライン用 DataFrame（Binding Energy / Intensity）へ変換する。"""
        df = pd.DataFrame(
            {
                "Binding Energy": np.asarray(self.binding_energy, dtype=float),
                "Intensity": np.asarray(self.intensity, dtype=float),
            }
        )
        return df.dropna().sort_values("Binding Energy", ascending=False).reset_index(drop=True)


def _read_ole_bstr(buf: io.BytesIO) -> str:
    """
    Avantage OLE 内の BSTR（長さ付き UTF-16LE）を読み取る。

    形式: uint32 バイト長 + UTF-16LE 文字列（末尾 null 含むことが多い）
    """
    raw_len = buf.read(4)
    if len(raw_len) < 4:
        return ""
    (nbytes,) = struct.unpack("<I", raw_len)
    data = buf.read(nbytes)
    if not data or data == b"\x00\x00":
        return ""
    try:
        return data.decode("utf-16-le").rstrip("\x00")
    except UnicodeDecodeError:
        return ""


def _parse_vg_space_axes(raw: bytes) -> list[dict[str, Any]]:
    """
    VGSpaceAxes ストリームを解析し、軸情報のリストを返す。

    線形 ENERGY 軸の場合、start / width(step) / points から
    運動エネルギー（または結合エネルギー）配列を再構築できます。
    """
    buf = io.BytesIO(raw)
    header = buf.read(8)
    if len(header) < 8:
        raise ValueError("VGSpaceAxes が短すぎます。")
    type_code, num_axis = struct.unpack("<2I", header)

    axes: list[dict[str, Any]] = []
    for _ in range(num_axis):
        axis: dict[str, Any] = {
            "label": _read_ole_bstr(buf),
            "symbol": _read_ole_bstr(buf),
            "unit": _read_ole_bstr(buf),
        }
        numeric = buf.read(20)
        if len(numeric) < 20:
            raise ValueError("VGSpaceAxes の軸数値部が不足しています。")
        points, start, width = struct.unpack("<Idd", numeric)
        axis["points"] = int(points)
        axis["start"] = float(start)
        axis["width"] = float(width)

        lin_flag = buf.read(1)
        if len(lin_flag) < 1:
            raise ValueError("VGSpaceAxes の線形フラグが不足しています。")
        axis["linear"] = bool(struct.unpack("<b", lin_flag)[0])

        if axis["linear"]:
            type_raw = buf.read(4)
            if len(type_raw) < 4:
                raise ValueError("VGSpaceAxes の軸タイプが不足しています。")
            (axis_type_id,) = struct.unpack("<I", type_raw)
            axis["type"] = _SPACE_AXIS_TYPES.get(axis_type_id, f"TYPE_{axis_type_id}")
            axis["values"] = None
        else:
            # 非線形軸: points 個の double + 末尾 4 バイト
            n = axis["points"]
            vals_raw = buf.read(8 * n)
            if len(vals_raw) < 8 * n:
                raise ValueError("VGSpaceAxes の非線形軸データが不足しています。")
            axis["values"] = list(struct.unpack(f"<{n}d", vals_raw))
            buf.read(4)
            axis["type"] = None

        axes.append(axis)

    # type_code は実装依存のため、解析自体は軸内容を優先する
    _ = type_code
    return axes


def _parse_vg_data_axes(raw: bytes) -> list[dict[str, int]]:
    """VGDataAxes ストリームを解析する（複数スペクトル分割に使用）。"""
    if len(raw) < 8:
        return []
    buf = io.BytesIO(raw)
    _type_code, naxes = struct.unpack("<II", buf.read(8))
    axes = []
    for _ in range(naxes):
        chunk = buf.read(16)
        if len(chunk) < 16:
            break
        start, end, nspace, unknown = struct.unpack("<4I", chunk)
        axes.append(
            {
                "start": int(start),
                "end": int(end),
                "nspace": int(nspace),
                "unknown": int(unknown),
            }
        )
    return axes


def _infer_multi_spectrum_shape(
    total_points: int, data_axes: list[dict[str, int]]
) -> tuple[int, int]:
    """
    全点数と VGDataAxes から (スペクトル数, 1本あたり点数) を推定する。

    Returns
    -------
    num_spectra, points_per_spectrum
    """
    if total_points <= 0:
        return 0, 0

    # vgd_reader 互換: data_axes の特定オフセット解釈
    # end+1 が points / unknown 側がスペクトル数、という経験則
    if len(data_axes) >= 2:
        dim1 = data_axes[0]["end"] + 1
        dim2 = data_axes[1]["end"] + 1
        if dim1 > 0 and dim2 > 0 and dim1 * dim2 == total_points:
            # dim2 がスペクトル数、dim1 が点数、とみなす（dim2>1 のとき）
            if dim2 > 1:
                return dim2, dim1
            if dim1 > 1 and total_points % dim1 == 0:
                return total_points // dim1, dim1

    if len(data_axes) >= 1:
        pts = data_axes[0]["end"] + 1
        if pts > 0 and total_points % pts == 0:
            n_spec = total_points // pts
            if n_spec >= 1:
                return n_spec, pts

    return 1, total_points


def _extract_source_energy(ole: olefile.OleFileIO) -> float:
    """
    OLE プロパティから X 線源エネルギー (eV) を取得する。
    見つからない場合は Al Kα (1486.68 eV) を返す。
    """
    # 1) olefile のプロパティ API
    for path in (_AVANTAGE_PROP_STREAM, "\x05SummaryInformation"):
        try:
            if ole.exists(path):
                props = ole.getproperties(path, convert_time=True)
                for key in ("SourceEnergy", "SOURCEENERGY", "source_energy"):
                    if key in props:
                        val = float(props[key])
                        if 100.0 < val < 10000.0:
                            return val
                # 値だけ走査
                for val in props.values():
                    if isinstance(val, (int, float)) and 1480.0 < float(val) < 1490.0:
                        return float(val)
                    if isinstance(val, (int, float)) and 1250.0 < float(val) < 1260.0:
                        return float(val)
        except Exception:
            pass

    # 2) 生バイトから Al/Mg Kα 近傍の float32 を探索
    try:
        if ole.exists(_AVANTAGE_PROP_STREAM):
            raw = ole.openstream(_AVANTAGE_PROP_STREAM).read()
            for i in range(0, max(0, len(raw) - 4), 4):
                (val,) = struct.unpack("<f", raw[i : i + 4])
                if 1480.0 < val < 1490.0 or 1250.0 < val < 1260.0:
                    return float(val)
    except Exception:
        pass

    return _DEFAULT_SOURCE_ENERGY_EV


def _extract_pass_energy(ole: olefile.OleFileIO) -> Optional[float]:
    """プロパティストリームから Pass Energy らしき値を推定する。"""
    common = {10.0, 20.0, 35.0, 40.0, 50.0, 100.0, 150.0, 160.0, 200.0}
    try:
        if not ole.exists(_AVANTAGE_PROP_STREAM):
            return None
        raw = ole.openstream(_AVANTAGE_PROP_STREAM).read()
        for i in range(0, max(0, len(raw) - 4), 4):
            (val,) = struct.unpack("<f", raw[i : i + 4])
            if val in common:
                return float(val)
    except Exception:
        return None
    return None


def _energy_axis_from_space(
    space_axes: list[dict[str, Any]],
    num_points: int,
    source_energy: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    VGSpaceAxes から運動エネルギー・結合エネルギーの配列を構築する。

    Returns
    -------
    binding_energy, kinetic_energy, axis_mode
        axis_mode は "ke_to_be" / "already_be" / "fallback"
    """
    energy_axis = None
    for ax in space_axes:
        label = (ax.get("label") or "").lower()
        unit = (ax.get("unit") or "").lower()
        atype = ax.get("type")
        if atype == "ENERGY" or "energy" in label or unit == "ev":
            energy_axis = ax
            break
    if energy_axis is None and space_axes:
        energy_axis = space_axes[0]

    if energy_axis is None:
        # 軸情報が無い場合はダミーの BE インデックス
        be = np.arange(num_points, dtype=float)[::-1]
        return be, source_energy - be, "fallback"

    if energy_axis.get("values") is not None:
        x = np.asarray(energy_axis["values"], dtype=float)
    else:
        start = float(energy_axis["start"])
        step = float(energy_axis["width"])
        pts = int(energy_axis.get("points") or num_points)
        x = start + step * np.arange(pts, dtype=float)

    if len(x) != num_points:
        # 点数不一致時は線形補間で合わせる
        if len(x) >= 2:
            x = np.linspace(x[0], x[-1], num_points)
        else:
            x = np.arange(num_points, dtype=float)

    label = (energy_axis.get("label") or "").lower()
    # ラベルが Binding を含む場合は既に BE
    if "binding" in label or label.startswith("be"):
        be = x.astype(float)
        ke = source_energy - be
        return be, ke, "already_be"

    # 通常 Avantage は Kinetic Energy 軸
    ke = x.astype(float)
    be = source_energy - ke
    return be, ke, "ke_to_be"


def _guess_core_level_from_filename(filename: str) -> str:
    """ファイル名から軌道名らしき文字列を抽出する（例: O1s_Scan.vgd → O1s）。"""
    base = Path(filename).stem
    for suffix in (
        "_Scan", " Scan", "_Region", " Region", "_spectrum", " spectrum",
        "_core level", " core level",
    ):
        if base.lower().endswith(suffix.lower()):
            base = base[: -len(suffix)]
            break
    match = re.match(r"^([A-Z][a-z]?\d+[spdf]\d*)", base)
    if match:
        return match.group(1)
    return base.strip() or "Spectrum"


def _find_vgdata_stream_paths(ole: olefile.OleFileIO) -> list[list[str]]:
    """
    OLE 内の VGData ストリームパスを列挙する。

    トップレベルだけでなく、ストレージ配下に複数スペクトルが
    入っている VGX/VGD にも対応します。
    """
    paths: list[list[str]] = []
    for entry in ole.listdir(streams=True, storages=False):
        if entry and str(entry[-1]).lower() == "vgdata":
            paths.append(list(entry))
    # 安定した表示順
    paths.sort(key=lambda p: "/".join(p).lower())
    return paths


def _sibling_stream(ole: olefile.OleFileIO, vgdata_path: list[str], name: str) -> Optional[bytes]:
    """VGData と同じ親にある兄弟ストリームを読む。"""
    sibling = list(vgdata_path[:-1]) + [name]
    try:
        if ole.exists(sibling):
            return ole.openstream(sibling).read()
    except Exception:
        return None
    # トップレベルフォールバック
    try:
        if ole.exists(name):
            return ole.openstream(name).read()
    except Exception:
        return None
    return None


def parse_avantage_file(file_bytes: bytes, filename: str = "data.vgd") -> list[AvantageSpectrum]:
    """
    .vgd / .vgx（OLE2）バイト列から XPS スペクトル一覧を抽出する。

    Parameters
    ----------
    file_bytes : bytes
        アップロードファイルの生バイト列。
    filename : str
        表示名・軌道名推定に使うファイル名。

    Returns
    -------
    list[AvantageSpectrum]
        抽出されたスペクトル（1 本以上）。

    Raises
    ------
    ValueError
        OLE2 でない、または VGData が見つからない場合。
    """
    if len(file_bytes) < 8 or data_is_not_ole2(file_bytes):
        raise ValueError(
            f"「{filename}」は OLE2 コンパウンドファイルではありません。"
            " Thermo Fisher Avantage の .vgd / .vgx か確認してください。"
        )

    try:
        ole = olefile.OleFileIO(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"OLE2 として開けませんでした: {exc}") from exc

    try:
        vg_paths = _find_vgdata_stream_paths(ole)
        if not vg_paths:
            raise ValueError(
                "VGData ストリームが見つかりません。"
                " Avantage 形式の .vgd / .vgx か確認してください。"
            )

        source_energy = _extract_source_energy(ole)
        pass_energy = _extract_pass_energy(ole)
        core_guess = _guess_core_level_from_filename(filename)

        # メタデータ（任意）
        title = ""
        try:
            meta = ole.get_metadata()
            if meta and meta.title:
                title = str(meta.title).split("\x00")[0]
        except Exception:
            pass

        spectra: list[AvantageSpectrum] = []

        for path_i, vg_path in enumerate(vg_paths):
            raw_data = ole.openstream(vg_path).read()
            if len(raw_data) < 8:
                continue
            # 末尾の端数バイトは切り捨て
            usable = len(raw_data) - (len(raw_data) % 8)
            intensities_all = np.frombuffer(raw_data[:usable], dtype="<f8").astype(float)
            total_points = int(intensities_all.size)
            if total_points < 2:
                continue

            space_raw = _sibling_stream(ole, vg_path, "VGSpaceAxes")
            data_axes_raw = _sibling_stream(ole, vg_path, "VGDataAxes")

            space_axes: list[dict[str, Any]] = []
            if space_raw:
                try:
                    space_axes = _parse_vg_space_axes(space_raw)
                except Exception:
                    space_axes = []

            data_axes = _parse_vg_data_axes(data_axes_raw) if data_axes_raw else []
            num_spectra, points_per = _infer_multi_spectrum_shape(total_points, data_axes)
            if points_per <= 0 or num_spectra <= 0:
                num_spectra, points_per = 1, total_points

            # space axis の points を優先
            if space_axes:
                ax_pts = int(space_axes[0].get("points") or 0)
                if ax_pts > 1 and total_points % ax_pts == 0:
                    points_per = ax_pts
                    num_spectra = total_points // ax_pts

            parent = "/".join(vg_path[:-1]) if len(vg_path) > 1 else ""
            block_label = parent or core_guess or f"Block{path_i + 1}"

            for spec_i in range(num_spectra):
                start = spec_i * points_per
                end = start + points_per
                if end > total_points:
                    break
                y = intensities_all[start:end]
                be, ke, mode = _energy_axis_from_space(space_axes, points_per, source_energy)

                if num_spectra == 1 and len(vg_paths) == 1:
                    name = core_guess
                elif num_spectra == 1:
                    name = f"{block_label}"
                else:
                    name = f"{block_label} #{spec_i + 1}"

                spectra.append(
                    AvantageSpectrum(
                        name=name,
                        binding_energy=be,
                        intensity=y,
                        kinetic_energy=ke,
                        source_energy=source_energy,
                        pass_energy=pass_energy,
                        num_points=points_per,
                        meta={
                            "filename": filename,
                            "ole_path": "/".join(vg_path),
                            "title": title,
                            "axis_mode": mode,
                            "spectrum_index": spec_i,
                            "total_in_block": num_spectra,
                            "be_min": float(np.min(be)),
                            "be_max": float(np.max(be)),
                        },
                    )
                )

        if not spectra:
            raise ValueError("スペクトル強度を抽出できませんでした。")
        return spectra
    finally:
        try:
            ole.close()
        except Exception:
            pass


def data_is_not_ole2(file_bytes: bytes) -> bool:
    """OLE2 マジックナンバー（D0 CF 11 E0 A1 B1 1A E1）でない場合 True。"""
    magic = bytes.fromhex("D0CF11E0A1B11AE1")
    return file_bytes[:8] != magic


def load_avantage_via_ui(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Avantage（.vgd / .vgx）読み込み UI。

    複数スペクトルがある場合は selectbox で選択し、
    既存パイプライン用 DataFrame を返す。
    """
    st.subheader("Avantage データ（.vgd / .vgx）の読み込み")

    file_bytes = uploaded_file.getvalue()
    spectra = parse_avantage_file(file_bytes, filename=uploaded_file.name)

    labels = []
    for i, sp in enumerate(spectra):
        be_lo, be_hi = float(np.min(sp.binding_energy)), float(np.max(sp.binding_energy))
        labels.append(
            f"{i + 1}. {sp.name}  "
            f"(BE {be_lo:.1f}–{be_hi:.1f} eV, {sp.num_points} pts)"
        )

    if len(spectra) == 1:
        idx = 0
        st.caption(f"スペクトル: **{labels[0]}**（1件のみ）")
    else:
        choice = st.selectbox(
            "解析対象のスペクトルを選択",
            options=labels,
            help="サーベイや複数ナローが含まれる場合、解析したい 1 本を選んでください。",
        )
        idx = labels.index(choice)

    selected = spectra[idx]
    df = selected.to_dataframe()

    # プレビューとメタ情報
    meta_cols = st.columns(3)
    meta_cols[0].metric("Source Energy", f"{selected.source_energy:.2f} eV")
    meta_cols[1].metric(
        "Pass Energy",
        f"{selected.pass_energy:.1f} eV" if selected.pass_energy else "N/A",
    )
    meta_cols[2].metric("データ点数", f"{len(df)}")

    st.markdown("#### スペクトルプレビュー（先頭行）")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(
        f"軸変換: `{selected.meta.get('axis_mode', '?')}` / "
        f"OLE: `{selected.meta.get('ole_path', '')}`"
    )
    return df


def load_uploaded_xps_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    アップロードファイルの種類に応じて XPS 用 DataFrame を返す。

    - CSV: 従来どおり先頭2数値列を自動採用
    - Excel (.xlsx/.xls): シート選択・プレビュー・X/Y列選択
    - Avantage (.vgd/.vgx): OLE2 からスペクトル抽出・選択 UI

    Parameters
    ----------
    uploaded_file : UploadedFile
        Streamlit のアップロードオブジェクト。

    Returns
    -------
    pd.DataFrame or None
        解析可能な形式。UI 待ちの場合は None（呼び出し側で st.stop）。
    """
    filename = uploaded_file.name

    # ----- Avantage VGD / VGX -----
    if is_avantage_filename(filename):
        return load_avantage_via_ui(uploaded_file)

    # ----- Excel -----
    if is_excel_filename(filename):
        st.subheader("Excel データの読み込み設定")

        sheet_names = list_excel_sheet_names(uploaded_file)
        if len(sheet_names) == 0:
            raise ValueError("Excel ファイルにシートが見つかりませんでした。")

        if len(sheet_names) == 1:
            sheet_name = sheet_names[0]
            st.caption(f"シート: **{sheet_name}**（1件のみ）")
        else:
            sheet_name = st.selectbox(
                "読み込むシートを選択",
                options=sheet_names,
                help="複数シートがある場合、解析対象のシートを選んでください。",
            )

        raw_df = load_excel_sheet(uploaded_file, sheet_name)

        st.markdown("#### データプレビュー（先頭行）")
        st.dataframe(raw_df.head(10), use_container_width=True)
        st.caption(f"シート「{sheet_name}」: {raw_df.shape[0]} 行 × {raw_df.shape[1]} 列")

        columns = list(raw_df.columns)
        if len(columns) < 2:
            raise ValueError("Excel シートに列が2つ以上必要です。")

        default_x = guess_column_name(
            columns,
            keywords=["binding energy", "binding", "energy", "結合エネルギー", "結合", "be"],
            fallback_index=0,
        )
        default_y = guess_column_name(
            columns,
            keywords=["intensity", "counts", "cps", "強度", "count", "y"],
            fallback_index=1 if len(columns) > 1 else 0,
        )
        if default_y == default_x and len(columns) > 1:
            default_y = columns[1] if columns[0] == default_x else columns[0]

        col_x, col_y = st.columns(2)
        with col_x:
            x_col = st.selectbox(
                "X軸列（結合エネルギー / Binding Energy）",
                options=columns,
                index=columns.index(default_x),
                help="横軸にする列（単位: eV を想定）",
            )
        with col_y:
            y_col = st.selectbox(
                "Y軸列（強度 / Intensity）",
                options=columns,
                index=columns.index(default_y),
                help="縦軸にする列（counts など）",
            )

        return build_xps_dataframe_from_columns(raw_df, x_col, y_col)

    # ----- CSV（およびその他テキスト） -----
    return load_xps_file(uploaded_file)


def shirley_background(
    binding_energy: np.ndarray,
    intensity: np.ndarray,
    tol: float = 1e-5,
    max_iter: int = 50,
) -> np.ndarray:
    """
    Shirley 法による背景（ベースライン）を計算する。

    Shirley 背景は、ある結合エネルギー位置での背景強度が
    「それより高結合エネルギー側の積分強度」に比例するという仮定に基づきます。
    反復計算で収束させます。

    Parameters
    ----------
    binding_energy : np.ndarray
        結合エネルギー配列（降順・昇順どちらでも可。内部でソートします）。
    intensity : np.ndarray
        強度配列。
    tol : float
        収束判定の相対誤差閾値。
    max_iter : int
        最大反復回数。

    Returns
    -------
    np.ndarray
        入力と同じ順序の背景強度配列。
    """
    # 昇順にソートして計算し、最後に元の順序へ戻す
    order = np.argsort(binding_energy)
    inv_order = np.argsort(order)
    x = binding_energy[order]
    y = intensity[order].astype(float)

    # 両端の強度を背景の端点とする
    y_left = y[0]
    y_right = y[-1]

    background = np.linspace(y_left, y_right, len(y))
    for _ in range(max_iter):
        # 右側（高 BE 側）からの累積積分に比例する背景を構築
        # Shirley: B(E) = k * ∫_E^{Emax} [I(E') - B(E')] dE'
        dy = y - background
        # 台形則による累積積分（右端から左へ）
        integral = np.cumsum(dy[::-1])[::-1]
        # 全体積分で正規化し、端点を合わせる
        total = integral[0] if integral[0] != 0 else 1.0
        k = (y_left - y_right) / total
        new_bg = y_right + k * integral

        # 収束判定
        denom = np.max(np.abs(new_bg)) + 1e-12
        if np.max(np.abs(new_bg - background)) / denom < tol:
            background = new_bg
            break
        background = new_bg

    return background[inv_order]


def linear_background(
    binding_energy: np.ndarray,
    intensity: np.ndarray,
) -> np.ndarray:
    """
    スペクトル両端を結ぶ直線近似による背景を計算する。

    Parameters
    ----------
    binding_energy : np.ndarray
        結合エネルギー配列。
    intensity : np.ndarray
        強度配列。

    Returns
    -------
    np.ndarray
        直線背景強度配列。
    """
    x0, x1 = binding_energy[0], binding_energy[-1]
    y0, y1 = intensity[0], intensity[-1]
    if abs(x1 - x0) < 1e-12:
        return np.full_like(intensity, y0, dtype=float)
    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (binding_energy - x0)


def detect_peaks(
    binding_energy: np.ndarray,
    intensity: np.ndarray,
    prominence: float,
    distance: int,
) -> np.ndarray:
    """
    scipy.signal.find_peaks でピーク位置インデックスを検出する。

    Parameters
    ----------
    binding_energy : np.ndarray
        結合エネルギー（未使用だがインターフェース統一のため受け取る）。
    intensity : np.ndarray
        背景補正後の強度。
    prominence : float
        ピークの突出度の最小値。大きいほど小さなピークを無視する。
    distance : int
        隣接ピーク間の最小サンプル数。

    Returns
    -------
    np.ndarray
        検出されたピークのインデックス配列。
    """
    # XPS は結合エネルギーが右→左（高→低）で描かれることが多いが、
    # find_peaks は配列インデックス上で局所最大を探すため順序は問わない。
    peaks, _ = find_peaks(intensity, prominence=prominence, distance=distance)
    return peaks


def assign_orbitals(
    peak_energies: np.ndarray,
    tolerance: float = ASSIGN_TOLERANCE_EV,
) -> list[str]:
    """
    検出ピークの結合エネルギーを xps_database.json の参照データと照合し、
    軌道候補をアサインする。

    判定優先順位:
      1. ピークが各軌道の search_window_ev（探索ウィンドウ）内にある候補から、
         代表結合エネルギーに最も近い軌道を選ぶ
      2. ウィンドウ内に無い場合、代表値との差が tolerance 以下の最近傍を採用
      3. どちらにも当てはまらなければ Unknown

    Parameters
    ----------
    peak_energies : np.ndarray
        検出ピークの結合エネルギー（eV）。
    tolerance : float
        ウィンドウ外ピーク向けのフォールバック許容差（eV）。
        UI スライダーから渡され、JSON の default_assign_tolerance_ev が初期値。

    Returns
    -------
    list[str]
        各ピークに対応する軌道名（候補なしは Unknown）。
    """
    assigned: list[str] = []
    orbital_list = list(ORBITAL_RECORDS.values())

    for energy in peak_energies:
        # --- 1) 探索ウィンドウ内の候補 ---
        in_window: list[tuple[str, float]] = []
        for orb in orbital_list:
            wmin = float(orb["search_window_ev"]["min"])
            wmax = float(orb["search_window_ev"]["max"])
            if wmin <= energy <= wmax:
                be_center = float(orb["binding_energy_ev"])
                in_window.append((orb["name"], abs(be_center - energy)))

        if in_window:
            in_window.sort(key=lambda t: t[1])
            assigned.append(in_window[0][0])
            continue

        # --- 2) 許容誤差フォールバック ---
        best_name = UNKNOWN_LABEL
        best_diff = float("inf")
        for orb in orbital_list:
            diff = abs(float(orb["binding_energy_ev"]) - energy)
            if diff < best_diff:
                best_diff = diff
                best_name = orb["name"]
        if best_diff <= tolerance:
            assigned.append(best_name)
        else:
            assigned.append(UNKNOWN_LABEL)

    return assigned


def fit_peaks(
    binding_energy: np.ndarray,
    intensity: np.ndarray,
    peak_indices: np.ndarray,
    orbital_labels: list[str],
) -> tuple[Optional[object], Optional[pd.DataFrame], Optional[np.ndarray], list[np.ndarray]]:
    """
    lmfit の PseudoVoigt（ガウス・ローレンツ複合）モデルで複数ピークを同時フィッティングする。

    PseudoVoigt はガウス型とローレンツ型の線形結合で、
    XPS ピーク形状の近似に広く使われます。

    Parameters
    ----------
    binding_energy : np.ndarray
        結合エネルギー配列。
    intensity : np.ndarray
        背景補正後の強度配列。
    peak_indices : np.ndarray
        初期ピーク位置のインデックス。
    orbital_labels : list[str]
        各ピークの軌道ラベル（結果テーブル・色分けに使用）。

    Returns
    -------
    result : lmfit.model.ModelResult or None
        フィット結果オブジェクト。
    params_df : pd.DataFrame or None
        各ピークの位置・FWHM・面積などをまとめた表。
    best_fit : np.ndarray or None
        合成フィッティング曲線。
    components : list[np.ndarray]
        各ピーク成分の強度配列リスト。
    """
    if len(peak_indices) == 0:
        return None, None, None, []

    # スペクトル幅から FWHM の初期値を推定
    x_span = abs(binding_energy.max() - binding_energy.min())
    default_sigma = max(x_span / 50.0, 0.3)

    model = ConstantModel(prefix="bg_")
    params = model.make_params(c=0.0)
    # 背景定数はほぼゼロ（すでに背景補正済み）に固定気味にする
    params["bg_c"].set(value=0.0, vary=True, min=-np.max(intensity) * 0.05, max=np.max(intensity) * 0.05)

    for i, idx in enumerate(peak_indices):
        prefix = f"p{i}_"
        peak_model = PseudoVoigtModel(prefix=prefix)
        model = model + peak_model

        center0 = float(binding_energy[idx])
        height0 = float(max(intensity[idx], 1e-6))
        # amplitude ≈ height * sigma * √(2π) のオーダーで初期値を設定
        amp0 = height0 * default_sigma * np.sqrt(2 * np.pi)

        peak_params = peak_model.make_params(
            amplitude=amp0,
            center=center0,
            sigma=default_sigma,
            fraction=0.5,  # ガウスとローレンツの混合比（0=純ガウス, 1=純ローレンツ）
        )
        # パラメータの探索範囲を制限（発散防止）
        peak_params[f"{prefix}center"].set(min=center0 - 3.0, max=center0 + 3.0)
        peak_params[f"{prefix}sigma"].set(min=0.1, max=max(x_span / 5.0, 1.0))
        peak_params[f"{prefix}amplitude"].set(min=0.0)
        peak_params[f"{prefix}fraction"].set(min=0.0, max=1.0)
        params.update(peak_params)

    try:
        result = model.fit(intensity, params, x=binding_energy)
    except Exception as exc:
        st.warning(f"フィッティングに失敗しました: {exc}")
        return None, None, None, []

    # 各ピーク成分を個別に評価
    components: list[np.ndarray] = []
    rows = []
    for i, label in enumerate(orbital_labels):
        prefix = f"p{i}_"
        comp = result.eval_components(x=binding_energy).get(prefix)
        if comp is None:
            continue
        components.append(comp)

        center = result.params[f"{prefix}center"].value
        sigma = result.params[f"{prefix}sigma"].value
        fraction = result.params[f"{prefix}fraction"].value
        amplitude = result.params[f"{prefix}amplitude"].value
        # PseudoVoigt の FWHM ≈ 2σ（近似）。より正確には fraction 依存だが実用上十分。
        fwhm = 2.0 * sigma
        # 面積は amplitude そのもの（lmfit PseudoVoigt の定義）
        area = amplitude

        rows.append(
            {
                "ピーク番号": i + 1,
                "軌道候補": label,
                "位置 (eV)": round(center, 3),
                "FWHM (eV)": round(fwhm, 3),
                "面積強度": round(area, 3),
                "ガウス/ローレンツ混合比": round(fraction, 3),
            }
        )

    params_df = pd.DataFrame(rows)
    best_fit = result.best_fit
    return result, params_df, best_fit, components


def build_spectrum_figure(
    binding_energy: np.ndarray,
    raw_intensity: np.ndarray,
    background: np.ndarray,
    corrected: np.ndarray,
    peak_indices: np.ndarray,
    orbital_labels: list[str],
    best_fit: Optional[np.ndarray],
    components: list[np.ndarray],
    show_corrected: bool = True,
) -> go.Figure:
    """
    Plotly でインタラクティブな XPS スペクトル図を構築する。

    描画レイヤ（下から上）:
      1. 元データ
      2. 背景曲線
      3. 背景補正後スペクトル（任意）
      4. 各軌道ピーク成分（ORBITAL_COLORS で色分け）
      5. 合成フィッティング曲線
      6. ピーク位置マーカー

    Parameters
    ----------
    binding_energy, raw_intensity, background, corrected : np.ndarray
        各スペクトルデータ。
    peak_indices : np.ndarray
        検出ピークのインデックス。
    orbital_labels : list[str]
        各ピークの軌道名（色の決定に使用）。
    best_fit : np.ndarray or None
        合成フィット曲線。
    components : list[np.ndarray]
        分離された各ピーク成分。
    show_corrected : bool
        背景補正後スペクトルを重ね描きするか。

    Returns
    -------
    go.Figure
        Plotly 図オブジェクト。
    """
    fig = go.Figure()

    # --- 元データ ---
    fig.add_trace(
        go.Scatter(
            x=binding_energy,
            y=raw_intensity,
            mode="lines",
            name="元データ",
            line=dict(color=ROLE_COLORS["raw"], width=1.5),
            hovertemplate="BE: %{x:.2f} eV<br>Intensity: %{y:.1f}<extra>元データ</extra>",
        )
    )

    # --- 背景 ---
    fig.add_trace(
        go.Scatter(
            x=binding_energy,
            y=background,
            mode="lines",
            name="背景",
            line=dict(color=ROLE_COLORS["background"], width=1.5, dash="dash"),
            hovertemplate="BE: %{x:.2f} eV<br>BG: %{y:.1f}<extra>背景</extra>",
        )
    )

    # --- 背景補正後 ---
    if show_corrected:
        fig.add_trace(
            go.Scatter(
                x=binding_energy,
                y=corrected,
                mode="lines",
                name="背景補正後",
                line=dict(color=ROLE_COLORS["corrected"], width=1.2),
                hovertemplate="BE: %{x:.2f} eV<br>Corr: %{y:.1f}<extra>背景補正後</extra>",
                visible="legendonly",  # 初期は凡例クリックで表示（グラフを見やすく）
            )
        )

    # --- 各ピーク成分（軌道ごとに固定色） ---
    for i, (comp, label) in enumerate(zip(components, orbital_labels)):
        color = get_orbital_color(label, index=i)
        display_name = f"{label} (#{i + 1})" if label != "Unknown" else f"Peak #{i + 1}"
        fig.add_trace(
            go.Scatter(
                x=binding_energy,
                y=comp,
                mode="lines",
                name=display_name,
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=_hex_to_rgba(color, alpha=0.25),
                hovertemplate=(
                    f"{display_name}<br>"
                    "BE: %{x:.2f} eV<br>Intensity: %{y:.1f}<extra></extra>"
                ),
            )
        )

    # --- 合成フィット曲線 ---
    if best_fit is not None:
        fig.add_trace(
            go.Scatter(
                x=binding_energy,
                y=best_fit + background,  # 背景を戻して元データと比較しやすくする
                mode="lines",
                name="合成フィット (+背景)",
                line=dict(color=ROLE_COLORS["fit_sum"], width=2, dash="dot"),
                hovertemplate="BE: %{x:.2f} eV<br>Fit: %{y:.1f}<extra>合成フィット</extra>",
            )
        )

    # --- ピーク位置マーカー ---
    if len(peak_indices) > 0:
        marker_colors = [
            get_orbital_color(orbital_labels[i], index=i) for i in range(len(peak_indices))
        ]
        fig.add_trace(
            go.Scatter(
                x=binding_energy[peak_indices],
                y=raw_intensity[peak_indices],
                mode="markers",
                name="検出ピーク",
                marker=dict(
                    color=marker_colors,
                    size=10,
                    symbol="triangle-down",
                    line=dict(width=1, color=OKABE_ITO["black"]),
                ),
                text=orbital_labels,
                hovertemplate="軌道: %{text}<br>BE: %{x:.2f} eV<br>I: %{y:.1f}<extra>検出ピーク</extra>",
            )
        )

    # XPS 慣例: 横軸は結合エネルギーを右→左（高→低）に表示
    fig.update_layout(
        title="XPS スペクトル解析結果",
        xaxis_title="Binding Energy (eV)",
        yaxis_title="Intensity (a.u.)",
        xaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="closest",
        template="plotly_white",
        height=560,
        margin=dict(l=60, r=30, t=80, b=60),
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float = 0.3) -> str:
    """
    HEX カラーコードを rgba() 文字列に変換する（塗りつぶし用）。

    Parameters
    ----------
    hex_color : str
        "#RRGGBB" 形式の色コード。
    alpha : float
        不透明度（0〜1）。

    Returns
    -------
    str
        "rgba(r, g, b, a)" 形式の文字列。
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def make_sample_data() -> pd.DataFrame:
    """
    デモ用の合成 XPS スペクトルを生成する。
    C 1s / O 1s / N 1s 付近に擬似ピークを配置しています。
    """
    be = np.linspace(600, 250, 700)
    intensity = (
        80
        + 400 * np.exp(-0.5 * ((be - 531.0) / 1.4) ** 2)   # O 1s
        + 250 * np.exp(-0.5 * ((be - 400.0) / 1.2) ** 2)   # N 1s
        + 500 * np.exp(-0.5 * ((be - 284.8) / 1.1) ** 2)   # C 1s
        + np.random.default_rng(42).normal(0, 8, size=be.shape)
    )
    # 簡易 Shirley 風の階段状背景を加算
    intensity += 30 * (be - be.min()) / (be.max() - be.min())
    return pd.DataFrame({"Binding Energy": be, "Intensity": intensity})


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    """アプリのメインエントリポイント。サイドバー設定と解析フローを制御する。"""
    st.set_page_config(
        page_title="XPS 自動解析",
        page_icon="⚛️",
        layout="wide",
    )

    st.title("XPS データ自動解析アプリ")
    st.caption(
        "CSV / Excel（.xlsx, .xls）/ Avantage（.vgd, .vgx）の XPS スペクトルを読み込み、"
        "背景補正・ピーク検出・軌道アサイン・PseudoVoigt フィッティングを行います。"
        "元素・軌道の参照値と配色は xps_database.json（Okabe-Ito）で一元管理しています。"
    )

    # ----- サイドバー: 解析パラメータ -----
    with st.sidebar:
        st.header("解析設定")

        uploaded = st.file_uploader(
            "XPS データファイル（CSV / Excel / VGD / VGX）",
            type=["csv", "xlsx", "xls", "vgx", "vgd"],
            help=(
                "CSV: 1列目 Binding Energy, 2列目 Intensity。"
                "Excel: シートと X/Y 列を選択。"
                "VGD/VGX: Thermo Avantage の OLE2 形式からスペクトルを抽出。"
            ),
        )
        use_sample = st.checkbox("サンプルデータを使う", value=uploaded is None)

        st.subheader("背景補正")
        bg_method = st.radio(
            "手法",
            options=["Shirley", "直線近似"],
            index=0,
            help="Shirley: 散乱二次電子を考慮した反復背景。直線近似: 両端を結ぶ単純な線。",
        )

        st.subheader("ピーク検出")
        # prominence はデータスケール依存のため、相対値で指定し後で絶対値に変換
        prominence_ratio = st.slider(
            "ピーク突出度（相対）",
            min_value=0.01,
            max_value=0.50,
            value=0.08,
            step=0.01,
            help="スペクトル最大強度に対する割合。大きいほど小さなピークを無視します。",
        )
        peak_distance = st.slider(
            "ピーク間最小距離（ポイント）",
            min_value=3,
            max_value=80,
            value=15,
            help="隣接ピークを分離するための最小サンプル間隔。",
        )
        assign_tol = st.slider(
            "軌道アサイン許容誤差 (eV)",
            min_value=1.0,
            max_value=15.0,
            value=float(ASSIGN_TOLERANCE_EV),
            step=0.5,
            help=(
                "探索ウィンドウ外のピーク向けフォールバック。"
                "通常は xps_database.json の search_window_ev を優先します。"
            ),
        )

        st.subheader("前処理")
        smooth = st.checkbox("Savitzky-Golay 平滑化を適用", value=False)
        if smooth:
            window = st.slider("平滑化ウィンドウ長（奇数）", 5, 51, 11, step=2)
            polyorder = st.slider("多項式次数", 2, 5, 3)

        run_fit = st.checkbox("ピークフィッティングを実行", value=True)

        st.markdown("---")
        st.caption(
            f"参照DB: `{DATABASE_PATH.name}`（軌道 {len(ORBITAL_RECORDS)} 件）"
        )
        with st.expander("登録軌道・配色（xps_database.json）"):
            st.markdown(
                "色覚バリアフリーの Okabe-Ito パレットを使用。"
                "元素追加は JSON の `orbitals` に追記するだけで反映されます。"
            )
            # JSON に登録された全軌道の色見本（Unknown 以外）
            swatches = ""
            for name, color in ORBITAL_COLORS.items():
                if name == UNKNOWN_LABEL:
                    continue
                swatches += (
                    f'<span style="display:inline-block;width:12px;height:12px;'
                    f'background:{color};margin-right:6px;border:1px solid #333;"></span>'
                    f"{name}<br>"
                )
            st.markdown(swatches, unsafe_allow_html=True)

    # ----- データ読み込み -----
    try:
        if uploaded is not None and not use_sample:
            # CSV は従来どおり自動整形、Excel はシート・列選択 UI 付き
            df = load_uploaded_xps_data(uploaded)
            if df is None:
                st.info("解析を続けるには、Excel のシートと X/Y 列を選択してください。")
                st.stop()
            st.success(
                f"ファイルを読み込みました: **{uploaded.name}**（{len(df)} 点）"
            )
        else:
            df = make_sample_data()
            st.info("サンプルデータ（C 1s / N 1s / O 1s 付近の合成スペクトル）を表示しています。")
    except Exception as exc:
        st.error(f"ファイル読み込みエラー: {exc}")
        st.stop()

    be = df["Binding Energy"].to_numpy(dtype=float)
    intensity = df["Intensity"].to_numpy(dtype=float)

    if smooth:
        # window_length はデータ点数以下の奇数である必要がある
        w = min(window, len(intensity) if len(intensity) % 2 == 1 else len(intensity) - 1)
        w = max(w, polyorder + 2)
        if w % 2 == 0:
            w -= 1
        intensity_proc = savgol_filter(intensity, window_length=w, polyorder=polyorder)
    else:
        intensity_proc = intensity.copy()

    # ----- 背景補正 -----
    if bg_method == "Shirley":
        background = shirley_background(be, intensity_proc)
    else:
        background = linear_background(be, intensity_proc)

    corrected = intensity_proc - background
    # 負値は物理的に意味が薄いため 0 クリップ（表示・検出の安定化）
    corrected = np.clip(corrected, 0, None)

    # ----- ピーク検出 & 軌道アサイン -----
    prominence_abs = float(np.max(corrected) * prominence_ratio) if np.max(corrected) > 0 else 1.0
    peak_idx = detect_peaks(be, corrected, prominence=prominence_abs, distance=peak_distance)
    peak_energies = be[peak_idx] if len(peak_idx) else np.array([])
    orbital_labels = assign_orbitals(peak_energies, tolerance=assign_tol) if len(peak_idx) else []

    # ----- フィッティング -----
    params_df = None
    best_fit = None
    components: list[np.ndarray] = []
    if run_fit and len(peak_idx) > 0:
        with st.spinner("lmfit でピークフィッティング中..."):
            _, params_df, best_fit, components = fit_peaks(
                be, corrected, peak_idx, orbital_labels
            )

    # ----- グラフ描画 -----
    fig = build_spectrum_figure(
        binding_energy=be,
        raw_intensity=intensity_proc,
        background=background,
        corrected=corrected,
        peak_indices=peak_idx,
        orbital_labels=orbital_labels,
        best_fit=best_fit,
        components=components,
        show_corrected=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ----- 結果テーブル -----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("検出ピーク一覧")
        if len(peak_idx) == 0:
            st.warning("ピークが検出されませんでした。突出度や距離パラメータを調整してください。")
        else:
            detect_df = pd.DataFrame(
                {
                    "ピーク番号": np.arange(1, len(peak_idx) + 1),
                    "結合エネルギー (eV)": np.round(peak_energies, 3),
                    "強度": np.round(corrected[peak_idx], 1),
                    "軌道候補": orbital_labels,
                    "色コード": [get_orbital_color(lb, i) for i, lb in enumerate(orbital_labels)],
                }
            )
            st.dataframe(detect_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("フィッティング結果")
        if params_df is None or params_df.empty:
            st.info("フィッティング未実行、またはピークなしです。")
        else:
            st.dataframe(params_df, use_container_width=True, hide_index=True)

    # ----- 参照テーブル・生データ（JSON データベース由来） -----
    with st.expander("参照結合エネルギーテーブル（xps_database.json）"):
        ref_rows = []
        for orb in sorted(
            XPS_DB["orbitals"],
            key=lambda o: -float(o["binding_energy_ev"]),
        ):
            ref_rows.append(
                {
                    "軌道": orb["name"],
                    "代表 BE (eV)": orb["binding_energy_ev"],
                    "探索ウィンドウ min": orb["search_window_ev"]["min"],
                    "探索ウィンドウ max": orb["search_window_ev"]["max"],
                    "色": orb["color"],
                    "カテゴリ": orb.get("category", ""),
                }
            )
        ref_df = pd.DataFrame(ref_rows)
        st.dataframe(ref_df, use_container_width=True, hide_index=True)
        st.caption(
            "※ 文献概算値です。化学状態により数 eV シフトすることがあります。"
            " 新しい元素は xps_database.json の orbitals に追加してください。"
        )

    with st.expander("読み込みデータ（先頭 20 行）"):
        st.dataframe(df.head(20), use_container_width=True)

    # 結果 CSV ダウンロード
    if params_df is not None and not params_df.empty:
        csv_bytes = params_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "フィッティング結果を CSV ダウンロード",
            data=csv_bytes,
            file_name="xps_fit_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
