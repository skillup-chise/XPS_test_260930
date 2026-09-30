"""
XPS（X線光電子分光）データ自動解析 Web アプリ
==============================================
Streamlit + Plotly + lmfit を用いて、ブラウザ上で XPS スペクトルの
読み込み・背景補正・ピーク検出・フィッティング・可視化を行います。

配色ルール:
  - Okabe-Ito パレット（色覚バリアフリー）を使用
  - 元素・軌道名ごとに色を固定（ORBITAL_COLORS）し、一貫性を確保
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lmfit.models import PseudoVoigtModel, ConstantModel
from scipy.signal import find_peaks, savgol_filter


# ---------------------------------------------------------------------------
# 定数・参照テーブル
# ---------------------------------------------------------------------------

# Okabe-Ito パレット（色盲・色弱でも区別しやすい色セット）
# 赤と緑の混同を避け、コントラストを確保した配色です。
OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
    "gray": "#999999",
}

# グラフ上の役割ごとの固定色（データ曲線・背景・合成曲線など）
ROLE_COLORS = {
    "raw": OKABE_ITO["black"],           # 元データ
    "background": OKABE_ITO["gray"],     # 背景（ベースライン）
    "corrected": OKABE_ITO["blue"],      # 背景補正後スペクトル
    "fit_sum": OKABE_ITO["vermillion"],  # 合成フィッティング曲線
    "peak_marker": OKABE_ITO["orange"],  # 検出ピーク位置マーカー
}

# 元素・軌道名 → 色コードの 1対1 対応辞書
# 同じ軌道は常に同じ色で描画され、解釈の一貫性を保ちます。
ORBITAL_COLORS: dict[str, str] = {
    "C 1s": OKABE_ITO["orange"],
    "O 1s": OKABE_ITO["sky_blue"],
    "N 1s": OKABE_ITO["bluish_green"],
    "Ti 2p": OKABE_ITO["yellow"],
    "Ti 2p3/2": OKABE_ITO["yellow"],
    "Ti 2p1/2": "#B3A000",  # Ti 2p と同系統のやや暗い黄
    "Si 2p": OKABE_ITO["blue"],
    "Si 2s": "#4A90A4",
    "Al 2p": OKABE_ITO["vermillion"],
    "Al 2s": "#A04500",
    "Fe 2p": OKABE_ITO["reddish_purple"],
    "Fe 2p3/2": OKABE_ITO["reddish_purple"],
    "Fe 2p1/2": "#9B5A7A",
    "S 2p": "#882255",
    "Cl 2p": "#44AA99",
    "Ca 2p": "#117733",
    "Na 1s": "#332288",
    "F 1s": "#AA4499",
    "P 2p": "#661100",
    "Unknown": OKABE_ITO["gray"],  # 未アサインピーク用
}

# 未登録軌道に順次割り当てる予備色（Okabe-Ito を循環使用）
_FALLBACK_PALETTE = [
    OKABE_ITO["orange"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["yellow"],
    OKABE_ITO["blue"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["reddish_purple"],
]

# 主要元素・軌道の代表的な結合エネルギー（eV）テーブル
# 文献値の概算であり、化学シフトにより実際の位置は前後します。
REFERENCE_BINDING_ENERGIES: dict[str, float] = {
    "Na 1s": 1071.0,
    "F 1s": 685.0,
    "O 1s": 531.0,
    "Ti 2p1/2": 460.0,
    "Ti 2p3/2": 454.0,
    "N 1s": 400.0,
    "Ca 2p": 347.0,
    "C 1s": 284.8,
    "Cl 2p": 199.0,
    "S 2p": 164.0,
    "Si 2s": 150.0,
    "P 2p": 133.0,
    "Al 2s": 119.0,
    "Si 2p": 99.0,
    "Al 2p": 73.0,
    "Fe 2p1/2": 720.0,
    "Fe 2p3/2": 707.0,
}

# 軌道アサインの許容誤差（eV）
# 検出ピーク位置が参照値からこの範囲内なら候補として採用します。
ASSIGN_TOLERANCE_EV = 5.0


# ---------------------------------------------------------------------------
# ユーティリティ関数
# ---------------------------------------------------------------------------

def get_orbital_color(orbital_name: str, index: int = 0) -> str:
    """
    軌道名に対応する固定色を返す。

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


def load_uploaded_xps_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    アップロードファイルの種類に応じて XPS 用 DataFrame を返す。

    - CSV / TXT: 従来どおり先頭2数値列を自動採用
    - Excel: シート選択・プレビュー・X/Y列選択の UI を表示し、
             ユーザー選択に基づいて整形（未確定時は None を返す）

    Parameters
    ----------
    uploaded_file : UploadedFile
        Streamlit のアップロードオブジェクト。

    Returns
    -------
    pd.DataFrame or None
        解析可能な形式。Excel で列未選択などの場合は None（呼び出し側で st.stop）。
    """
    filename = uploaded_file.name

    # ----- CSV / TXT: 既存ロジック -----
    if not is_excel_filename(filename):
        return load_xps_file(uploaded_file)

    # ----- Excel: シート選択 → プレビュー → 列選択 -----
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

    # 列名から Binding Energy / Intensity らしき列を初期選択
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
    # X/Y が同じ列に推定された場合は Y を別列へずらす
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
    検出ピークの結合エネルギーを参照テーブルと照合し、軌道候補をアサインする。

    Parameters
    ----------
    peak_energies : np.ndarray
        検出ピークの結合エネルギー（eV）。
    tolerance : float
        参照値との許容差（eV）。この範囲内で最も近い軌道を採用。

    Returns
    -------
    list[str]
        各ピークに対応する軌道名（候補なしは "Unknown"）。
    """
    assigned: list[str] = []
    ref_names = list(REFERENCE_BINDING_ENERGIES.keys())
    ref_values = np.array([REFERENCE_BINDING_ENERGIES[n] for n in ref_names])

    for energy in peak_energies:
        diffs = np.abs(ref_values - energy)
        best_idx = int(np.argmin(diffs))
        if diffs[best_idx] <= tolerance:
            assigned.append(ref_names[best_idx])
        else:
            assigned.append("Unknown")
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
        "CSV / Excel（.xlsx, .xls）の XPS スペクトルを読み込み、背景補正・ピーク検出・"
        "軌道アサイン・PseudoVoigt フィッティングを行います。"
        "配色は Okabe-Ito パレット（色覚バリアフリー）を使用し、"
        "元素・軌道ごとに色を固定しています。"
    )

    # ----- サイドバー: 解析パラメータ -----
    with st.sidebar:
        st.header("解析設定")

        uploaded = st.file_uploader(
            "XPS データファイル（CSV / Excel）",
            type=["csv", "xlsx", "xls"],
            help=(
                "CSV: 1列目 Binding Energy, 2列目 Intensity。"
                "Excel: シートと X/Y 列を画面上で選択できます。"
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
            help="参照結合エネルギーとの差がこの値以下なら軌道候補として採用。",
        )

        st.subheader("前処理")
        smooth = st.checkbox("Savitzky-Golay 平滑化を適用", value=False)
        if smooth:
            window = st.slider("平滑化ウィンドウ長（奇数）", 5, 51, 11, step=2)
            polyorder = st.slider("多項式次数", 2, 5, 3)

        run_fit = st.checkbox("ピークフィッティングを実行", value=True)

        st.markdown("---")
        with st.expander("配色ルール（Okabe-Ito）"):
            st.markdown(
                "色覚バリアフリーのため Okabe-Ito パレットを使用し、"
                "軌道ごとに色を固定しています。"
            )
            # 主要軌道の色見本を表示
            swatches = ""
            for name, color in list(ORBITAL_COLORS.items())[:10]:
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

    # ----- 参照テーブル・生データ -----
    with st.expander("参照結合エネルギーテーブル"):
        ref_df = pd.DataFrame(
            [
                {
                    "軌道": name,
                    "代表 BE (eV)": be_val,
                    "色": ORBITAL_COLORS.get(name, ORBITAL_COLORS["Unknown"]),
                }
                for name, be_val in sorted(
                    REFERENCE_BINDING_ENERGIES.items(), key=lambda x: -x[1]
                )
            ]
        )
        st.dataframe(ref_df, use_container_width=True, hide_index=True)
        st.caption("※ 文献概算値です。化学状態により数 eV シフトすることがあります。")

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
