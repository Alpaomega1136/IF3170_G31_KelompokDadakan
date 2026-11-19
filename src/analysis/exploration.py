import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def overview(data):
    return pd.DataFrame({
        "tipe": data.dtypes.astype(str),
        "nilai_unik": data.nunique(),
        "kosong_train": data.isna().sum(),
        "persen_kosong_train": data.isna().mean().mul(100).round(2),
    })


def plot_distributions(data):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    data["is_canceled"].value_counts().sort_index().plot.bar(ax=axes[0], rot=0)
    axes[0].set(title="Jumlah per kelas", xlabel="is_canceled", ylabel="Jumlah pemesanan")
    data["days_before_arrival"].plot.hist(bins=30, ax=axes[1])
    axes[1].set(title="Jarak pemesanan", xlabel="Hari", ylabel="Jumlah pemesanan")
    data["nightly_rate_idr"].plot.hist(bins=30, ax=axes[2])
    axes[2].set(title="Harga per malam", xlabel="IDR", ylabel="Jumlah pemesanan")
    plt.tight_layout()
    plt.show()


def check_data(train, test):
    rows = []
    for name, data in [("train", train), ("test", test)]:
        assert data["id"].notna().all() and data["id"].is_unique, f"Periksa ID pada {name}"
        rows.append({
            "data": name, "baris": len(data),
            "baris_dengan_nilai_kosong": data.isna().any(axis=1).sum(),
            "duplikat_seluruh_kolom": data.duplicated().sum(),
            "fitur_sama_id_berbeda": data.drop(columns=["id", "is_canceled"], errors="ignore").duplicated().sum(),
        })
    assert train["is_canceled"].isin([0, 1]).all(), "Target harus 0 atau 1 tanpa nilai kosong"
    return pd.DataFrame(rows).set_index("data")


def heatmap(data):
    correlation = data.drop(columns="id").corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
    ax.set_yticks(range(len(correlation.index)), correlation.index)
    ax.set_title("Korelasi fitur numerik dan target - data training")
    for row in range(len(correlation)):
        for column in range(len(correlation.columns)):
            value = correlation.iloc[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                    color="white" if abs(value) > 0.55 else "black", fontsize=7)
    fig.colorbar(image, ax=ax, label="Korelasi Pearson", shrink=0.8)
    plt.tight_layout()
    plt.show()


def missing_values(X_train, X_valid, X_test):
    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = X_train.select_dtypes(exclude="number").columns.tolist()
    fill = pd.concat([X_train[numeric].median(), X_train[categorical].mode().iloc[0]]).reindex(X_train.columns)
    assert fill.notna().all(), "Ada kolom training yang seluruh nilainya kosong"
    assert fill[["needs_parking", "is_returning_guest"]].isin([0, 1]).all()
    return pd.DataFrame({"kosong_training": X_train.isna().sum(),
                         "kosong_validasi": X_valid.isna().sum(),
                         "kosong_test": X_test.isna().sum(),
                         "nilai_pengisi_training": fill})


def duplicates(train_set, val_set, X_train, X_valid):
    print("Duplikat lengkap training:", train_set.duplicated().sum())
    print("Duplikat lengkap validasi:", val_set.duplicated().sum())
    print("Fitur sama dalam training:", X_train.duplicated().sum())
    print("Fitur sama dalam validasi:", X_valid.duplicated().sum())
    return train_set.loc[train_set.duplicated(subset=X_train.columns.tolist(), keep=False)].head(6)


def outliers(X_train, X_valid, X_test):
    columns = ["nightly_rate_idr", "days_before_arrival", "weekend_nights", "weekday_nights"]
    q1, q3 = X_train[columns].quantile(0.25), X_train[columns].quantile(0.75)
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    for name, data in [("training", X_train), ("validasi", X_valid), ("test", X_test)]:
        print(name, "| harga nol:", data["nightly_rate_idr"].eq(0).sum(),
              "| total malam nol:", (data["weekend_nights"] + data["weekday_nights"]).eq(0).sum())
    return pd.DataFrame({"batas_bawah": lower, "batas_atas": upper,
                         "jumlah_di_luar_batas": (X_train[columns].lt(lower) | X_train[columns].gt(upper)).sum()})


def validate_values(X_train, X_valid, X_test):
    valid_categories = {
        "meal_package": ["Room Only", "Breakfast", "Half Board", "Full Board"],
        "room_category": [f"Type {letter}" for letter in "ABCDEFG"],
        "booking_channel": ["Online Travel Agent", "Offline Agent", "Corporate", "Airline", "Complimentary"],
    }
    numeric = X_train.select_dtypes(include="number").columns.tolist()
    for name, data in [("training", X_train), ("validasi", X_valid), ("test", X_test)]:
        numbers = data[numeric]
        assert (numbers.isna() | np.isfinite(numbers)).all().all(), f"Nilai tak hingga pada {name}"
        assert (numbers.isna() | numbers.ge(0)).all().all(), f"Angka negatif pada {name}"
        assert (numbers.isna() | numbers.mod(1).eq(0)).all().all(), f"Hitungan pecahan pada {name}"
        for column in ["needs_parking", "is_returning_guest"]:
            assert data[column].dropna().isin([0, 1]).all(), f"Nilai biner tidak valid: {name}, {column}"
        for column, allowed in valid_categories.items():
            assert data[column].dropna().isin(allowed).all(), f"Kategori tidak valid: {name}, {column}"
        date = pd.to_datetime({"year": data["arrival_year"], "month": data["arrival_month"], "day": data["arrival_day"]}, errors="coerce")
        assert date.notna().all(), f"Tanggal kosong atau tidak valid pada {name}"
        print(name, ": pemeriksaan nilai valid lulus")
