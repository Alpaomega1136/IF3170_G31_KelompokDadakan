import pandas as pd


def isi_lead_kanal(data, referensi, hari, metode, indikator):
    hasil = data.copy()
    if indikator:
        hasil["lead_missing"] = hasil["days_before_arrival"].isna().astype(int)
    if metode == "baseline":
        return hasil
    tanggal_ref = pd.to_datetime(pd.DataFrame({"year": referensi["arrival_year"], "month": referensi["arrival_month"], "day": referensi["arrival_day"]}), errors="coerce")
    tanggal_data = pd.to_datetime(pd.DataFrame({"year": hasil["arrival_year"], "month": hasil["arrival_month"], "day": hasil["arrival_day"]}), errors="coerce")
    median_global = referensi["days_before_arrival"].median()
    assert pd.notna(median_global)
    for indeks in hasil.index[hasil["days_before_arrival"].isna()]:
        kanal = hasil.loc[indeks, "booking_channel"]
        tanggal = tanggal_data.loc[indeks]
        kanal_sama = referensi["booking_channel"].eq(kanal)
        dekat = (tanggal_ref - tanggal).abs().dt.days <= hari
        nilai = referensi.loc[kanal_sama & dekat, "days_before_arrival"].dropna()
        if len(nilai) >= 20:
            nilai_isi = nilai.mean() if metode == "mean" else nilai.median()
        else:
            nilai_kanal = referensi.loc[kanal_sama, "days_before_arrival"].dropna()
            nilai_isi = nilai_kanal.median() if len(nilai_kanal) >= 20 else median_global
        hasil.loc[indeks, "days_before_arrival"] = nilai_isi
    return hasil

