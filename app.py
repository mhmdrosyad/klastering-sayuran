from flask import Flask, render_template, \
    request, redirect, url_for, session, flash, get_flashed_messages, jsonify
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import io
import base64
import matplotlib.pyplot as plt
from io import BytesIO

df = pd.DataFrame()
scaler = StandardScaler()

application = Flask(__name__)

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/upload', methods=['POST'])
def upload():
    global df, scaler  # Gunakan df dan scaler yang telah diinisialisasi di luar fungsi

    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Baca file CSV yang diunggah
        uploaded_data = pd.read_csv(io.StringIO(file.read().decode('utf-8')), index_col="provinsi")
        
        # Fitting scaler jika belum difit sebelumnya
        if not hasattr(scaler, 'mean_'):
            scaler.fit(uploaded_data)

        # Standarisasi data yang diunggah menggunakan scaler yang telah difit
        uploaded_data_scaled = scaler.transform(uploaded_data)

        # Lakukan klasterisasi KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(uploaded_data_scaled)

        # Tambahkan kolom baru ke dataframe asli dengan label klaster
        df = uploaded_data.copy()
        df['klaster'] = kmeans.labels_

        # Kirim data klaster ke halaman HTML menggunakan JSON
        cluster_data = df.to_dict(orient='index')

        provinsi = list(cluster_data.keys())

        jenis_tanaman = ['bawang.daun', 'bawang.merah', 'bawang.putih', 'bayam', 'buncis', 'cabai.rawit', 'kacang.panjang']
        klaster = [cluster_data[prov]['klaster'] for prov in provinsi]

        plt.figure(figsize=(10, 8))

        # Menggunakan subplot agar sumbu x lebih panjang
        plt.subplot(1, 1, 1)

        for tanaman in jenis_tanaman:
            values = [cluster_data[prov][tanaman] for prov in provinsi]
            plt.plot(provinsi, values, label=tanaman)

        plt.xlabel('Provinsi (Klaster {})'.format(klaster[0]))  # Menampilkan klaster pada sumbu x
        plt.ylabel('Jumlah Tanaman')
        plt.title('Grafik Jumlah Produksi per Provinsi')
        plt.legend()
        plt.xticks(rotation=45, ha="right")  # Mengatur rotasi dan penempatan label sumbu x
        plt.tight_layout()

        # Mengonversi grafik ke format gambar
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')

        return render_template('result.html', cluster_data=cluster_data, encoded_image=encoded_image)

if __name__ == '__main__':
    application.run(debug=True)