# Laporan Proyek Machine Learning - Istia Budi

## Domain Proyek

Konsumsi energi merupakan salah satu isu yang semakin menjadi perhatian di berbagai sektor, khususnya dalam pengelolaan bangunan komersial, perumahan, dan industri. Dalam era modern, kebutuhan akan energi terus meningkat seiring dengan perkembangan teknologi, pertumbuhan populasi, dan urbanisasi yang pesat. Bangunan, sebagai salah satu penyumbang utama konsumsi energi global, memanfaatkan energi untuk berbagai keperluan seperti pencahayaan, pengaturan suhu, hingga perangkat elektronik. Namun, pemanfaatan energi yang tidak efisien dapat menyebabkan peningkatan biaya operasional yang signifikan, pemborosan sumber daya, serta dampak negatif terhadap lingkungan, seperti emisi karbon yang berkontribusi pada perubahan iklim. Oleh karena itu, pemantauan dan pengelolaan konsumsi energi menjadi hal yang sangat penting untuk mendukung efisiensi energi, mengurangi biaya operasional, dan mendorong keberlanjutan lingkungan. Dengan memahami pola konsumsi energi melalui analisis data yang cermat, pihak pengelola dapat mengidentifikasi area yang membutuhkan perbaikan, merancang strategi penghematan energi, serta mengoptimalkan penggunaan energi tanpa mengorbankan kenyamanan pengguna bangunan. Dalam konteks ini, teknologi modern seperti machine learning menawarkan solusi cerdas untuk menganalisis pola konsumsi energi yang dapat membantu mencapai efisiensi energi secara optimal, sehingga menciptakan bangunan yang lebih hemat energi, berkelanjutan, dan ramah lingkungan.

## Business Understanding

Sistem manajemen energi sering digunakan untuk memprediksi kebutuhan energi berdasarkan faktor-faktor lingkungan, aktivitas, data bangunan. Informasi ini sangat penting untuk mengurangi konsumsi energi dan mengoptimalkan penggunaan sumber daya.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Dari serangkaian faktor yang ada, faktor apa yang paling mempengaruhi konsumsi energi.
- Bagaimana prediksi konsumsi energi dengan aktivitas, data bagunan dan faktor lingkungan tertentu.

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mengetahui faktor yang berkolerasi dengan konsumsi energi
- membuat model machine learning yang dapat memprediksi konsumsi energi seakurat mungkin berdasarkan data yang ada.

### Solution statements

- Melakukan Exploratory Data Analysis - EDA
- Membuat model machine learning dan memilih yang terbaik diantara ketiga algoritma yaitu:
    - K-Nearest Neighbor
    - Random Forest
    - Boosting
- Metrik yang digunakan adalah Mean Squared Error (MSE)

## Data Understanding

Dataset yang digunakan dalam proyek ini berasal dari situs Kaggle, dengan total 5000 baris data dan 12 kolom fitur yang mencakup informasi terkait konsumsi energi. Dataset ini telah melalui proses pembersihan, sehingga tidak terdapat missing value, duplikat data, ataupun outlier. Fitur-fitur dalam dataset terbagi menjadi dua jenis: fitur kategorikal (non-numerik) dan numerik, yang akan digunakan untuk menemukan pola dalam data. Kolom EnergyConsumption berfungsi sebagai fitur target untuk prediksi konsumsi energi.
Fitur-fitur pada Dataset
Berikut adalah deskripsi seluruh fitur yang terdapat dalam dataset:

### Variabel-variabel pada Energy Consumption Prediction Dataset adalah sebagai berikut:

Fitur Kategorikal (Non-Numerik)
- DayOfWeek: Hari dalam seminggu (Senin hingga Minggu).
- Holiday: Indikator apakah hari tersebut merupakan hari libur (Yes/No).
- HVACUsage: Status penggunaan sistem pendingin atau pemanas ruangan (On/Off).
- LightingUsage: Status penggunaan lampu (On/Off).
Fitur Numerik
- Month: Bulan (rentang 1–12).
- Hour: Jam dalam format 24 jam (rentang 0–23).
- Temperature: Suhu lingkungan dalam derajat Celcius.
- Humidity: Tingkat kelembapan udara dalam persen (%).
- SquareFootage: Luas area bangunan atau ruangan dalam meter persegi (m²).
- Occupancy: Jumlah individu atau orang yang berada di lokasi pada waktu tertentu.
- RenewableEnergy: Jumlah energi yang dihasilkan dari sumber energi terbarukan (dalam kWh).
- EnergyConsumption: Total konsumsi energi (dalam kWh), yang merupakan target prediksi.

Dataset dapat diunduh melalui tautan berikut: [energy-consumption-prediction.csv](https://www.kaggle.com/datasets/ajinilpatel/energy-consumption-prediction/data).

### Kondisi Data
- Jumlah Data: Dataset terdiri dari 5000 baris dan 12 kolom.
- Kondisi: Data telah bersih, tanpa missing value, duplikasi, atau outlier, sehingga siap digunakan untuk analisis lebih lanjut.
Dengan karakteristik data ini, seluruh fitur dapat dioptimalkan untuk mengidentifikasi pola konsumsi energi serta meningkatkan akurasi prediksi menggunakan model machine learning.

### EDA - Univariate

![DayOfWeek Univariate](https://github.com/user-attachments/assets/eb5f3e91-a051-4c71-b74b-01f92c210d56)

Terdapat 7 kategori yang mempresentasikan hari pada fitur DayOfWeek, dari data tersebut bisa disimpulkan bahwa datanya hampir merata dengan yang tertinggi pada hari sabtu yaitu 14.8% dan hari senin yang paling kecil datanya dengan presentase 13.8 persen.

![Holiday Univariate](https://github.com/user-attachments/assets/70bbb964-4ec1-4ecc-912f-d7da0af15015)

Fitur holiday atau liburan memiliki sebaran data yang hampir seimbang, No memiliki presentase data sebesar 53.1% dan Yes memiliki presentase data sebesar 46.9%.

![HVACUsage Univariate](https://github.com/user-attachments/assets/6bb5b301-7d56-483e-93e2-f6872c096f50)

Pada grafik diatas HVACUsage atau status sistem pendingin datanya seimbang memiliki masing masing 50% data.

![LightningUsage Univariate](https://github.com/user-attachments/assets/23493dcf-b2f2-4b10-afad-5e8f34cb8186)

Fitur LightningUsage yaitu penggunaan lampu nyala atau tidak juga memiliki sebaran data yang hampir seimbang, Off memiliki presentase data sebesar 50.9% dan On memiliki presentase data sebesar 49.1%.

![hist univariate](https://github.com/user-attachments/assets/4381f79f-418e-4ee5-8876-859b3c64e92d)

Dilihat dari histogram variabel 'EnergyConsumption', yang merupakan target fitur (label) pada data. Bisa diperoleh beberapa informasi yaitu:
- Rentang penggunaan energi yaitu 53-100, menunjukkan konsumsi energi dalam batas wajar
- Distribusi datanya adalah simetris, memiliki distribusi yang seimbang disekitar pusatnya.

### EDA - Multivariate

![Day Multivariate](https://github.com/user-attachments/assets/7f292486-a5d9-40fe-b8bc-f2dff4c2160f)

![holiday multi](https://github.com/user-attachments/assets/53fb4c1c-730b-466d-9ebd-c005b2a572ed)

![hvac multi](https://github.com/user-attachments/assets/74df2b43-18ba-4863-8e63-6b76733e82c2)

![lightning multi](https://github.com/user-attachments/assets/a93873f0-be93-4e20-bf92-d92276490b2b)

Berdasarkan rata rata konsumsi energi terhadap fitur kategori memberikan beberapa informasi, berikut analisisnya:
- Rentang rata-rata yang sempit mengindikasikan bahwa konsumsi energi relatif stabil meskipun ada perubahan kategori pada variabel 'DayOfWeek', 'Holiday', 'HVACUsage', atau 'LightningUsage'.
- Faktor-faktor tersebut tidak memiliki pengaruh yang signifikan terhadap rata-rata konsumsi energi.

![hist multi](https://github.com/user-attachments/assets/25bdfdff-6a1a-4c71-a2ee-748c4b3e8e91)

Berdasarkan scatter plot, hanya Temperature saja yang terlihat berhubungan dengan EnergyConsumption secara visual. Hubungan lain seperti Humidity, SquareFootage, Hour, RenewableEnergy terlihat lemah.

![conf matrix](https://github.com/user-attachments/assets/984142e4-bd86-40a9-83e0-b59bf3fa17e3)

Jika diamati, fitur 'Temperature' satu satunya yang berkolerasi dengan 'EnergyConsumption' (bernilai 0.54). Karena sebagian fitur memiliki korelasi rendah dengan target model yang berbasis linear regression mungkin akan tidak optimal.


## Data Preparation

Tahap menyiapkan data yaitu:
- Encoding fitur kategori
- Split dataset dengan fungsi train_test_split
- Standarisasi

**Encoding**

Untuk melakukan proses encoding fitur kategori dalam studi kasus ini menggunakan LabelEncoder dari library scikt-learn. Label Encoder menghasilkan representasi numerik sederhana dari data kategori (misalnya, "Monday" menjadi 0, "Tuesday" menjadi 1).
```python
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for feature in categorical_feature:
    label_encoder = LabelEncoder()
    energy_df[feature] = label_encoder.fit_transform(energy_df[feature])
    label_encoders[feature] = label_encoder

energy_df.head(5)
```

**Split Dataset**

Setelah melakukan encoding data selanjutnya adalah membagi data menjadi data train dan data test dengan perbandingan 80% untuk train dan 20% untuk test menggunakan train_test_split dari library scikit-learn.
```python
from sklearn.model_selection import train_test_split

X = energy_df.drop('EnergyConsumption', axis=1)
y = energy_df['EnergyConsumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Standarisasi**

Standarisasi dengan menggunakan StandardScaler pada data dan mengubah nilai rata-rata(mean) menjadi 0 dan nilai standar deviasi menjadi 1.
```python
from sklearn.preprocessing import StandardScaler

numerical_features = ['Month', 'Hour', 'Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'RenewableEnergy']

scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train[numerical_features])
X_train[numerical_features].head()
```

## Modeling

Model yang digunakan adalah:
- K-Nearest Neighbor
Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). 
```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

models.loc['train_mse', 'KNN'] = mean_squared_error(y_pred = knn.predict(X_train), y_true = y_train)
```
Pada model K-Nearest Neighbors (KNN), digunakan nilai k=5 sebagai jumlah tetangga terdekat, dengan metrik Euclidean yang digunakan untuk mengukur jarak antar titik.

- Random Forest
Algoritma kedua adalah Random Forest, Random Forest (RF) adalah algoritma yang dapat meningkatkan hasil akurasi dalam membangkitkan atribut untuk setiap node yang dilakukan secara acak. Random forest merupakan salah satu model machine learning yang terdiri dari beberapa model dan bekerja secara bersama-sama.
```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=50, max_depth=16, n_jobs=-1, random_state=55)
rf.fit(X_train, y_train)

models.loc['train_mse', 'Random Forest'] = mean_squared_error(y_pred = rf.predict(X_train), y_true = y_train)
```
Di sini menggunakan parameter n_estimator sebanyak 50 yang mana parameter tersebut akan membuat sebanyak 50 cabang pohon, dengan kedalaman maksimal 16.

- Boosting (ADABoost)
Seperti namanya, boosting, algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner).
```python
from sklearn.ensemble import AdaBoostRegressor

adaboost = AdaBoostRegressor(learning_rate=0.05, random_state=55)
adaboost.fit(X_train, y_train)

models.loc['train_mse', 'ADABoost'] = mean_squared_error(y_pred = adaboost.predict(X_train), y_true = y_train)
```
Parameter learning rate yang dipilih adalah 0.05 dan random state 55

## Evaluation

Metode evaluasi yang digunakan dalam proyek ini adalah Mean Squared Error (MSE), yang menghitung rata-rata selisih kuadrat antara nilai aktual dan nilai prediksi. MSE digunakan karena mampu memberikan gambaran kuantitatif tentang seberapa baik model memprediksi target variabel. Rumus MSE adalah:

![Rumus MSE](https://github.com/user-attachments/assets/2109c780-e604-4cd1-9101-f1c6fa85cb90)

Semakin kecil nilai MSE, semakin baik performa model dalam membuat prediksi.

**Hasil Evaluasi**

Berdasarkan hasil evaluasi menggunakan MSE dapat disimpulkan bahwa model Boosting memiliki MSE yang kecil dibandingkan dengan model lainnya

Tabel perbandingan MSE
| Model | Train | Test |
| ---- | ---- | ---- |
| KNN | 0.047223 | 0.206707 |
| Random Forest | 0.010006 | 0.156066 |
| ADABoost | 0.057788 | 0.13968 |

- Boosting (ADABoost) memiliki performa terbaik dengan nilai RMSE terkecil pada data uji (0.139680), menunjukkan bahwa model ini memberikan prediksi yang paling akurat dibandingkan model lainnya.
- Random Forest menunjukkan performa yang cukup baik pada data uji, namun sedikit lebih buruk dibandingkan Boosting (ADABoost).
- K-Nearest Neighbor (KNN) memiliki performa yang kurang memuaskan karena RMSE pada data uji jauh lebih tinggi dibandingkan model lainnya, yang menunjukkan kurang optimalnya model ini dalam menangkap pola data.

**Visualisasi Perbandingan**

![Grafik Perbandingan](https://github.com/user-attachments/assets/c9b3ce38-15f0-4c74-a42f-4f6bfa50f9bc)

Grafik berikut menunjukkan perbandingan nilai MSE untuk setiap model.

**Prediksi Model**

Berikut adalah tabel perbandingan nilai aktual (y_true) dan prediksi dari masing-masing model 

| y_true | prediksi_KNN | prediksi_Random Forest | prediksi_ADABoost |
| ---- | ---- | ---- | ---- |
| 84.778571	 | 64.3 | 67.4 | 69.1 |

Berdasarkan tabel prediksi, model Boosting memberikan hasil prediksi yang lebih mendekati nilai aktual dibandingkan model KNN dan Random Forest, mendukung hasil evaluasi MSE.

**Dampak terhadap Business Understanding**

1. Apakah model sudah menjawab problem statement?
   - Berdasarkan hasil analisis temperature, HVACUsage, dan SquareFootage menunjukkan pengaruh yang signifikan terhadap konsumsi energi.
   - Model yang dikembangkan, terutama Boosting, memberikan prediksi konsumsi energi yang akurat dengan nilai RMSE terkecil pada data uji (0.139680). Prediksi ini membantu pemangku kepentingan untuk merencanakan kebutuhan energi di berbagai kondisi, seperti jadwal aktivitas, cuaca tertentu, atau penggunaan fasilitas bangunan.
2. Apakah goals tercapai?
   - Goals pertama, yaitu mengetahui faktor yang berkolerasi dengan konsumsi energi, telah tercapai melalui analisis data dan eksplorasi fitur, temperature adalah faktor yang paling berkolerasi. Hal ini memudahkan pengambilan keputusan berbasis data untuk memprioritaskan pengelolaan faktor-faktor tertentu yang berpengaruh besar.
   - Goals kedua, yaitu membuat model machine learning untuk prediksi konsumsi energi yang akurat, juga berhasil dicapai. ADABoost muncul sebagai model terbaik dengan performa yang konsisten, mendekati nilai aktual konsumsi energi.
3. Apakah solusi yang direncanakan berdampak?
   - EDA: Melalui Exploratory Data Analysis, hubungan antarvariabel dapat dipahami dengan lebih baik. Faktor-faktor penting telah diidentifikasi, yang menjadi landasan untuk pelatihan model yang lebih terarah.
   - Model prediksi: Model ADABoost memberikan prediksi yang akurat dan andal. Dengan prediksi ini, pemangku kepentingan dapat mengantisipasi kebutuhan energi berdasarkan aktivitas dan kondisi tertentu, sehingga meminimalkan pemborosan energi.
   - Metrik MSE: Penggunaan MSE sebagai metrik evaluasi memberikan ukuran kuantitatif yang jelas untuk membandingkan performa model dan memastikan solusi yang diberikan efektif.
  
**Kesimpulan**

Model ADABoost telah menjawab problem statement dengan baik dan berhasil mencapai goals yang diharapkan. Solusi ini memberikan dampak positif bagi pengelolaan energi di bangunan, mendukung keputusan yang berbasis data, dan meningkatkan efisiensi operasional secara signifikan.
