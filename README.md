# Laporan Proyek Machine Learning - Istia Budi

## Domain Proyek

Konsumsi energi merupakan salah satu aspek yang kritis dalam pengelolaan bangunan. Pemantauan pola penggunaan energi dapat membantu meningkatkan efesiensi dan mengurangi biaya oprasional.

## Business Understanding

Sistem manajemen energi sering digunakan untuk memprediksi kebutuhan energi berdasarkan faktor-faktor lingkungan, aktivitas, data bangunan. Informasi ini sangat penting untuk mengurangi konsumsi energi dan mengoptimalkan penggunaan sumber daya.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Dari serangkaian faktor yang ada, faktor apa yang paling mempengaruhi konsumsi energi.
- Berapa konsumsi energi dengan aktivitas, data bagunan dan faktor lingkungan tertentu.

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

Dataset yang digunakan berasal dari situs [Kaggle](https://www.kaggle.com/). Dataset ini memiliki 5000 data terkait dengan konsumsi energi dengan berbagai faktor. Faktor yang dimaksud adalah fitur non numerik seperti DayOfWeek, Holiday, HVACUsage, dan LightningUsage serta fitur numerik seperti Month, Hour, Temperature, Humidity, SquareFootage, Occupacy, dan RenewableEnergy. Untuk menemukan pola pada data kesebelas fitur ini akan digunakan, sedangkan DataConsumption merupakan fitur target. 

Dataset dapat diunduh melalui: [energy-consumption-prediction.csv](https://www.kaggle.com/datasets/ajinilpatel/energy-consumption-prediction/data).

### Variabel-variabel pada Energy Consumption Prediction Dataset adalah sebagai berikut:
Berdasarkan data tersebut variabel-variabel pada Energy Consumption Prediction Dataset adalah sebagai berikut:

- Month: Bulan (1-12)
- Hour: Jam (0-23)
- DayOfWeek: Hari dalam seminggu (Senin-Minggu)
- Holiday: Boolean liburan apa tidak (Yes-No)
- Temperature: Suhu lingkungan menggunakan celcius
- Humidity: Tingkat kelembapan lingkungan (%) dalam waktu tertentu
- SquareFootage: Luas bangunan atau lapangan (dalam satuan meter persegi)
- Occupancy: Jumlah orang dilokasi
- HVACUsage: Status sistem pendingin atau pemanas (On/Off).
- LightingUsage: Status penggunaan lampu (On/Off).
- RenewableEnergy: Energi yang dihasilkan dari sumber terbarukan (kWh).
- EnergyConsumption: Total konsumsi energi (kWh).

### EDA - Univariate

![DayOfWeek Univariate](https://github.com/user-attachments/assets/44adae7c-3871-4184-a094-2214e7a6c0e0)

Terdapat 7 kategori yang mempresentasikan hari pada fitur DayOfWeek, dari data tersebut bisa disimpulkan bahwa datanya hampir merata dengan yang tertinggi pada hari sabtu yaitu 14.8% dan hari senin yang paling kecil datanya dengan presentase 13.8 persen.

![Holiday Univariate](https://github.com/user-attachments/assets/158d3098-18aa-4a3b-a67d-c141aad322ee)

Fitur holiday atau liburan memiliki sebaran data yang hampir seimbang, No memiliki presentase data sebesar 53.1% dan Yes memiliki presentase data sebesar 46.9%.

![HVACUsage Univariate](https://github.com/user-attachments/assets/983677e7-4958-4105-a073-a52a45d96f9e)

Pada grafik diatas HVACUsage atau status sistem pendingin datanya seimbang memiliki masing masing 50% data.

![LightningUsage Univariate](https://github.com/user-attachments/assets/bdcaadd3-024b-455c-9636-f971e6007c2d)

Fitur LightningUsage yaitu penggunaan lampu nyala atau tidak juga memiliki sebaran data yang hampir seimbang, Off memiliki presentase data sebesar 50.9% dan On memiliki presentase data sebesar 49.1%.

![hist univariate](https://github.com/user-attachments/assets/ef845a3d-ba40-4778-8188-32c67302f405)

Dilihat dari histogram variabel 'EnergyConsumption', yang merupakan target fitur (label) pada data. Bisa diperoleh beberapa informasi yaitu:
- Rentang penggunaan energi yaitu 53-100, menunjukkan konsumsi energi dalam batas wajar
- Distribusi datanya adalah simetris, memiliki distribusi yang seimbang disekitar pusatnya.

### EDA - Multivariate

![Day Multivariate](https://github.com/user-attachments/assets/44453235-a561-4286-a03e-078afe55dd44)
![holiday multi](https://github.com/user-attachments/assets/83d73d46-8932-4294-a9e1-342b0547ff83)
![hvac multi](https://github.com/user-attachments/assets/b259f827-53f2-4bee-87fb-9cb557079d57)
![lightning multi](https://github.com/user-attachments/assets/72d01ca0-d1f5-40dc-b51b-75f8f2be4504)

Berdasarkan rata rata konsumsi energi terhadap fitur kategori memberikan beberapa informasi, berikut analisisnya:
- Rentang rata-rata yang sempit mengindikasikan bahwa konsumsi energi relatif stabil meskipun ada perubahan kategori pada variabel 'DayOfWeek', 'Holiday', 'HVACUsage', atau 'LightningUsage'.
- Faktor-faktor tersebut tidak memiliki pengaruh yang signifikan terhadap rata-rata konsumsi energi.

![hist multi](https://github.com/user-attachments/assets/91cc54f6-702f-412e-a0ae-11298a25b5c5)

Berdasarkan scatter plot, hanya Temperature saja yang terlihat berhubungan dengan EnergyConsumption secara visual. Hubungan lain seperti Humidity, SquareFootage, Hour, RenewableEnergy terlihat lemah.

![conf matrix](https://github.com/user-attachments/assets/96c82e17-4556-4e24-9450-292aa1054305)

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
Kita menggunakan k = 5 neighbor dan metric Euclidean untuk mengukur jarak antara titik.

- Random Forest
Algoritma kedua adalah Random Forest, Random Forest (RF) adalah algoritma yang dapat meningkatkan hasil akurasi dalam membangkitkan atribut untuk setiap node yang dilakukan secara acak. Random forest merupakan salah satu model machine learning yang terdiri dari beberapa model dan bekerja secara bersama-sama.
```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=50, max_depth=16, n_jobs=-1, random_state=55)
rf.fit(X_train, y_train)

models.loc['train_mse', 'Random Forest'] = mean_squared_error(y_pred = rf.predict(X_train), y_true = y_train)
```
Di sini menggunakan parameter n_estimator sebanyak 50 yang mana parameter tersebut akan membuat sebanyak 50 cabang pohon, dengan kedalaman maksimal 16.

- Boosting
Seperti namanya, boosting, algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner).
```python
from sklearn.ensemble import AdaBoostRegressor

adaboost = AdaBoostRegressor(learning_rate=0.05, random_state=55)
adaboost.fit(X_train, y_train)

models.loc['train_mse', 'ADABoost'] = mean_squared_error(y_pred = adaboost.predict(X_train), y_true = y_train)
```
Parameter learning rate yang dipilih adalah 0.05 dan random state 55

| Model | Train | Test |
| ---- | ---- | ---- |
| KNN | 0.047223 | 0.206707 |
| Random Forest | 0.010006 | 0.156066 |
| ADABoost | 0.057788 | 0.13968 |

MSE Boosting lebih rendah daripada KNN dan RF, maka akan menggunakan model Boosting


## Evaluation

Matriks Evaluasi yang digunakan adalah MSE (Mean Squared Error) yang menghitung selisih kuadrat rata rata dari nilai sebenarnya dengan nilai prediksi. Rumus MSE adalah:
![Rumus MSE](https://user-images.githubusercontent.com/111114060/192172988-a8427c11-74c6-4911-9fd1-4c2f6956bb6c.png)

Berdasarkan hasil evaluasi menggunakan MSE dapat disimpulkan bahwa model Boosting memiliki MSE yang kecil dibandingkan dengan model lainnya

Tabel Train Test RMSE
| Model | Train | Test |
| ---- | ---- | ---- |
| KNN | 0.047223 | 0.206707 |
| Random Forest | 0.010006 | 0.156066 |
| ADABoost | 0.057788 | 0.13968 |

![Grafik MSE](https://github.com/user-attachments/assets/1e7322fc-e2d2-4b41-80f8-9e73bed46df4)


Tabel Prediksi
| y_true | prediksi_KNN | prediksi_Random Forest | prediksi_ADABoost |
| ---- | ---- | ---- | ---- |
| 84.778571	 | 64.3 | 67.4 | 69.1 |

Dapat dilihat pada tabel bahwa prediksi menggunakan Boost lebih mendekati nilai y.