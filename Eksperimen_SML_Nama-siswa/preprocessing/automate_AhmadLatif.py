import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(path):
    # Membaca dataset dari file CSV ke dalam DataFrame
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Membersihkan data dari nilai kosong (missing values) dan data duplikat
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def encode_data(df):
    # Mengubah data kategorikal menjadi numerik menggunakan one-hot encoding
    df = pd.get_dummies(df, drop_first=True)
    return df

def normalize_data(df):
    # Menormalisasi data numerik ke rentang 0-1 menggunakan MinMaxScaler
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

def save_data(df, path):
    # Menyimpan DataFrame hasil preprocessing ke file CSV
    df.to_csv(path, index=False)

def main():
    # Menjalankan seluruh pipeline preprocessing dari load hingga save
    input_path = '../dataset_raw/student_performance.csv'
    output_path = '../preprocessing/dataset_preprocessing.csv'
    
    df = load_data(input_path)
    df = clean_data(df)
    df = encode_data(df)
    df = normalize_data(df)
    save_data(df, output_path)

if __name__ == "__main__":
    # Menjalankan fungsi utama ketika script dieksekusi
    main()
