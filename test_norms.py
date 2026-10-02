import librosa
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = tf.keras.models.load_model('cnn_bilstm_5k_model.h5', compile=False)

def test_norm(name, file, norm_fn):
    audio, sr = librosa.load(file, sr=16000)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < 130:
        mel_db = np.pad(mel_db, ((0,0),(0,130-mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :130]
    
    input_data = norm_fn(mel_db)
    input_data = input_data[np.newaxis, ..., np.newaxis]
    prob = float(model.predict(input_data, verbose=0)[0][0])
    return prob

files = {'Real': 'short.wav', 'Fake': 'test_fake.wav'}
norms = {
    'None': lambda x: x,
    '/ 255': lambda x: x / 255.0,
    'Min-Max': lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8),
    'Standard': lambda x: (x + 80) / 80.0,
    'No ref=max': lambda x: librosa.power_to_db(librosa.feature.melspectrogram(y=librosa.load(x, sr=16000)[0], sr=16000, n_mels=128)[:, :130]),
}

# Special case for 'No ref=max' as it needs to reload
def test_no_ref(file):
    audio, sr = librosa.load(file, sr=16000)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel) # Default ref=1.0
    if mel_db.shape[1] < 130:
        mel_db = np.pad(mel_db, ((0,0),(0,130-mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :130]
    input_data = mel_db[np.newaxis, ..., np.newaxis]
    return float(model.predict(input_data, verbose=0)[0][0])

print(f"{'Normalization':<15} | {'Real':<10} | {'Fake':<10}")
print("-" * 40)

for norm_name, fn in norms.items():
    if norm_name == 'No ref=max':
        p_real = test_no_ref('short.wav')
        p_fake = test_no_ref('test_fake.wav')
    else:
        p_real = test_norm('Real', 'short.wav', fn)
        p_fake = test_norm('Fake', 'test_fake.wav', fn)
    print(f"{norm_name:<15} | {p_real:<10.6f} | {p_fake:<10.6f}")
