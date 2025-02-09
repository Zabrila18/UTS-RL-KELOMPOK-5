# UTS-RL-KELOMPOK-5
1. Kahfi Zairan Maulana G1A021041
2. Esti Asmareta Ayu G1A021042
3. Zabrila Amrina Zadia Putri G1A021053

[](https://github.com/user-attachments/assets/020688de-1a0c-4706-aaa0-ba0cb0927ccd)

## Pengantar

Proyek ini mengimplementasikan agen **Q-learning** sederhana untuk memainkan game pemecah bata (brick-breaking) menggunakan PyTorch dan Pygame. Agen tersebut belajar mengendalikan paddle dengan tujuan menghancurkan bata dengan memantulkan bola. **Q-learning** adalah algoritma *reinforcement learning* yang memungkinkan agen untuk belajar dari interaksi dengan lingkungannya agar dapat memaksimalkan total reward (hadiah) dari waktu ke waktu.

### Konsep Utama:

1. **Algoritma Q-Learning**:  
   Q-learning adalah algoritma *model-free*, *off-policy* dalam *reinforcement learning* yang digunakan untuk menemukan tindakan terbaik yang harus diambil dalam kondisi tertentu. Tujuan dari algoritma ini adalah memaksimalkan total reward dengan cara mempelajari nilai Q (Q-values), yang mewakili estimasi reward masa depan ketika melakukan suatu tindakan pada kondisi tertentu.

2. **Q-Network (Deep Q-Learning)**:  
   Jaringan saraf, yang disebut **Q-Network**, digunakan untuk mendekati nilai Q. Jaringan ini memetakan kondisi permainan (posisi bola dan paddle, arah bola, dan keberadaan bata) ke nilai Q untuk setiap aksi yang mungkin (bergerak ke kiri atau kanan). Arsitektur jaringan terdiri dari beberapa *fully connected layers* dengan fungsi aktivasi ReLU.

3. **Representasi Kondisi**:  
   Kondisi permainan direpresentasikan sebagai vektor berdimensi 8 yang berisi informasi sebagai berikut:
   - Posisi bola relatif terhadap paddle.
   - Arah bola (timur laut, barat laut, tenggara, atau barat daya).
   - Keberadaan bata di sisi kiri atau kanan paddle.

4. **Pemilihan Aksi**:  
   Agen dapat mengambil dua aksi: menggerakkan paddle ke kiri atau ke kanan. Selama pelatihan, agen menggunakan kebijakan **epsilon-greedy**, di mana ia menjelajahi lingkungan dengan mengambil tindakan acak dengan probabilitas tertentu dan mengeksploitasi pengetahuan yang telah dipelajarinya dengan memilih aksi dengan nilai Q tertinggi jika tidak.

5. **Mekanisme Reward**:  
   Agen mendapatkan reward berdasarkan interaksinya dengan lingkungan:
   - +15 jika berhasil memantulkan bola menggunakan paddle.
   - +2 untuk setiap bata yang dihancurkan.
   - Reward negatif berdasarkan jarak antara paddle dan bola jika bola terlepas dari paddle.

### Ketergantungan

Proyek ini memerlukan beberapa pustaka berikut:
- **PyTorch**: Untuk membangun dan melatih Q-network.
- **Pygame**: Untuk menampilkan lingkungan permainan.
- **Matplotlib**: Untuk memvisualisasikan perkembangan pembelajaran agen.

## Cara Memanggil Codingan
1. Pip Install Requirements.txt
2. Python q_learning_breakout.py
