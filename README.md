PROJE KURULUM VE ÇALIŞTIRMA REHBERİ

1. GEREKSİNİMLER
Bilgisayarınızda Python yüklü olmalıdır. Ayrıca gerekli kütüphaneleri yüklemek için terminalde (komut satırında) şu komutu çalıştırın:
pip install opencv-python mediapipe numpy

2. DOSYA OLUŞTURMA
- Bilgisayarınızda boş bir Python dosyası açın (örneğin: puzzle_filter.py).
- Size verdiğimiz güncel Python kodunu bu dosyanın içine yapıştırıp kaydedin.

3. ÇALIŞTIRMA
- Terminal veya komut penceresini açın.
- Dosyanın bulunduğu klasöre gidin.
- Programı başlatmak için şu komutu yazın ve Enter'a basın:
  python puzzle_filter.py

4. KULLANIM
- Kamera açıldığında parmaklarınızı (başparmak ve işaret parmağı) birleştirerek geri sayımı başlatın.
- 3 saniye geri sayımın ardından ekrandan fotoğrafınız çekilecek ve sağ tarafta 3x3 yapboza dönüşecektir.
- Sağ taraftaki yapboz parçalarını fare ile tıklayarak hareket ettirebilir ve çözebilirsiniz.
- Yapboz tamamlandığında fotoğraf otomatik olarak bilgisayarınıza 'tamamlanan_yapboz.jpg' adıyla kaydedilir.
- 'r' tuşuna basarak başa dönebilir, 'ESC' tuşu ile programdan çıkabilirsiniz.
