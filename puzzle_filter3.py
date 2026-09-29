import random
import time
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe el ve çizim modülleri
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

state = "NORMAL"  # NORMAL, COUNTDOWN, PUZZLE, COMPLETED
captured_frame = None
grid_size = 3
board = []
empty_pos = (2, 2)
piece_h, piece_w = 0, 0
puzzle_size = 350  # Yapboz ekran boyutu

# Sürükle-bırak için değişkenler
dragging = False
drag_start_pos = None

# Geri sayım için değişkenler
countdown_start_time = 0
countdown_duration = 3  # 3 saniye

# Sabit sol alan (ROI) koordinatları (Sol tarafta yer alacak çerçeve)
box_w, box_h = 350, 350

# Yapboz animasyon/konum değişkenleri
puzzle_target_x, puzzle_target_y = 0, 0

# Kayıt kontrol bayrağı (Sadece bir kez kaydetmesi için)
saved_completion_photo = False


def init_puzzle():
  global board, empty_pos, piece_h, piece_w, puzzle_size, saved_completion_photo
  piece_h = puzzle_size // grid_size
  piece_w = puzzle_size // grid_size
  saved_completion_photo = False

  # 3x3 matris oluştur
  board = [[r * grid_size + c for c in range(grid_size)] for r in range(grid_size)]
  empty_pos = (grid_size - 1, grid_size - 1)
  board[empty_pos[0]][empty_pos[1]] = -1  # -1 boş parça

  # Çözülebilir olması için rastgele karıştır (Asla tamamen çözülmüş başlamasın)
  while True:
    for _ in range(40):
      r, c = empty_pos
      neighbors = []
      if r > 0:
        neighbors.append((r - 1, c))
      if r < grid_size - 1:
        neighbors.append((r + 1, c))
      if c > 0:
        neighbors.append((r, c - 1))
      if c < grid_size - 1:
        neighbors.append((r, c + 1))

      nr, nc = random.choice(neighbors)
      board[r][c], board[nr][nc] = board[nr][nc], board[r][c]
      empty_pos = (nr, nc)

    # Eğer şans eseri çözülmüş haliyle başladıysa tekrar karıştır
    if not check_win():
      break


# Puzzle'ın çözülüp çözülmediğini kontrol eden fonksiyon
def check_win():
  global board, grid_size
  expected_val = 0
  for r in range(grid_size):
    for c in range(grid_size):
      if r == grid_size - 1 and c == grid_size - 1:
        if board[r][c] != -1:
          return False
      else:
        if board[r][c] != expected_val:
          return False
      expected_val += 1
  return True


# Fare olaylarını yakalayan fonksiyon (Sağ taraftaki puzzle alanı için)
def mouse_callback(event, x, y, flags, param):
  global board, empty_pos, state, dragging, drag_start_pos
  global puzzle_target_x, puzzle_target_y, puzzle_size, piece_w, piece_h, grid_size

  if state == "PUZZLE":
    # Tıklamanın sağdaki puzzle ekranı içinde olup olmadığını kontrol et
    if (
        puzzle_target_x <= x < puzzle_target_x + puzzle_size
        and puzzle_target_y <= y < puzzle_target_y + puzzle_size
    ):
      local_x = x - puzzle_target_x
      local_y = y - puzzle_target_y

      clicked_c = local_x // piece_w
      clicked_r = local_y // piece_h

      if not (0 <= clicked_r < grid_size and 0 <= clicked_c < grid_size):
        return

      # Tıklama ile parça taşıma
      if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        drag_start_pos = (clicked_r, clicked_c)

        empty_r, empty_c = empty_pos
        if (abs(clicked_r - empty_r) + abs(clicked_c - empty_c)) == 1:
          board[empty_r][empty_c], board[clicked_r][clicked_c] = (
              board[clicked_r][clicked_c],
              board[empty_r][empty_c],
          )
          empty_pos = (clicked_r, clicked_c)

      elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        drag_start_pos = None


cv2.namedWindow("Sol Kamera - Sag Yapboz")
cv2.setMouseCallback("Sol Kamera - Sag Yapboz", mouse_callback)

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break

  frame = cv2.flip(frame, 1)
  h, w, c = frame.shape

  # Sol taraftaki çerçevenin koordinatları
  bx1 = 50
  by1 = (h - box_h) // 2
  bx2 = bx1 + box_w
  by2 = by1 + box_h

  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = hands.process(rgb_frame)

  if state == "NORMAL":
    distance = 100
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]

        x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
        x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        distance = int(np.hypot(x2 - x1, y2 - y1))

    # Sol taraftaki hedef alanı çiz (Köşeli dikdörtgen çerçeve)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 3)
    cv2.putText(
        frame,
        "Sol alani hedefle",
        (bx1, by1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    # Parmaklar birleştiğinde (pinch) 3, 2, 1 sayımını başlat
    if distance < 45:
      countdown_start_time = time.time()
      state = "COUNTDOWN"

    cv2.putText(
        frame,
        "Parmaklarini birlestirerek 3-2-1 sayimini baslat!",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

  elif state == "COUNTDOWN":
    elapsed_time = time.time() - countdown_start_time
    remaining_time = int(countdown_duration - elapsed_time) + 1

    # Sol çerçeveyi sabit göster
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 3)

    if remaining_time > 0:
      # Ekran ortasında büyük 3, 2, 1 sayımı
      cv2.putText(
          frame,
          str(remaining_time),
          (w // 2 - 40, h // 2 + 20),
          cv2.FONT_HERSHEY_SIMPLEX,
          5,
          (0, 0, 255),
          8,
      )
      cv2.putText(
          frame,
          "KAYIT YAPILIYOR...",
          (w // 2 - 120, h // 2 - 60),
          cv2.FONT_HERSHEY_SIMPLEX,
          1,
          (255, 255, 0),
          2,
      )
    else:
      # Süre bittiğinde sol taraftaki kutuyu kırp ve sağ tarafa puzzle olarak aktar
      cropped = frame[by1:by2, bx1:bx2]
      if cropped.size > 0:
        captured_frame = cv2.resize(cropped, (puzzle_size, puzzle_size))
      else:
        captured_frame = cv2.resize(frame, (puzzle_size, puzzle_size))

      init_puzzle()
      state = "PUZZLE"

  elif state in ["PUZZLE", "COMPLETED"]:
    # Sağ tarafın koordinatlarını belirle (Ekranın sağ orta kısmı)
    puzzle_target_x = w - puzzle_size - 80
    puzzle_target_y = (h - puzzle_size) // 2

    # Sol tarafta orijinal kamerayı ve çerçeveyi göstermeye devam et
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (200, 200, 200), 2)

    # Sağ taraftaki puzzle tahtasını oluştur
    puzzle_canvas = np.zeros((puzzle_size, puzzle_size, 3), dtype=np.uint8)

    for r in range(grid_size):
      for c in range(grid_size):
        val = board[r][c]
        dest_y1, dest_y2 = r * piece_h, (r + 1) * piece_h
        dest_x1, dest_x2 = c * piece_w, (c + 1) * piece_w

        if val != -1 or state == "COMPLETED":
          # Eğer tamamlandısa boş parça da dahil tam resmi göster
          if state == "COMPLETED" and val == -1:
            orig_r, orig_c = grid_size - 1, grid_size - 1
          else:
            orig_r = val // grid_size
            orig_c = val % grid_size

          src_y1, src_y2 = orig_r * piece_h, (orig_r + 1) * piece_h
          src_x1, src_x2 = orig_c * piece_w, (orig_c + 1) * piece_w

          puzzle_canvas[dest_y1:dest_y2, dest_x1:dest_x2] = captured_frame[
              src_y1:src_y2, src_x1:src_x2
          ]
        else:
          puzzle_canvas[dest_y1:dest_y2, dest_x1:dest_x2] = (30, 30, 30)

        # Tamamlandığında ızgara çizgilerini gizle ki tam fotoğraf görünsün
        if state != "COMPLETED":
          cv2.rectangle(
              puzzle_canvas,
              (dest_x1, dest_y1),
              (dest_x2, dest_y2),
              (200, 200, 200),
              1,
          )

    # Puzzle'ı ana ekranın sağ tarafına yerleştir
    frame[
        puzzle_target_y : puzzle_target_y + puzzle_size,
        puzzle_target_x : puzzle_target_x + puzzle_size,
    ] = puzzle_canvas

    # Duruma göre çerçeve rengi ve yazı değiştir
    if state == "PUZZLE":
      # Çözülüp çözülmediğini kontrol et
      if check_win():
        state = "COMPLETED"

      cv2.rectangle(
          frame,
          (puzzle_target_x - 2, puzzle_target_y - 2),
          (puzzle_target_x + puzzle_size + 2, puzzle_target_y + puzzle_size + 2),
          (0, 255, 0),
          2,
      )
      cv2.putText(
          frame,
          "Sag taraftaki puzzle'i fare ile coz | 'r': Basa Don",
          (15, 30),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 255, 255),
          2,
      )
    elif state == "COMPLETED":
      # TEBRİKLER! Otomatik fotoğraf kaydetme
      if not saved_completion_photo:
        cv2.imwrite("tamamlanan_yapboz.jpg", puzzle_canvas)
        print("Tebrikler! Yapboz bitti ve 'tamamlanan_yapboz.jpg' kaydedildi.")
        saved_completion_photo = True

      cv2.rectangle(
          frame,
          (puzzle_target_x - 2, puzzle_target_y - 2),
          (puzzle_target_x + puzzle_size + 2, puzzle_target_y + puzzle_size + 2),
          (0, 0, 255),
          3,
      )
      cv2.putText(
          frame,
          "TEBRIKLER! YAPBOZ TAMAMLANDI! ('r': Basa Don)",
          (15, 30),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          (0, 255, 0),
          2,
      )

  cv2.imshow("Sol Kamera - Sag Yapboz", frame)

  key = cv2.waitKey(1) & 0xFF
  if key == 27:  # ESC ile çıkış
    break
  elif key == ord("r"):  # Başa dön
    state = "NORMAL"

cap.release()
cv2.destroyAllWindows()