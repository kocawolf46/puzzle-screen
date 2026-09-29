import time
import random
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

state = "NORMAL"  # NORMAL, COUNTDOWN, PUZZLE
captured_frame = None
grid_size = 3
board = []
empty_pos = (2, 2)
piece_h, piece_w = 0, 0
puzzle_size = 300  # Küçük ekran puzzle boyutu

# Sürükle-bırak için değişkenler
dragging = False
drag_start_pos = None

# Geri sayım için değişkenler
countdown_start_time = 0
countdown_duration = 3  # 3 saniye

# Seçilen boyut alanı (ROI) koordinatları
selected_box = None

# Kayma (Slide) animasyonu için değişkenler
puzzle_start_time = 0
current_puzzle_x, current_puzzle_y = 0, 0


def init_puzzle():
  global board, empty_pos, piece_h, piece_w, puzzle_size
  piece_h = puzzle_size // grid_size
  piece_w = puzzle_size // grid_size

  # 3x3 matris oluştur
  board = [[r * grid_size + c for c in range(grid_size)] for r in range(grid_size)]
  empty_pos = (grid_size - 1, grid_size - 1)
  board[empty_pos[0]][empty_pos[1]] = -1  # -1 boş parça

  # Çözülebilir olması için rastgele karıştır
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


# Fare olaylarını yakalayan fonksiyon (Küçük ekran koordinatlarına göre ayarlandı)
def mouse_callback(event, x, y, flags, param):
  global board, empty_pos, state, dragging, drag_start_pos
  global current_puzzle_x, current_puzzle_y, puzzle_size, piece_w, piece_h, grid_size

  if state == "PUZZLE":
    # Tıklamanın küçük puzzle ekranı içinde olup olmadığını kontrol et
    if (
        current_puzzle_x <= x < current_puzzle_x + puzzle_size
        and current_puzzle_y <= y < current_puzzle_y + puzzle_size
    ):
      local_x = x - current_puzzle_x
      local_y = y - current_puzzle_y

      clicked_c = local_x // piece_w
      clicked_r = local_y // piece_h

      if not (0 <= clicked_r < grid_size and 0 <= clicked_c < grid_size):
        return

      # 1. Tıklama (Tek tıklama ile komşu taşıma)
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

      # 2. Fareyi bırakma (Sürükleyip bırakma mantığı)
      elif event == cv2.EVENT_LBUTTONUP:
        if dragging and drag_start_pos:
          end_r, end_c = clicked_r, clicked_c
          start_r, start_c = drag_start_pos

          empty_r, empty_c = empty_pos
          if (
              (start_r, start_c) != (end_r, end_c)
              and (end_r, end_c) == (empty_r, empty_c)
              and (abs(start_r - empty_r) + abs(start_c - empty_c)) == 1
          ):
            board[empty_r][empty_c], board[start_r][start_c] = (
                board[start_r][start_c],
                board[empty_r][empty_c],
            )
            empty_pos = (start_r, start_c)

        dragging = False
        drag_start_pos = None


cv2.namedWindow("Python Puzzle Filter")
cv2.setMouseCallback("Python Puzzle Filter", mouse_callback)

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break

  frame = cv2.flip(frame, 1)
  h, w, c = frame.shape

  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = hands.process(rgb_frame)

  if state == "NORMAL":
    distance = 0
    box_coords = None
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]

        x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
        x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        distance = int(np.hypot(x2 - x1, y2 - y1))

        # Parmaklar arasındaki mesafeye göre dinamik alan çerçevesi
        box_size = max(150, distance * 3)
        bx1 = max(0, int((x1 + x2) / 2 - box_size // 2))
        by1 = max(0, int((y1 + y2) / 2 - box_size // 2))
        bx2 = min(w, bx1 + box_size)
        by2 = min(h, by1 + box_size)
        box_coords = (bx1, by1, bx2, by2)

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2)

        # Parmaklar birleştiğinde geri sayım moduna geç
        if distance < 45:
          selected_box = box_coords
          countdown_start_time = time.time()
          state = "COUNTDOWN"

    cv2.putText(
        frame,
        "Boyutu sec, parmaklari birlestirince sayim baslar",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

  elif state == "COUNTDOWN":
    elapsed_time = time.time() - countdown_start_time
    remaining_time = int(countdown_duration - elapsed_time) + 1

    if remaining_time > 0:
      cv2.putText(
          frame,
          str(remaining_time),
          (w // 2 - 40, h // 2 + 40),
          cv2.FONT_HERSHEY_SIMPLEX,
          4,
          (0, 0, 255),
          6,
      )
      cv2.putText(
          frame,
          "Hazirlan!",
          (w // 2 - 70, h // 2 - 60),
          cv2.FONT_HERSHEY_SIMPLEX,
          1,
          (255, 255, 0),
          2,
      )
    else:
      # Süre bitti, seçilen alanı kırpıp puzzle'ı hazırla
      if selected_box is not None:
        bx1, by1, bx2, by2 = selected_box
        if bx2 > bx1 and by2 > by1:
          cropped = frame[by1:by2, bx1:bx2]
          captured_frame = cv2.resize(cropped, (puzzle_size, puzzle_size))
        else:
          captured_frame = cv2.resize(frame, (puzzle_size, puzzle_size))
      else:
        captured_frame = cv2.resize(frame, (puzzle_size, puzzle_size))

      init_puzzle()
      puzzle_start_time = time.time()
      state = "PUZZLE"

  elif state == "PUZZLE":
    # Arka plana canlı kamerayı koyalım
    display_frame = frame.copy()
    display_frame = cv2.addWeighted(display_frame, 0.4, display_frame, 0, 0)

    # Yana kayma (Slide) animasyonu hesabı - Sol tarafa kayması için target_x_pos 40 yapıldı
    anim_progress = min(1.0, (time.time() - puzzle_start_time) / 0.8)
    start_x_pos = (w - puzzle_size) // 2
    target_x_pos = 40  # Sol kenara yakın
    start_y_pos = (h - puzzle_size) // 2
    target_y_pos = (h - puzzle_size) // 2

    current_puzzle_x = int(
        start_x_pos + (target_x_pos - start_x_pos) * anim_progress
    )
    current_puzzle_y = int(
        start_y_pos + (target_y_pos - start_y_pos) * anim_progress
    )

    # Küçük puzzle ekranını oluştur
    puzzle_canvas = np.zeros(
        (puzzle_size, puzzle_size, 3), dtype=np.uint8
    )

    for r in range(grid_size):
      for c in range(grid_size):
        val = board[r][c]
        dest_y1, dest_y2 = r * piece_h, (r + 1) * piece_h
        dest_x1, dest_x2 = c * piece_w, (c + 1) * piece_w

        if val != -1:
          orig_r = val // grid_size
          orig_c = val % grid_size
          src_y1, src_y2 = orig_r * piece_h, (orig_r + 1) * piece_h
          src_x1, src_x2 = orig_c * piece_w, (orig_c + 1) * piece_w

          puzzle_canvas[dest_y1:dest_y2, dest_x1:dest_x2] = captured_frame[
              src_y1:src_y2, src_x1:src_x2
          ]
        else:
          puzzle_canvas[dest_y1:dest_y2, dest_x1:dest_x2] = (30, 30, 30)

        cv2.rectangle(
            puzzle_canvas,
            (dest_x1, dest_y1),
            (dest_x2, dest_y2),
            (200, 200, 200),
            1,
        )

    # Puzzle'ı ana ekranın üzerine yerleştir
    display_frame[
        current_puzzle_y : current_puzzle_y + puzzle_size,
        current_puzzle_x : current_puzzle_x + puzzle_size,
    ] = puzzle_canvas

    # Çerçeve ekle
    cv2.rectangle(
        display_frame,
        (current_puzzle_x - 2, current_puzzle_y - 2),
        (current_puzzle_x + puzzle_size + 2, current_puzzle_y + puzzle_size + 2),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        display_frame,
        "Kamera calisiyor | Fare ile puzzle'i çöz ('r': basa don)",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
    )
    frame = display_frame

  cv2.imshow("Python Puzzle Filter", frame)

  key = cv2.waitKey(1) & 0xFF
  if key == 27:  # ESC
    break
  elif key == ord("r"):  # Normale dön
    state = "NORMAL"

cap.release()
cv2.destroyAllWindows()
