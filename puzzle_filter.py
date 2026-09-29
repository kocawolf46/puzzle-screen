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

state = "NORMAL"  # NORMAL, PUZZLE
captured_frame = None
grid_size = 3
board = []
empty_pos = (2, 2)
piece_h, piece_w = 0, 0

# Sürükle-bırak için değişkenler
dragging = False
drag_start_pos = None


def init_puzzle(frame):
  global board, empty_pos, piece_h, piece_w
  h, w, _ = frame.shape
  piece_h = h // grid_size
  piece_w = w // grid_size

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


# Fare olaylarını yakalayan fonksiyon (Tıklama ve Sürükleme)
def mouse_callback(event, x, y, flags, param):
  global board, empty_pos, state, dragging, drag_start_pos

  if state == "PUZZLE":
    clicked_c = x // piece_w
    clicked_r = y // piece_h

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

        # Eğer başlangıç ve bırakılan yer farklıysa ve boş yer ile hedef yer birleştirilebilirse
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
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]

        x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
        x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        distance = int(np.hypot(x2 - x1, y2 - y1))

        if distance < 50:
          captured_frame = frame.copy()
          init_puzzle(captured_frame)
          state = "PUZZLE"

    cv2.putText(
        frame,
        f"Mesafe: {distance} (Fotograf cekmek icin parmaklarini birlestir)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

  elif state == "PUZZLE":
    if captured_frame is not None:
      display_frame = np.zeros_like(captured_frame)

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

            display_frame[dest_y1:dest_y2, dest_x1:dest_x2] = captured_frame[
                src_y1:src_y2, src_x1:src_x2
            ]
          else:
            display_frame[dest_y1:dest_y2, dest_x1:dest_x2] = (30, 30, 30)

          cv2.rectangle(
              display_frame,
              (dest_x1, dest_y1),
              (dest_x2, dest_y2),
              (200, 200, 200),
              1,
          )

      frame = display_frame
      cv2.putText(
          frame,
          "Puzzle Modu! Tikla veya Surukle ('r' normale doner)",
          (15, 30),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 0, 255),
          2,
      )

  cv2.imshow("Python Puzzle Filter", frame)

  key = cv2.waitKey(1) & 0xFF
  if key == 27:  # ESC
    break
  elif key == ord("r"):  # Normale dön
    state = "NORMAL"

cap.release()
cv2.destroyAllWindows()