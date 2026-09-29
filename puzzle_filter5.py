import random
import time
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe el ve çizim modülleri
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

state = "NORMAL"  # NORMAL, COUNTDOWN, PUZZLE, COMPLETED
captured_frame = None
grid_size = 3
board = []
empty_pos = (2, 2)
piece_h, piece_w = 0, 0
puzzle_size = 350

# Sürükle-bırak için değişkenler
dragging = False
drag_start_pos = None

# Geri sayım için değişkenler
countdown_start_time = 0
countdown_duration = 3  # 3 saniye

selected_box = None
box_w, box_h = 300, 300

puzzle_target_x, puzzle_target_y = 0, 0
saved_completion_photo = False


def init_puzzle():
  global board, empty_pos, piece_h, piece_w, puzzle_size, saved_completion_photo
  piece_h = puzzle_size // grid_size
  piece_w = puzzle_size // grid_size
  saved_completion_photo = False

  board = [[r * grid_size + c for c in range(grid_size)] for r in range(grid_size)]
  empty_pos = (grid_size - 1, grid_size - 1)
  board[empty_pos[0]][empty_pos[1]] = -1

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

    if not check_win():
      break


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


def mouse_callback(event, x, y, flags, param):
  global board, empty_pos, state, dragging, drag_start_pos
  global puzzle_target_x, puzzle_target_y, puzzle_size, piece_w, piece_h, grid_size

  if state == "PUZZLE":
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


cv2.namedWindow("Video Puzzle Efekti")
cv2.setMouseCallback("Video Puzzle Efekti", mouse_callback)

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break

  frame = cv2.flip(frame, 1)
  h, w, c = frame.shape

  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = hands.process(rgb_frame)

  if state == "NORMAL":
    distance = 200
    hand_center_x, hand_center_y = w // 2, h // 2

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]

        x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
        x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

        # Parmaklar arasında mavi çizgi
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        distance = int(np.hypot(x2 - x1, y2 - y1))

        hand_center_x = int((x1 + x2) / 2)
        hand_center_y = int((y1 + y2) / 2)

    # Elin ortasında dinamik kare çerçeve (Videodaki gibi)
    bx1 = max(0, hand_center_x - box_w // 2)
    by1 = max(0, hand_center_y - box_h // 2)
    bx2 = min(w, bx1 + box_w)
    by2 = min(h, by1 + box_h)
    selected_box = (bx1, by1, bx2, by2)

    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 3)
    cv2.putText(
        frame,
        "Kareyi olustur ve parmaklari birlestir!",
        (bx1, by1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
    )

    # Parmaklar birleştiğinde (mesafe 50 pikselin altına düşerse) sayımı başlat
    if distance < 50:
      countdown_start_time = time.time()
      state = "COUNTDOWN"

    cv2.putText(
        frame,
        "Basparmak ve isaret parmagini birlestir!",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

  elif state == "COUNTDOWN":
    elapsed_time = time.time() - countdown_start_time
    remaining_time = int(countdown_duration - elapsed_time) + 1

    if selected_box:
      cv2.rectangle(
          frame,
          (selected_box[0], selected_box[1]),
          (selected_box[2], selected_box[3]),
          (0, 0, 255),
          4,
      )

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
          "FOTOGRAF CEKILIYOR...",
          (w // 2 - 130, h // 2 - 60),
          cv2.FONT_HERSHEY_SIMPLEX,
          1,
          (255, 255, 0),
          2,
      )
    else:
      if selected_box is not None:
        bx1, by1, bx2, by2 = selected_box
        cropped = frame[by1:by2, bx1:bx2]
        if cropped.size > 0:
          captured_frame = cv2.resize(cropped, (puzzle_size, puzzle_size))
        else:
          captured_frame = cv2.resize(frame, (puzzle_size, puzzle_size))
      else:
        captured_frame = cv2.resize(frame, (puzzle_size, puzzle_size))

      init_puzzle()
      state = "PUZZLE"

  elif state in ["PUZZLE", "COMPLETED"]:
    puzzle_target_x = w - puzzle_size - 80
    puzzle_target_y = (h - puzzle_size) // 2

    # Sol tarafta canlı kamerayı göstermeye devam et
    if selected_box:
      cv2.rectangle(
          frame,
          (selected_box[0], selected_box[1]),
          (selected_box[2], selected_box[3]),
          (100, 100, 100),
          2,
      )

    puzzle_canvas = np.zeros((puzzle_size, puzzle_size, 3), dtype=np.uint8)

    for r in range(grid_size):
      for c in range(grid_size):
        val = board[r][c]
        dest_y1, dest_y2 = r * piece_h, (r + 1) * piece_h
        dest_x1, dest_x2 = c * piece_w, (c + 1) * piece_w

        if val != -1 or state == "COMPLETED":
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

        if state != "COMPLETED":
          cv2.rectangle(
              puzzle_canvas,
              (dest_x1, dest_y1),
              (dest_x2, dest_y2),
              (200, 200, 200),
              1,
          )

    frame[
        puzzle_target_y : puzzle_target_y + puzzle_size,
        puzzle_target_x : puzzle_target_x + puzzle_size,
    ] = puzzle_canvas

    if state == "PUZZLE":
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
          "Puzzle'i fare ile coz | 'r': Basa Don",
          (15, 30),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 255, 255),
          2,
      )
    elif state == "COMPLETED":
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

  cv2.imshow("Video Puzzle Efekti", frame)

  key = cv2.waitKey(1) & 0xFF
  if key == 27:
    break
  elif key == ord("r"):
    state = "NORMAL"

cap.release()
cv2.destroyAllWindows()