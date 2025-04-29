import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp
import pygame
import sys
import math
import time
import collections
import pygame.gfxdraw


# Initialize pygame
pygame.init()
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Controlled Tower of Hanoi - Sabato La Manna") 
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PEG_COLOR = (100, 100, 100)
DISC_COLORS = [(55, 237, 26), (237, 26, 188), (26, 237, 157)]
GLOW_COLOR = (55, 237, 26)  


# Peg positions
pegs = [WIDTH // 4, WIDTH // 2, 3 * WIDTH // 4]
peg_height = 300

# Disc setup
start_discs = [3, 2, 1]
discs = [[*start_discs], [], []]

# Hand detection setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.85,
    min_tracking_confidence=0.85
)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Game state
holding_disc = None
holding_from = None
holding_pos_x = None
won = False
move_counter = 0
start_time = time.time()
end_time = None

# For smoothing hand position
x_history = collections.deque(maxlen=5)

font = pygame.font.SysFont(None, 50)

def reset_game():
    global discs, holding_disc, holding_from, won, move_counter, start_time, end_time
    discs = [[*start_discs], [], []]
    holding_disc = None
    holding_from = None
    won = False
    move_counter = 0
    start_time = time.time()
    end_time = None
    x_history.clear()

def draw_game(frame):
    # Convert OpenCV frame to pygame surface
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame_surface = pygame.surfarray.make_surface(frame)
    screen.blit(pygame.transform.rotate(frame_surface, -90), (0, 0))

    # Draw pegs
    for idx, peg_x in enumerate(pegs):
        # Glow effect if hand is near this peg and the game is not won
        if holding_pos_x is not None and get_peg_from_x(holding_pos_x) == idx and not won:
            pygame.gfxdraw.filled_circle(screen, peg_x, HEIGHT - peg_height // 2, 60, (GLOW_COLOR[0], GLOW_COLOR[1], GLOW_COLOR[2], 80))
            pygame.gfxdraw.aacircle(screen, peg_x, HEIGHT - peg_height // 2, 60, GLOW_COLOR)

        pygame.draw.rect(screen, PEG_COLOR, (peg_x - 5, HEIGHT - peg_height, 10, peg_height))

    # Draw discs (biggest at bottom)
    for idx, peg in enumerate(discs):
        for h_idx, disc in enumerate(peg):
            width = disc * 40
            height = 20
            x = pegs[idx]
            y = HEIGHT - 20 - h_idx * 25
            pygame.draw.rect(screen, DISC_COLORS[disc - 1], (x - width // 2, y - height // 2, width, height))

    # Draw floating disc if holding
    if holding_disc and holding_pos_x is not None and not won:
        width = holding_disc * 40
        height = 20
        x_pixel = holding_pos_x * WIDTH
        y_pixel = HEIGHT // 4
        pygame.draw.rect(screen, DISC_COLORS[holding_disc - 1], (x_pixel - width // 2, y_pixel - height // 2, width, height))

    # Display Move Counter (only if game is not won)
    move_text = font.render(f"Moves: {move_counter if not won else move_counter}", True, BLACK)
    screen.blit(move_text, (10, 10))

    # Display Timer (only if game is not won)
    elapsed = (end_time if end_time else time.time()) - start_time
    timer_text = font.render(f"Time: {int(elapsed) if not won else int(elapsed)}s", True, BLACK)
    screen.blit(timer_text, (10, 60))

    # If won
    if won:
        text = font.render("You Won!", True, (0, 128, 0))
        screen.blit(text, (WIDTH // 2 - text.get_width() // 2, 50))

    pygame.display.update()

def detect_hand_position_and_gesture(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    hand_info = []

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            x = hand_landmark.landmark[8].x  # Index fingertip
            y = hand_landmark.landmark[8].y

            # Better pinch detection
            thumb_tip = hand_landmark.landmark[4]
            index_tip = hand_landmark.landmark[8]
            middle_tip = hand_landmark.landmark[12]

            pinch_index = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
            pinch_middle = math.hypot(thumb_tip.x - middle_tip.x, thumb_tip.y - middle_tip.y)

            hand_closed = pinch_index < 0.05 and pinch_middle < 0.08

            hand_info.append({'x': x, 'y': y, 'closed': hand_closed})

    return hand_info, results

def get_peg_from_x(x_normalized):
    x_pixel = x_normalized * WIDTH
    if x_pixel < WIDTH / 3:
        return 0
    elif x_pixel < 2 * WIDTH / 3:
        return 1
    else:
        return 2

def check_victory():
    return len(discs[2]) == 3

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            sys.exit()

    success, img = cap.read()
    if not success:
        continue

    raw_img = img.copy()  # Save raw image for mediapipe
    hand_infos, results = detect_hand_position_and_gesture(raw_img)

    img = cv2.flip(img, 1)  # Mirror *after* hand detection
    if hand_infos and not won:
        main_hand = hand_infos[0]
        x_history.append(main_hand['x'])
        smoothed_x = sum(x_history) / len(x_history)
        holding_pos_x = smoothed_x
        peg_index = get_peg_from_x(smoothed_x)
        if not holding_disc and main_hand['closed'] and discs[peg_index]:
            holding_disc = discs[peg_index].pop()
            holding_from = peg_index
        if holding_disc and not main_hand['closed']:
            if not discs[peg_index] or holding_disc < discs[peg_index][-1]:
                discs[peg_index].append(holding_disc)
                holding_disc = None
                holding_from = None
                move_counter += 1
                if check_victory():
                    won = True
                    end_time = time.time()
            else:
                discs[holding_from].append(holding_disc)
                holding_disc = None
                holding_from = None
    # Draw hand wireframe
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmark.x = 1 - landmark.x
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    draw_game(frame=img)
    clock.tick(30)