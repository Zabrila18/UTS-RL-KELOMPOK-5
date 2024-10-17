import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
x_values = []
y_values = []

# Inisialisasi library pygame
pygame.init()
print("starting")

# Definisi model jaringan saraf untuk Q-Learning
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        # Membangun jaringan saraf sederhana dengan 3 layer fully connected
        self.fc = nn.Sequential(
            nn.Linear(8, 64), # Input 8 dimensi, hidden layer dengan 64 node
            nn.ReLU(), # Fungsi aktivasi ReLU
            nn.Linear(64, 32), # Layer ke-2 dengan 32 node
            nn.ReLU(), # Fungsi aktivasi ReLU
            nn.Linear(32, 2) # Output dengan 2 pilihan aksi (kiri atau kanan)
        )
        self._initialize_weights()

    # Inisialisasi bobot menggunakan kaidah Xavier
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # Fungsi untuk melakukan forward pass
    def forward(self, x):
        return self.fc(x)

# Definisi agen Q-Learning
class QLearningAgent:
    def __init__(self, learning_rate=0.00005, discount_factor=0.99, exploration_prob=1.0, exploration_decay=0.998):
        # Inisialisasi model Q-Network, optimizer, dan fungsi loss
        self.model = QNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss() # Mean squared error loss
        self.discount_factor = discount_factor # Faktor diskon untuk Q-Learning
        self.exploration_prob = exploration_prob # Probabilitas eksplorasi (epsilon)
        self.exploration_decay = exploration_decay # Faktor penurunan eksplorasi

    # Fungsi untuk memilih aksi berdasarkan Q-value atau eksplorasi acak
    def select_action(self, state):
        if random.random() < self.exploration_prob:
            return random.randint(0, 1) # Aksi acak (eksplorasi)
        with torch.no_grad():
            q_values = self.model(torch.FloatTensor(state)) # Hitung Q-value dari state
            return torch.argmax(q_values).item() # Pilih aksi dengan Q-value tertinggi

    # Fungsi untuk melatih agen berdasarkan reward dan state transition
    def train(self, state, action, reward, next_state, done):
        # Hitung Q-value untuk state berikutnya
        with torch.no_grad():
            target_q_values = self.model(torch.FloatTensor(next_state))
            max_target_q_value = torch.max(target_q_values)

        # Hitung Q-value untuk state saat ini
        q_values = self.model(torch.FloatTensor(state))
        target_q_value = reward + (1 - done) * self.discount_factor * max_target_q_value
        target_q_values = q_values.clone().detach()
        target_q_values[action] = target_q_value

        # Hitung dan perbarui loss
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward() # Backpropagation
        self.optimizer.step() # Optimisasi model

        # Jika game selesai, kurangi probabilitas eksplorasi
        if done:
            self.exploration_prob *= self.exploration_decay

# Membuat agen Q-Learning
agent = QLearningAgent()

# Fungsi utama untuk menjalankan permainan
def play(agent):
    WIDTH, HEIGHT = 500, 450
    BALL_RADIUS = 10
    PADDLE_WIDTH, PADDLE_HEIGHT = 80, 10
    BRICK_WIDTH, BRICK_HEIGHT = 40, 20
    BRICK_ROWS, BRICK_COLS = 5, 10
    WHITE = (255, 255, 255)

    total_reward = 0

    # Fungsi untuk membuat balok dalam permainan
    def create_bricks():
        bricks = []
        for i in range(BRICK_ROWS):
            for j in range(BRICK_COLS):
                brick = pygame.Rect(j * (BRICK_WIDTH + 5) + 25, i * (BRICK_HEIGHT + 5) + 25, BRICK_WIDTH, BRICK_HEIGHT)
                bricks.append(brick)
        return bricks

    # Inisialisasi paddle, bola, dan bricks
    paddle = pygame.Rect(WIDTH // 2 - PADDLE_WIDTH // 2, HEIGHT - 30, PADDLE_WIDTH, PADDLE_HEIGHT)
    
    lowest_brick_point = (BRICK_HEIGHT + 5) * BRICK_ROWS + 25
    ball = pygame.Rect(WIDTH // 2 - BALL_RADIUS, HEIGHT // 2 - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)

    bricks = create_bricks()

    # Arah gerakan awal bola
    ball_dx = random.choice([3, -3])
    ball_dy = 3

    # Membuat layar permainan dengan pygame
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    # Fungsi untuk mendapatkan vektor state
    def get_state_vector(ball, paddle, ball_dx, ball_dy, bricks):
        state_vector = [0] * 8
        
        # Posisi bola relatif terhadap paddle
        if ball.centerx < paddle.centerx:
            state_vector[0] = 1
        else:
            state_vector[1] = 1

        # Arah bola
        if ball_dy < 0 and ball_dx > 0:
            state_vector[2] = 1 # Timur Laut
        elif ball_dy < 0 and ball_dx < 0:
            state_vector[3] = 1 # Barat Laut
        elif ball_dy > 0 and ball_dx > 0:
            state_vector[4] = 1 # Tenggara
        elif ball_dy > 0 and ball_dx < 0:
            state_vector[5] = 1 # Barat Daya

        # Apakah masih ada balok di kiri atau kanan paddle
        left_bricks_present = any(brick.centerx < paddle.centerx for brick in bricks)
        right_bricks_present = any(brick.centerx > paddle.centerx for brick in bricks)

        if left_bricks_present:
            state_vector[6] = 1
        if right_bricks_present:
            state_vector[7] = 1

        return state_vector

    # Inisialisasi variabel state
    state_vector = get_state_vector(ball, paddle, ball_dx, ball_dy, bricks)

    running = True
    while running:
        reward = 0
        screen.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if len(bricks) == 0:
            running = False
            
        state_vector = get_state_vector(ball, paddle, ball_dx, ball_dy, bricks)
        # Agen memilih aksi (bergerak ke kiri atau kanan)
        action = agent.select_action(state_vector)
        if action == 0 and paddle.left > 0:
            paddle.move_ip(-5, 0)
        elif action != 0 and paddle.right < WIDTH:
            paddle.move_ip(5, 0)

        # Memindahkan bola
        ball.move_ip(ball_dx, ball_dy)
        if ball.left <= 0 or ball.right >= WIDTH:
            ball_dx = -ball_dx
        if ball.top <= 0:
            ball_dy = -ball_dy
        if ball.bottom >= HEIGHT:
            distance_from_paddle = abs((paddle.left + paddle.right) / 2 - (ball.left + ball.right) / 2)
            reward = 0 - distance_from_paddle / 25
            running = False
        if ball.colliderect(paddle):
            if ball_dy < 0:
                ball_dy = random.uniform(2, 4)
            else:
                ball_dy = 0 - random.uniform(2, 4)
            reward = 15
        for brick in bricks[:]:
            if ball.colliderect(brick):
                reward = 2
                bricks.remove(brick)
                ball_dy = -ball_dy

        # Menggambar elemen permainan
        pygame.draw.ellipse(screen, (0, 0, 0), ball)
        pygame.draw.rect(screen, (0, 0, 0), paddle)
        for brick in bricks:
            pygame.draw.rect(screen, (0, 0, 0), brick)

        pygame.display.flip()
        next_state_vector = get_state_vector(ball, paddle, ball_dx, ball_dy, bricks)
        agent.train(state_vector, action, reward, next_state_vector, running)
        total_reward += reward

        if not running:
            return total_reward

        clock.tick(60)

# Fungsi untuk memplot perkembangan skor
def plot_xy(x_values, y_values, title="Progress", x_label="Iteration", y_label="Score"):
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.show()

# Loop untuk menjalankan game sebanyak 200 iterasi
for i in range(200):
    reward = int(play(agent))
    x_values.append(i)
    y_values.append(reward)
    print(i, ". ", reward, sep = "")

# Plot hasil perkembangan agen
plot_xy(x_values, y_values)
