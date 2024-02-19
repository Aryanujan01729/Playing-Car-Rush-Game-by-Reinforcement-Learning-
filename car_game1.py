import os
import sys
import time
import random 
import pygame as pg


BLACK = (0,0,0)
WHITE = (255, 255, 255)
BLUE = (0,0,255)

DISPLAY_WIDTH = 1100
DISPLAY_HEIGHT = 800

PLAYER_MOVEMENT_SPEED = 10

CAR_WIDTH = 90
CAR_HEIGHT = 120

CLOCK_TICKRATE = 1000

file = open("scores.txt", "a+")

class car():
    def __init__(self, x, y, speed, name):
        self.image = pg.image.load(os.path.join(sys.path[0], f'assets\{name}.png'))
        self.image = pg.transform.scale(self.image, (CAR_WIDTH, CAR_HEIGHT))
        self.x = x
        self.y = y
        self.speed = speed


class traffic():
    def __init__(self, x, y, speed, name):
        self.image = pg.image.load(os.path.join(sys.path[0], f'assets\{name}.png'))
        self.image = pg.transform.scale(self.image, (CAR_WIDTH, CAR_HEIGHT))
        self.x = x
        self.y = y
        self.speed = speed

    def move(self):
        self.y+=self.speed


def collided_with(self, player):
        if (self.x <= player.x + 10 <= self.x + 90) or \
                (self.x <= player.x + 90 - 10 <= self.x + 90):
            return (self.y <= player.y + 5 <= self.y + 120) or \
                   (self.y <= player.y + 120 - 5 <= self.y + 120)
        return False


pg.init()

window = pg.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pg.display.set_caption('Mehenga Road Rash')
background = pg.image.load(os.path.join(sys.path[0], r'assets\background.png'))
background = pg.transform.scale(background, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
background_x = 0
background_y = 0
background2_y = -DISPLAY_HEIGHT
background_speed = 5

font = pg.font.Font('freesansbold.ttf', 30)
font1 = pg.font.Font('freesansbold.ttf', 48)

paused_text = font.render('GAME PAUSED - press space to resume', True, WHITE, BLACK)
paused_textRect = paused_text.get_rect()
paused_textRect.center = (DISPLAY_WIDTH // 2, DISPLAY_HEIGHT - 150)

text_1 = font.render('1', True, WHITE)
text_1Rect = paused_text.get_rect()
text_1Rect.center = (527, 280)

text_2 = font.render('2', True, WHITE)
text_2Rect = paused_text.get_rect()
text_2Rect.center = (1130, 280)

text_3 = font.render('3', True, WHITE)
text_3Rect = paused_text.get_rect()
text_3Rect.center = (526, 610)

text_4 = font.render('4', True, WHITE)
text_4Rect = paused_text.get_rect()
text_4Rect.center = (1130, 610)

text_carchoice = font1.render('SELECT A CAR', True, WHITE)
text_carchoiceRect = paused_text.get_rect()
text_carchoiceRect.center = (680, DISPLAY_HEIGHT - 100)

cars = ['red_car', 'yellow_car', 'green_car', 'purple_car']
cars_pos = [[200, 100], [800, 100], [200, 425], [800, 425]]

traffic_cars = ['traffic_car_1', 'traffic_car_2', 'traffic_car_3', 'traffic_car_4']

police1 = car(DISPLAY_WIDTH // 2 - 208, DISPLAY_HEIGHT - 150, speed=PLAYER_MOVEMENT_SPEED, name='police')
police2 = car(DISPLAY_WIDTH // 2 + 135, DISPLAY_HEIGHT - 150, speed=PLAYER_MOVEMENT_SPEED, name='police')

font2 = pg.font.SysFont('arial', 60)
crashed_text = font2.render('YOU CRASHED!', True, WHITE, BLACK)
crashed_textRect = crashed_text.get_rect()
crashed_textRect.center = (DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2)


def cscore(s):
    crash_score = font1.render(f'Score: {s}', True, WHITE)
    crash_scoreRect = crash_score.get_rect()
    crash_scoreRect.center = (DISPLAY_WIDTH // 2, (DISPLAY_HEIGHT // 2)+50)
    window.blit(crash_score, crash_scoreRect)


def hscoref():
    h_score = font1.render('HIGH SCORE!', True, BLUE)
    h_scoreRect = h_score.get_rect()
    h_scoreRect.center = (DISPLAY_WIDTH // 2, (DISPLAY_HEIGHT // 2)+100)
    window.blit(h_score, h_scoreRect)
    
font3 = pg.font.SysFont('arial' , 30)

def score(s):
    score = font3.render(f'Score: {s}', True, BLACK, WHITE)
    scoreRect = score.get_rect()
    scoreRect.center = (50,20)
    window.blit(score, scoreRect)


def get_sound(name, vol):
    sound = pg.mixer.Sound(os.path.join(sys.path[0], f'assets./sfx/{name}.wav'))
    sound.set_volume(vol)
    return sound


crash_sound = get_sound('crash', 0.5)
police_siren = get_sound('police_siren', 0.1)
pg.mixer_music.load(os.path.join(sys.path[0], 'assets/sfx/jazz_in_paris.wav'))
pg.mixer_music.set_volume(0)

def play_music():
    pg.mixer_music.set_volume(0.2)
    pg.mixer_music.rewind()


def choose_a_car():
    car1_image = pg.image.load(os.path.join(sys.path[0], f'assets/red_car.png'))
    car1_image = pg.transform.scale(car1_image, (75, 140))
    car2_image = pg.image.load(os.path.join(sys.path[0], f'assets/yellow_car.png'))
    car2_image = pg.transform.scale(car2_image, (75, 140))
    car3_image = pg.image.load(os.path.join(sys.path[0], f'assets/green_car.png'))
    car3_image = pg.transform.scale(car3_image, (75, 140))
    car4_image = pg.image.load(os.path.join(sys.path[0], f'assets/purple_car.png'))
    car4_image = pg.transform.scale(car4_image, (75, 140))
    window.fill(BLACK)
    window.blit(car1_image, cars_pos[0])
    window.blit(car2_image, cars_pos[1])
    window.blit(car3_image, cars_pos[2])
    window.blit(car4_image, cars_pos[3])
    window.blit(text_1, text_1Rect)
    window.blit(text_2, text_2Rect)
    window.blit(text_3, text_3Rect)
    window.blit(text_4, text_4Rect)
    window.blit(text_carchoice, text_carchoiceRect)


game_running = True
dx = 0
tspeed = 6
clock = pg.time.Clock()
car_chosen = False
i = 0
time_passed = 0
loop = False
traffic_car = [0,0,0,0,0,0,0,0,0]
start_time = 0
z=0
counter = 0
trspeed = 6
bruh = 0
temp =5
spawn_time = 1
t2 = False
spawned = False
lanes = [250,425,600,775]
crashed = False
crashsoundcheck = False
score_written = False
max = 0
file_closed = False
high_score = False
pg.mixer_music.play(-1)


while game_running:

    for event in pg.event.get():
        if event.type == pg.QUIT:

            game_running = False

        if event.type == pg.KEYDOWN and not car_chosen:
            
            if event.key == pg.K_1:
                car_choice = cars[0]
                
                car_chosen = True
            elif event.key == pg.K_2:
                car_choice = cars[1]
                
                car_chosen = True
            elif event.key == pg.K_3:
                car_choice = cars[2]
                
                car_chosen = True
            elif event.key == pg.K_4:
                car_choice = cars[3]
                car_chosen = True

            if car_chosen:
                player = car((DISPLAY_WIDTH // 2) - 47, DISPLAY_HEIGHT - 225, PLAYER_MOVEMENT_SPEED, car_choice)

        if event.type == pg.KEYDOWN and car_chosen:
            if event.key == pg.K_a or event.key == pg.K_LEFT:
                dx = -PLAYER_MOVEMENT_SPEED
            if event.key == pg.K_d or event.key == pg.K_RIGHT:
                dx = PLAYER_MOVEMENT_SPEED
            
        if event.type == pg.KEYUP:
            dx = 0


    if crashed:
        pg.mixer_music.stop()
        if crashsoundcheck == False:
            crash_sound.play()
        crashsoundcheck = True
        window.fill(BLACK)
        window.blit(crashed_text, crashed_textRect)
        cscore(counter)
        if high_score:
            hscoref()
        if score_written == False:
            if file_closed == False:
                file.write(f'{counter}\n')
            file_closed = True
            file.close()
            file2 = open("scores.txt", "r")
            f1 = file2.readlines()
            for x in f1:
                print(int(x))
                if int(x) > max:
                    max = int(x)
            if max == counter:
                print("High Score")
                high_score = True
            score_written = True
  
    if car_chosen and not crashed:
        if time_passed < 5400:
            police_siren.play(0)
            background_y+=10
            background2_y+=10
            if background_y > DISPLAY_HEIGHT:
                background_y = -DISPLAY_HEIGHT
            if background2_y > DISPLAY_HEIGHT:
                background2_y = -DISPLAY_HEIGHT
            window.blit(background, (background_x, background2_y))
            window.blit(background, (background_x, background_y))
            window.blit(player.image, (player.x, player.y))
            window.blit(police1.image, (police1.x, police1.y))
            window.blit(police2.image, (police2.x, police2.y))
        else: 
            if time_passed > 6000:
                police_siren.stop()
            if time_passed > 6200:
                play_music()

            player.x += dx

            if player.x < 210:
                player.x = 210
            if player.x > DISPLAY_WIDTH - 285:
                player.x = DISPLAY_WIDTH - 285

            police1.y += 3
            police2.y += 3
  
            if background_y >= DISPLAY_HEIGHT:
                background_y = -DISPLAY_HEIGHT

            if background2_y >= DISPLAY_HEIGHT:
                background2_y = -DISPLAY_HEIGHT

            background_y+=background_speed
            background2_y+=background_speed

            if player.x < 195:
                player.x = 195

            if player.x > DISPLAY_WIDTH - CAR_WIDTH - 190:
                player.x = DISPLAY_WIDTH - CAR_WIDTH - 190

            if (time.time()) - start_time > spawn_time or z == 0: 
                start_time = (time.time())
                if z == 1:
                    counter+=1
                    n = random.randint(0,3)
                    nt = random.randint(0,3)
                
                    if temp == n:
                        bruh+=1
                    else:
                        bruh = 0
                    if bruh == 2:
                        while n==temp:
                            n = random.randint(0,3)
                        bruh = 0
                    if counter%10 == 0 and trspeed < 14:
                        trspeed += 1
                    if counter % 20 == 0 and spawn_time>0.4:
                        spawn_time-=0.2

                    round(spawn_time, 2)
                    temp = n
                    if trspeed != 4 or t2 == True:
                        traffic_car[i] = traffic(lanes[n], -130, trspeed, traffic_cars[nt])
                    if trspeed == 4:
                        t2 = True
                    i+=1
                    spawned = True

                z = 1

            if spawned:
                if loop == False:
                    for y in range(i):
                        traffic_car[y].move()
                else:
                    for y in range(8):
                        traffic_car[y].move()
           
            window.blit(background, (background_x,background2_y))
            window.blit(background, (background_x,background_y))
            window.blit(player.image, (player.x, player.y))

            if police1.y<DISPLAY_HEIGHT:
                window.blit(police1.image, (police1.x, police1.y))
                window.blit(police2.image, (police2.x, police2.y))
       
            if loop:
                for k in range(8):
                    window.blit(traffic_car[k].image, (traffic_car[k].x,traffic_car[k].y))
            else:
                for k in range(i):
                    window.blit(traffic_car[k].image, (traffic_car[k].x,traffic_car[k].y))
            if loop:
                for k in range(8):
                    if(collided_with(traffic_car[k], player)):
                        print("Crash")
                        crashed = True 
            else:
                for k in range(i):
                    if(collided_with(traffic_car[k], player)):
                        print("Crash")
                        crashed = True
            if i>=8:
                i = 0
                loop = True
            score(counter)
            
    else:
        if not crashed:
            choose_a_car()

    if car_chosen:
        time_passed += clock.get_time()

    pg.display.update()
    clock.tick(CLOCK_TICKRATE)

file2.close()    
pg.quit()




