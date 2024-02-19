import os
import sys
import time
import random 
import pygame as pg
import cv2
import numpy as np
import json
import pygame.locals
from PIL import ImageGrab
from collections import defaultdict

AGENT_THRESHOLD=1000
BLACK = (0,0,0)
WHITE = (255, 255, 255)
BLUE = (0,0,255)

DISPLAY_WIDTH = 1100
DISPLAY_HEIGHT = 800


PLAYER_X_MOVEMENT_SPEED = 175
CAR_WIDTH = 90
CAR_HEIGHT = 120
PLAYER_MOVEMENT_SPEED = 175 
CLOCK_TICKRATE = 100

file = open("scores.txt", "a+")

class car():
    def __init__(self, x, y, speed, name):
        self.image = pg.image.load(os.path.join(sys.path[0], f'assets/{name}.png'))
        self.image = pg.transform.scale(self.image, (CAR_WIDTH, CAR_HEIGHT))
        self.x = x
        self.y = y
        self.speed = speed

class traffic():
    def __init__(self, x, y, speed, name):
        self.image = pg.image.load(os.path.join(sys.path[0], f'assets/{name}.png'))
        self.image = pg.transform.scale(self.image, (CAR_WIDTH, CAR_HEIGHT))
        rgb_value = self.image.get_at((int(CAR_WIDTH/2),int(CAR_HEIGHT/2)))
        self.rgb=rgb_value
        self.x = x
        self.y = y
        self.speed = speed

    def move(self):
        self.y+=self.speed

def agent_rec(color,mov_obj):
    moving_obj=[]
    player=[]
    #car_name=["police","green_car","purple_car","red_car","traffic_car_1","traffic_car_2","traffic_car_3","traffic_car_4","yellow_car"]
    #print(len(mov_obj))
    #print(color)
    #for m in mov_obj:
        #print(m.image.get_at((int(CAR_WIDTH/2),int(CAR_HEIGHT/2))))
    for obj in mov_obj:
        #raw_image=pg.image.load(os.path.join(sys.path[0], f'assets/{name}.png'))
        #image=pg.transform.scale(raw_image, (CAR_WIDTH, CAR_HEIGHT))
        if (color==obj.image.get_at((int(CAR_WIDTH/2),int(CAR_HEIGHT/2)))):
            player.append(obj)
        else:
            moving_obj.append(obj)
    return player[0],moving_obj

def collided_with(self, player):
        if (self.x <= player.x + 10 <= self.x + 90) or \
                (self.x <= player.x + 90 - 10 <= self.x + 90):
            return (self.y <= player.y + 5 <= self.y + 120) or \
                   (self.y <= player.y + 120 - 5 <= self.y + 120)
        return False


pg.init()

window = pg.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT),pg.RESIZABLE)
pg.display.set_caption('Mehenga Road Rash')
background = pg.image.load(os.path.join(sys.path[0], r'assets/background.png'))
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
    sound = pg.mixer.Sound(os.path.join(sys.path[0], f'assets/sfx/{name}.wav'))
    sound.set_volume(vol)
    return sound
def dist_norm(st,ed):
    #import pdb;pdb.set_trace()
    dist=st-ed
    if dist<0:
        dist=dist*-1
    if dist> 700:
        return 7
    elif dist > 600:
        return 6
    elif dist > 500:
        return 5
    elif dist > 400:
        return 4
    elif dist > 300:
        return 3
    elif dist > 200:
        return 2
    elif dist > 100:
        return 1
    else:
        return 0
def update_state(state,x_cor,player_r,opponent):
    #print("//////////////////////////////////////////////////////")
    #print(x_cor)
    y_cor=player_r.y
    x_oppo=opponent.x
    #print(x_oppo)
    y_oppo=opponent.y
    w=CAR_WIDTH/2 + 10
    #print((x_cor-w))
    #print((x_cor+w))
    #print("???????????????????????????")
    #print((x_oppo-w))
    #print((x_oppo+w))
    #if ((x_cor-w)<=(x_oppo+w) and (x_cor+w)>=(x_oppo-w) and (x_cor+w)>=(x_oppo+w) and (x_cor-w)>=(x_oppo-w)) or ((x_cor-w)<=(x_oppo+w) and (x_cor+w)>=(x_oppo-w) and (x_cor+w)<=(x_oppo+w) and (x_cor-w)<=(x_oppo-w)):
    if ((x_cor+w)>=(x_oppo+w)>=(x_cor-w)) or ((x_cor+w)>=(x_oppo-w)>=(x_cor-w)):
        '''print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print(y_cor)
        print(y_oppo)
        print(np.abs(y_cor-y_oppo))'''
        return dist_norm(y_cor,y_oppo)
    else:
        return 4
    
crash_sound = get_sound('crash', 0.5)
police_siren = get_sound('police_siren', 0.1)
pg.mixer_music.load(os.path.join(sys.path[0], 'assets/sfx/jazz_in_paris.wav'))
pg.mixer_music.set_volume(0)
 

Q =defaultdict(lambda:[0.,0.,0.] )

def play_music():
    pg.mixer_music.set_volume(0.2)
    pg.mixer_music.rewind()

def process_state(player_rec,moving_object):
    state_len=7
    #print("##############################")
    #print(len(moving_object))
    state =  np.repeat(a = 4, repeats = state_len)
    state_list=[]
    min_close=[]
    x_c=player_rec.x
    # input()
    for ind in range(state_len):
        if ind==0:
            if x_c > 250:
                state[ind]=1
            else:
                state[ind]=0
                print("**********************************")
        if ind==1:
            if x_c < 775: #DISPLAY_WIDTH-350 :
                state[ind]=1
            else:
                state[ind]=0
                print("/////////////////////////////////////")
        elif ind>=2:
            x_c=player_rec.x+(ind-4)*(PLAYER_MOVEMENT_SPEED+5)
            #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            for traff in moving_object:
                #print(traff.x,traff.y,player_rec.x,player_rec.y)
                min_close.append(update_state(state,x_c,player_rec,traff))
            if len(min_close) > 0:
                mini=np.min(min_close)
                state[ind]=mini
        min_close=[]
    #return state
   # action(state)
    min_close=[]
    state = [str(element) for element in state]
    stat = "".join(state)
    #print(stat)
    #print("kkkkkkkk")
    return stat
    
    #print(state_list)
    with open('q_table.txt','w') as convert_file:
                #print("*******************************************")
                convert_file.write(json.dumps(st)+'\n')
    q_learning(st)
    

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
def find_moving_object_color(all_obj_lst,agnt):
    m_lst=[]
    for obj in all_obj_lst:
        if obj != agnt :
            m_lst.append(obj)
    return m_lst
player_rect=[]
turn=0
total_car_object_colors = {}

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        #print("jjjjjjjjj")
        #print(observation)
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

# %%
def action(Q,state):
    policy = make_epsilon_greedy_policy(Q,0.3,3)# epsilon, env.action_space.n)
    action_probs = policy(state)
    actn = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    return actn

def q_learning(Q, state, action, reward, next_state, discount_factor=0.9, alpha=0.3, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    # import pdb;pdb.set_trace()

    
    reward = reward/30.
    best_next_action = np.argmax(Q[next_state])    
    td_target = reward + discount_factor * Q[next_state][best_next_action]
    # print(state,action)
    # print(Q[state][action])
    #print("rrrrrrrr")
    #print(action)
    td_delta = td_target - Q[state][action]
    Q[state][action] += alpha * td_delta
    Q[state][action] = np.round(Q[state][action],2)
    #print("llllllllllll")
    #print(Q)
    with open('Q.txt','w+',encoding='utf-8') as convert_file:
                #print("*******************************************")
                convert_file.write(json.dumps(Q)+'\n')
    
    return Q,td_delta
curr_state=None
next_state=None

while True:
    scrn_obj=[]
    mov_obj=[]
    game_running = True
    #find_object_frequency(screenshot)
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
    lanes = [250,425,600,775] # lanes x coordinate
    crashed = False
    crashsoundcheck = False
    score_written = False
    max = 0
    file_closed = False
    high_score = False
    # pg.mixer_music.play(-1)
    print(car_chosen,crashed)
    car_choice = cars[0] 
    car_chosen = True
    # player = car((DISPLAY_WIDTH // 2) - 47, DISPLAY_HEIGHT - 225, PLAYER_MOVEMENT_SPEED, car_choice)
    player = car(600, DISPLAY_HEIGHT - 225, PLAYER_MOVEMENT_SPEED, car_choice)

    while game_running:
        #print("her1")
        # input()
        dx=0
        for event in pg.event.get():
            if event.type == pg.QUIT:
                game_running = False
            if event.type == pg.KEYDOWN and car_chosen:
                if event.key == pg.K_a or event.key == pg.K_LEFT:
                    dx = -PLAYER_X_MOVEMENT_SPEED
                    #import pdb;pdb.set_trace()
                if event.key == pg.K_d or event.key == pg.K_RIGHT:
                    dx = PLAYER_X_MOVEMENT_SPEED
            # print("DXX",dx)
            # if event.type == pg.KEYUP:
            #     dx = 0
           # print("her2")
        #print("her3")
        #print(player.x,player.y , "player")
        cur_car_objects = [player,police1,police2]
        for tc in traffic_car:
            if tc !=0:
                cur_car_objects.append(tc)
        
        #print(len(cur_car_objects))

        # cur_car_objects = cur_car_objects+ traffic_car
        scrn_obj=[]
        for objects in cur_car_objects:
            if objects !=0 and  0<objects.y< DISPLAY_HEIGHT:
                scrn_obj= scrn_obj+ [objects]
                #import pdb;pdb.set_trace()
        #print(len(scrn_obj))
        cur_car_objects_color = [car.image.get_at((int(CAR_WIDTH/2),int(CAR_HEIGHT/2))) if car is not 0 else (0,0,0,0) for car in scrn_obj ]
        # # import pdb;pdb.set_trace()
        if turn<500:
            for color in cur_car_objects_color:
                if tuple(color) not in total_car_object_colors.keys():
                    total_car_object_colors[tuple(color)] = 0
                else:
                    total_car_object_colors[tuple(color)]+=1
            # print(total_car_object_colors)
        if turn== 500:
            # import pdb;pdb.set_trace()
            agent = next(iter(total_car_object_colors))
            for key in total_car_object_colors:
                if total_car_object_colors[key] > total_car_object_colors[agent] and key != (0,0,0,255):
                    agent = key
            #print(agent)
            
            #player_rect = car((DISPLAY_WIDTH // 2) - 47, DISPLAY_HEIGHT - 225, PLAYER_MOVEMENT_SPEED, nam)
            # print("Tunr",agent, total_car_object_colors[agent])
            # input()
        # print("Tunr",turn)
        turn=turn+1
        actn=0
        if turn>500:
            #input()
            mov_obj_col=find_moving_object_color(total_car_object_colors,agent)
            moving_object=[]
            
            #print(len(scrn_obj))
            player_rect,moving_object= agent_rec(agent,scrn_obj)
            #print(player_rect.x)
            
            '''for col in mov_obj_col:
                moving_object.append(move_rec(col,scrn_obj))
            print(moving_object)'''

            #print("Agent color",agent," moving obj colors",mov_obj_col)
            #print(agent)
            #nam= agent_rec(agent)
            #print(nam)
            '''moving_object=[]
            for obj in cur_car_objects:
                cur_car_color= obj.image.get_at((int(CAR_WIDTH/2),int(CAR_HEIGHT/2)))
                # print(cur_car_objects)
                if cur_car_color == agent:
                    agent_obj = obj
                else:
                    moving_object.append((cur_car_color))
            #print(moving_object)'''
            # input()
                    

            st=process_state(player_rect,moving_object)
            next_state=st
            actn=0
            if next_state is not None and curr_state is not None:
                actn=action(Q,curr_state)
                #print(actn)
                q_learning(Q, curr_state, actn, reward, next_state, discount_factor=0.9, alpha=0.3, epsilon=0.1)
            #act=action(Q,st)     
             #self.action = input()
            act=input()
            if act == '1':
                act = int(act) 
            elif act == '2':
               act = int(act) 
            else:
                act= 0
            actn=act
            #print("++++++++++++++++++++++++++++++++++++++++++++++++++++")


            if actn == 1: # 1073741904,    
                newevent = pygame.event.Event(pygame.locals.KEYDOWN, key=pygame.locals.K_LEFT, mod=pygame.locals.KMOD_NONE)# pg.event.Event(pg.KEYDOWN,key=pg.K_LEFT,mod=pg.KMOD_NONE)
                pg.event.post(newevent)
                #print("************************************")
                # print(newevent)    
            elif actn == 2:#1073741903,
                
                newevent = pygame.event.Event(pygame.locals.KEYDOWN, key=pygame.locals.K_RIGHT, mod=pygame.locals.KMOD_NONE)# pg.event.Event(pg.KEYDOWN, key=pg.K_RIGHT, mod=pg.KMOD_NONE)
                pg.event.post(newevent)
            # else:
                
                # print(newevent)
            
            
        if crashed:

            break
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
        #print("her4",car_chosen,crashed)

        if car_chosen and not crashed:

            #print("her6")

            if time_passed < 400:
                # police_siren.play(0)
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
                #print("her7")

                # if time_passed > 400:
                #     police_siren.stop()
                # if time_passed > 6200:
                #     play_music()

                player.x += dx

                if player.x < 250:
                    player.x = 250
                if player.x > 775:     #DISPLAY_WIDTH - 285:
                    player.x =775 # DISPLAY_WIDTH - 285

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
                # print(traffic_car)
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
                        #print("9*************************************")
                        print("traffic_car_"+str(k),traffic_car[k].x,traffic_car[k].y)
                        window.blit(traffic_car[k].image, (traffic_car[k].x,traffic_car[k].y))
                if loop:
                    for k in range(8):
                        if(collided_with(traffic_car[k], player)):
                            #print("Crash")
                            crashed = True 
                else:
                    for k in range(i):
                        if(collided_with(traffic_car[k], player)):
                            #print("Crash")
                            crashed = True
                if i>=8:
                    i=0
                    loop = True
                score(counter)
        # else:
            # if not crashed:
            #     choose_a_car()
        reward=50
        if crashed:
            if turn >500:
                print(curr_state)
                print(actn)
                print(Q[curr_state][actn])
            reward=-50
        
        if car_chosen:
            time_passed += clock.get_time()

        pg.display.update()
        clock.tick(CLOCK_TICKRATE)
        print(curr_state,actn,next_state)

        curr_state=next_state
        # pg.display.update()
        # clock.tick(CLOCK_TICKRATE)
        # detect if the current color object is in frame or not


file2.close()    
pg.quit()




'''
gAME SETUP DONE

do object categorisation/detction 
and q learning

y coordinate comapare then increase count 
store the color of the maximun object 
store the moving objects
make a function curr object 


'''