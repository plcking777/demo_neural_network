import numpy as np
import neuralNetwork
import pygame as pg
import sys



RESOLUTION = 28



### NN - START

data = np.array(np.loadtxt('/home/woutv/Documents/numbas/numba0.csv', delimiter=',', dtype=np.uint8))

nn = neuralNetwork.NeuralNetwork([2, 10, 10, 1], 28*28, 0.1)



pg.init()

window_width, window_height = 800, 600
screen = pg.display.set_mode((window_width, window_height))

pg.display.set_caption("Neural network demo")

background_color = (30, 30, 30)

count = 0

running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    screen.fill(background_color)

    count += 1
    inp = np.array([[x%28, int(x/28)] for x in range(28*28)]).T
    nn.set_input(inp)
    nn.forward()
    if count >= 100:
        print('cost: ', np.sum(nn.get_cost(data)))
        count = 0
    nn.backward(data)



    custom_inp = np.array([[x%RESOLUTION, int(x/RESOLUTION)] for x in range(RESOLUTION*RESOLUTION)]).T

    #nn.set_input(custom_inp)
    #nn.forward()
    out = nn.get_output()
    print(out)

    rect_size = 20
    for y in range(RESOLUTION):
        for x in range(RESOLUTION):
            val = max(min(out[0][x + (y*28)], 255), 0)
            pg.draw.rect(screen, (0, val, 0), (x * rect_size, y * rect_size, rect_size, rect_size))


    pg.display.flip()

pg.quit()
sys.exit()
