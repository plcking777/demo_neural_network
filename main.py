import numpy as np
import neuralNetwork
import pygame as pg
import sys



UPSCALE = False

RESOLUTION = 28



### NN - START

data = np.array(np.loadtxt('/home/woutv/Documents/numbas/numba0.csv', delimiter=',', dtype=np.uint8)) / 255


nn = neuralNetwork.NeuralNetwork([2, 32, 5, 1], RESOLUTION*RESOLUTION, 0.01)



pg.init()

window_width, window_height = 800, 600
screen = pg.display.set_mode((window_width, window_height))

pg.display.set_caption("Neural network demo")

background_color = (30, 30, 30)

count = 0

running = True


if UPSCALE:
    custom_inp = np.array([[x / 28.0], [y / 28.0]])
else:
    custom_inp = np.array([[(x%28) / 28.0, int(x/28) / 28.0] for x in range(28*28)]).T

while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    screen.fill(background_color)

    count += 1
    inp = np.array([[(x%28) / 28.0, int(x/28) / 28.0] for x in range(28*28)]).T
    nn.set_input(inp)
    nn.forward()
    if count >= 100:
        print('cost: ', np.sum(nn.get_cost(data)))
        count = 0


        if UPSCALE:
            rect_size = 20
            for y in range(28):
                for x in range(28):
                    custom_inp = np.array([[x / 28.0], [y / 28.0]])
                    nn.set_input(custom_inp)
                    nn.forward()
                    out = nn.get_output()
                    #print(out.shape)

                    val = out[0][0] * 255
                    pg.draw.rect(screen, (0, val, 0), (x * rect_size, y * rect_size, rect_size, rect_size))
        else:
            rect_size = 20

            nn.set_input(custom_inp)
            nn.forward()
            out = nn.get_output()
            #print(out)
            for y in range(28):
                for x in range(28):
                    val = (1 - out[0][(x + y * 28)]) * 255
                    pg.draw.rect(screen, (0, val, 0), (x * rect_size, y * rect_size, rect_size, rect_size))

        pg.display.flip()

    nn.backward(data)


pg.quit()



out = ''

custom_inp = np.array([[(x%28) / 28.0, int(x/28) / 28.0] for x in range(28*28)]).T
nn.set_input(custom_inp)
nn.forward()
nn_out = nn.get_output()
for i in range(28 * 28):
    out += str(min(max(nn_out[0][i], 0), 255)) + ', '

out += '\n\n\n\n\n\n\n\n\n'
out += '\n----------------------------------------------------\n'
out += '\n\n\n\n\n\n\n\n\n'

for i in range(28 * 28):
    out += str(min(max(data[i], 0), 255)) + ', '



with open("OUT.log", "w") as file:
    file.write(out)
file.close()



sys.exit()
